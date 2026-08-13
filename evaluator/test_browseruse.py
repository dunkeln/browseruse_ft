"""Fireworks-hosted browser evaluator."""

import asyncio
import json
import os
import time

from dotenv import load_dotenv
from eval_protocol import EvaluateResult, EvaluationRow
from eval_protocol.mcp.execution.policy import LiteLLMPolicy
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_agent_rollout_processor import Agent
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.pytest.utils import normalize_fireworks_model_for_litellm
from openai.types import CompletionUsage

load_dotenv()

LOCAL_MODEL = os.environ.get(
    "BROWSERUSE_EVAL_MODEL",
    "fireworks_ai/accounts/fireworks/models/qwen3-vl-8b-instruct",
)
INPUT_DATASET = os.environ.get(
    "BROWSERUSE_EVAL_DATASET", "evaluator/long_train_tasks.jsonl"
)
MAX_OUTPUT_TOKENS = 256
TEMPERATURE = 0.8
MAX_STEPS = int(os.environ.get("BROWSERUSE_MAX_STEPS", "12"))


class SamplingAgent(Agent):
    """Agent that honors the managed rollout sampling contract."""

    def __init__(self, *, completion_params: dict, steps: int, **kwargs):
        super().__init__(**kwargs)
        self._policy = LiteLLMPolicy(
            model_id=completion_params["model"],
            temperature=completion_params.get("temperature", TEMPERATURE),
            max_tokens=completion_params.get("max_tokens", MAX_OUTPUT_TOKENS),
            base_url=completion_params.get("base_url"),
            use_caching=False,
            caching=False,
            top_p=completion_params.get("top_p", 1.0),
            extra_headers=completion_params.get("extra_headers"),
        )
        self.remaining_steps = steps

    async def call_agent(self):
        if self.remaining_steps <= 0:
            return None
        self.remaining_steps -= 1
        return await super().call_agent()


class SamplingAgentRolloutProcessor(RolloutProcessor):
    """Run each candidate independently with its requested sampling parameters."""

    def __call__(
        self, rows: list[EvaluationRow], config: RolloutProcessorConfig
    ) -> list[asyncio.Task[EvaluationRow]]:
        async def process_row(row: EvaluationRow) -> EvaluationRow:
            async with config.semaphore:
                started = time.perf_counter()
                completion_params = (
                    normalize_fireworks_model_for_litellm(
                        row.input_metadata.completion_params
                    )
                    or {}
                )
                row.input_metadata.completion_params = completion_params
                agent = SamplingAgent(
                    model=completion_params["model"],
                    row=row,
                    config_path=config.mcp_config_path,
                    logger=config.logger,
                    completion_params=completion_params,
                    steps=config.steps,
                )
                try:
                    await agent.setup()
                    await agent.call_agent()
                    row.execution_metadata.usage = CompletionUsage(**agent.usage)
                    row.execution_metadata.rollout_duration_seconds = (
                        time.perf_counter() - started
                    )
                    _checkpoint_row(row)
                    return row
                finally:
                    if agent.mcp_client:
                        await agent.mcp_client.cleanup()

        return [asyncio.create_task(process_row(row)) for row in rows]


def _terminal_result(row: EvaluationRow) -> dict:
    for message in reversed(row.messages):
        if message.role != "tool" or not isinstance(message.content, str):
            continue
        try:
            result = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and isinstance(result.get("result"), str):
            try:
                result = json.loads(result["result"])
            except json.JSONDecodeError:
                continue
        if "done" in result:
            return result
    return {"done": False}


def _score_terminal_result(result: dict, task_id: str) -> float:
    """Convert only a verified terminal state into binary task success."""
    return float(
        result.get("terminal") is True
        and result.get("done") is True
        and result.get("task_id") == task_id
        and result.get("reward") == 1.0
    )


def _checkpoint_row(row: EvaluationRow) -> None:
    """Keep completed local rollouts durable if the larger eval is interrupted."""
    output_dir = os.environ.get("EP_OUTPUT_DIR")
    if not output_dir:
        return
    task_id = row.input_metadata.row_id
    row.evaluation_result = EvaluateResult(
        score=_score_terminal_result(_terminal_result(row), task_id),
        reason=f"terminal browser verifier for {task_id}",
    )
    path = os.path.join(output_dir, "completed_rollouts.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as checkpoint:
        checkpoint.write(row.model_dump_json() + "\n")


def reward_contract() -> dict[str, float]:
    """Offline calibration used by the run-config preflight."""
    scores = {
        "success": _score_terminal_result(
            {"terminal": True, "done": True, "task_id": "task-1", "reward": 1.0},
            "task-1",
        ),
        "failed_verifier": _score_terminal_result(
            {"done": True, "task_id": "task-1", "reward": 0.0}, "task-1"
        ),
        "wrong_task": _score_terminal_result(
            {"done": True, "task_id": "other", "reward": 1.0}, "task-1"
        ),
        "not_terminal": _score_terminal_result(
            {"done": False, "task_id": "task-1", "reward": 1.0}, "task-1"
        ),
        "observation_is_not_terminal": _score_terminal_result(
            {"done": True, "task_id": "task-1", "reward": 1.0}, "task-1"
        ),
    }
    assert scores == {
        "success": 1.0,
        "failed_verifier": 0.0,
        "wrong_task": 0.0,
        "not_terminal": 0.0,
        "observation_is_not_terminal": 0.0,
    }
    return scores


@evaluation_test(
    input_dataset=[INPUT_DATASET],
    completion_params=[
        {
            "model": LOCAL_MODEL,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
        }
    ],
    rollout_processor=SamplingAgentRolloutProcessor(),
    mcp_config_path="evaluator/mcp.json",
    max_concurrent_rollouts=1,
    max_concurrent_evaluations=1,
    steps=MAX_STEPS,
    mode="pointwise",
    disable_browser_open=True,
)
def test_browseruse(row: EvaluationRow) -> EvaluationRow:
    """Score only the terminal state emitted by the constrained browser tools."""
    result = _terminal_result(row)
    task_id = row.input_metadata.row_id
    reward = _score_terminal_result(result, task_id)
    row.evaluation_result = EvaluateResult(
        score=reward,
        reason=f"terminal browser verifier for {task_id}",
    )
    return row
