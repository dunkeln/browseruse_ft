"""Fireworks-hosted browser evaluator."""

import json
import os

from dotenv import load_dotenv
from eval_protocol import EvaluateResult, EvaluationRow
from eval_protocol.pytest import AgentRolloutProcessor, evaluation_test

load_dotenv()

LOCAL_MODEL = os.environ.get(
    "BROWSERUSE_EVAL_MODEL",
    "fireworks_ai/accounts/fireworks/models/qwen3-vl-8b-instruct",
)
INPUT_DATASET = os.environ.get("BROWSERUSE_EVAL_DATASET", "evaluator/tasks.jsonl")
MAX_OUTPUT_TOKENS = 256
TEMPERATURE = 0.8


def _terminal_result(row: EvaluationRow) -> dict:
    for message in reversed(row.messages):
        if message.role != "tool" or not isinstance(message.content, str):
            continue
        try:
            result = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if result.get("done") is True:
            return result
    raise ValueError("rollout ended without a finish tool result")


def _score_terminal_result(result: dict, task_id: str) -> float:
    """Convert only a verified terminal state into binary task success."""
    return float(
        result.get("done") is True
        and result.get("task_id") == task_id
        and result.get("reward") == 1.0
    )


def reward_contract() -> dict[str, float]:
    """Offline calibration used by the run-config preflight."""
    scores = {
        "success": _score_terminal_result(
            {"done": True, "task_id": "task-1", "reward": 1.0}, "task-1"
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
    }
    assert scores == {
        "success": 1.0,
        "failed_verifier": 0.0,
        "wrong_task": 0.0,
        "not_terminal": 0.0,
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
    rollout_processor=AgentRolloutProcessor(),
    mcp_config_path="evaluator/mcp.json",
    max_concurrent_rollouts=1,
    max_concurrent_evaluations=1,
    steps=12,
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
