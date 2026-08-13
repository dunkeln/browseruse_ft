"""Minimal sampled MCP rollout used by the hosted MiniWoB evaluator."""

import asyncio
import json
import os
import time

from eval_protocol import EvaluateResult, EvaluationRow
from eval_protocol.mcp.execution.policy import LiteLLMPolicy
from eval_protocol.pytest.default_agent_rollout_processor import Agent
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.pytest.utils import normalize_fireworks_model_for_litellm
from mcp.types import TextContent
from openai.types import CompletionUsage


class SamplingAgent(Agent):
    def __init__(self, *, completion_params: dict, steps: int, **kwargs):
        super().__init__(**kwargs)
        self._policy = LiteLLMPolicy(
            model_id=completion_params["model"],
            temperature=completion_params.get("temperature", 0.8),
            max_tokens=completion_params.get("max_tokens", 256),
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

    async def _call_model(self, messages, tools):
        """Keep the task opening and recent observations in the live prompt."""
        tool_indexes = [
            index for index, message in enumerate(messages) if message.role == "tool"
        ]
        keep = set(tool_indexes[:1] + tool_indexes[-3:])
        prompt = [
            message.model_copy(update={"content": '{"compacted":true}'})
            if message.role == "tool" and index not in keep
            else message
            for index, message in enumerate(messages)
        ]
        return await super()._call_model(prompt, tools)

    async def _execute_tool_call(self, tool_call_id, tool_name, tool_args_dict):
        try:
            return await super()._execute_tool_call(
                tool_call_id, tool_name, tool_args_dict
            )
        except KeyError:
            return tool_call_id, [
                TextContent(
                    type="text",
                    text=(
                        f"Unknown tool {tool_name!r}. Call the MCP tool 'act' with "
                        f'JSON like {{"action":"{tool_name}(...)"}}.'
                    ),
                )
            ]


class SamplingAgentRolloutProcessor(RolloutProcessor):
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
                    checkpoint(row)
                    return row
                finally:
                    if agent.mcp_client:
                        await agent.mcp_client.cleanup()

        return [asyncio.create_task(process_row(row)) for row in rows]


def terminal_result(row: EvaluationRow) -> dict:
    for message in reversed(row.messages):
        if message.role != "tool" or not isinstance(message.content, str):
            continue
        try:
            result = json.loads(message.content)
            if isinstance(result, dict) and isinstance(result.get("result"), str):
                result = json.loads(result["result"])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and "done" in result:
            return result
    return {"done": False}


def score_terminal_result(result: dict, task_id: str) -> float:
    return float(
        result.get("terminal") is True
        and result.get("done") is True
        and result.get("task_id") == task_id
        and result.get("reward") == 1.0
    )


def checkpoint(row: EvaluationRow) -> None:
    output_dir = os.environ.get("EP_OUTPUT_DIR")
    if not output_dir:
        return
    task_id = row.input_metadata.row_id
    row.evaluation_result = EvaluateResult(
        score=score_terminal_result(terminal_result(row), task_id),
        reason=f"native MiniWoB terminal reward for {task_id}",
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "completed_rollouts.jsonl"), "a", encoding="utf-8"
    ) as output:
        output.write(row.model_dump_json() + "\n")
