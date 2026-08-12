"""Fireworks-hosted browser evaluator."""

import json

from dotenv import load_dotenv
from eval_protocol import EvaluateResult, EvaluationRow
from eval_protocol.pytest import AgentRolloutProcessor, evaluation_test

load_dotenv()


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


@evaluation_test(
    input_dataset=["evaluator/tasks.jsonl"],
    completion_params=[
        {
            "model": "fireworks_ai/accounts/fireworks/models/qwen3-4b",
            "max_tokens": 256,
            "temperature": 0.8,
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
    reward = float(result.get("reward", 0.0)) if result.get("task_id") == task_id else 0.0
    row.evaluation_result = EvaluateResult(
        score=reward,
        reason=f"terminal browser verifier for {task_id}",
    )
    return row
