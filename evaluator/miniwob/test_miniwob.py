"""Fireworks-hosted MiniWoB evaluator."""

import os
import sys
from pathlib import Path

from eval_protocol import EvaluateResult, EvaluationRow
from eval_protocol.pytest import evaluation_test

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rollout import (  # noqa: E402
    SamplingAgentRolloutProcessor,
    score_terminal_result,
    terminal_result,
)

MODEL = os.environ.get(
    "MINIWOB_EVAL_MODEL",
    "fireworks_ai/accounts/fireworks/models/qwen3-vl-8b-instruct",
)
DATASET = os.environ.get("MINIWOB_EVAL_DATASET", str(HERE / "train_tasks.jsonl"))
MAX_STEPS = int(os.environ.get("MINIWOB_MAX_STEPS", "20"))


def reward_contract() -> dict[str, float]:
    scores = {
        "success": score_terminal_result(
            {"terminal": True, "done": True, "task_id": "task", "reward": 1.0},
            "task",
        ),
        "failure": score_terminal_result(
            {"terminal": True, "done": True, "task_id": "task", "reward": 0.0},
            "task",
        ),
    }
    assert scores == {"success": 1.0, "failure": 0.0}
    return scores


@evaluation_test(
    input_dataset=[DATASET],
    completion_params=[{"model": MODEL, "max_tokens": 256, "temperature": 0.8}],
    rollout_processor=SamplingAgentRolloutProcessor(),
    mcp_config_path=str(HERE / "mcp.json"),
    max_concurrent_rollouts=1,
    max_concurrent_evaluations=1,
    steps=MAX_STEPS,
    mode="pointwise",
    disable_browser_open=True,
)
def test_miniwob(row: EvaluationRow) -> EvaluationRow:
    task_id = row.input_metadata.row_id
    row.evaluation_result = EvaluateResult(
        score=score_terminal_result(terminal_result(row), task_id),
        reason=f"native MiniWoB terminal reward for {task_id}",
    )
    return row
