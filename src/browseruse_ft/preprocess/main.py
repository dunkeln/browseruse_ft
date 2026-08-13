"""Apply the current dataset admission and normalization policy."""

import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset

from browseruse_ft.preprocess.policy import render_policy_messages


def _canonical_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def preprocess(dataset: Dataset) -> tuple[Dataset, Dataset, dict[str, int]]:
    """Normalize accepted rows and quarantine contradictory supervision."""
    unique_rows = []
    seen_rows = set()
    targets_by_input = defaultdict(set)
    duplicates = 0

    for source_row, row in enumerate(dataset):
        row_key = _canonical_key(row)
        if row_key in seen_rows:
            duplicates += 1
            continue
        seen_rows.add(row_key)

        input_key = _canonical_key([row["problem"], row["history"], row["observation"]])
        targets_by_input[input_key].add((row["remarks"], row["solution"]))
        unique_rows.append((source_row, input_key, row))

    conflicting_inputs = {
        input_key
        for input_key, targets in targets_by_input.items()
        if len(targets) > 1
    }
    accepted = []
    quarantine = []
    history_steps = 0
    empty_history_remarks = 0

    for source_row, input_key, row in unique_rows:
        if input_key in conflicting_inputs:
            quarantine.append(
                {
                    "quarantine_reason": "same input has multiple targets",
                    "source": "webarena_lite_sft",
                    "source_row": source_row,
                    **row,
                }
            )
            continue

        history = []
        for step in row["history"]:
            remarks = step["remarks"] or None
            history.append({"action": step["action"], "remarks": remarks})
            history_steps += 1
            empty_history_remarks += remarks is None

        accepted.append(
            {
                "source": "webarena_lite_sft",
                "source_row": source_row,
                "task": row["problem"].removeprefix("Task Instruction: ").strip(),
                "history": history,
                "current_observation": row["observation"],
                "target_action": row["solution"],
                "target_remarks": row["remarks"] or None,
            }
        )

    report = {
        "input_rows": dataset.num_rows,
        "exact_duplicates_removed": duplicates,
        "conflicting_groups": len(conflicting_inputs),
        "conflicting_rows_quarantined": len(quarantine),
        "output_rows": len(accepted),
        "history_placeholders_removed": history_steps,
        "rounds_removed": history_steps,
        "empty_history_remarks_omitted": empty_history_remarks,
    }
    return Dataset.from_list(accepted), Dataset.from_list(quarantine), report


def preprocess_sft(dataset: Dataset) -> tuple[Dataset, Dataset, dict[str, int]]:
    """Apply admission policy and render accepted rows as SFT messages."""
    accepted, quarantine, report = preprocess(dataset)
    rendered = accepted.map(
        lambda example: {
            "messages": [
                *render_policy_messages(example),
                {"role": "assistant", "content": example["target_action"]},
            ]
        },
        remove_columns=accepted.column_names,
        desc="Preprocessing SFT messages",
    )
    return rendered, quarantine, report


def preprocess_rollouts(
    results: Path, heldout_tasks: Path, *, max_context_chars: int
) -> tuple[Dataset, Dataset, dict[str, int]]:
    """Materialize complete verified trajectories without held-out leakage."""
    heldout = [json.loads(line) for line in heldout_tasks.read_text().splitlines() if line]
    blocked_ids = {row["input_metadata"]["row_id"] for row in heldout}
    blocked_templates = {
        row["input_metadata"]["session_data"].get("template_id") for row in heldout
    }
    accepted, quarantine = [], []
    for path in sorted(results.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            metadata = row.get("input_metadata", {})
            session = metadata.get("session_data", {})
            messages = row.get("messages", [])
            reasons = []
            if metadata.get("row_id") in blocked_ids or session.get("template_id") in blocked_templates:
                reasons.append("heldout_leakage")
            if row.get("rollout_status", {}).get("code") != 100:
                reasons.append("unfinished_rollout")
            if row.get("evaluation_result", {}).get("is_score_valid") is not True:
                reasons.append("invalid_score")
            terminal = None
            for message in reversed(messages):
                if message.get("role") != "tool" or not isinstance(
                    message.get("content"), str
                ):
                    continue
                try:
                    candidate = json.loads(message["content"])
                    if isinstance(candidate, dict) and isinstance(
                        candidate.get("result"), str
                    ):
                        candidate = json.loads(candidate["result"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(candidate, dict) and candidate.get("terminal") is True:
                    terminal = candidate
                    break
            if terminal is None:
                reasons.append("missing_finish")
            elif float(terminal.get("reward", -1)) != float(
                row.get("evaluation_result", {}).get("score", -2)
            ):
                reasons.append("reward_mismatch")
            serialized_chars = len(_canonical_key(messages))
            if serialized_chars > max_context_chars:
                reasons.append("over_context")
            output = {
                "source_rollout_id": row.get("execution_metadata", {}).get("rollout_id"),
                "task_id": metadata.get("row_id"),
                "family_id": session.get("family_id"),
                "messages": messages,
                "tools": row.get("tools", []),
                "outcome": row.get("evaluation_result"),
                "usage": row.get("execution_metadata", {}).get("usage", {}),
                "rollout_duration_seconds": row.get("execution_metadata", {}).get(
                    "rollout_duration_seconds"
                ),
                "serialized_chars": serialized_chars,
            }
            (quarantine if reasons else accepted).append(
                {**output, **({"quarantine_reasons": reasons} if reasons else {})}
            )
    sizes = sorted(row["serialized_chars"] for row in accepted)
    report = {
        "accepted": len(accepted),
        "quarantined": len(quarantine),
        "context_chars_p95": sizes[min(len(sizes) - 1, int(len(sizes) * 0.95))]
        if sizes
        else 0,
        "context_chars_max": sizes[-1] if sizes else 0,
    }
    return Dataset.from_list(accepted), Dataset.from_list(quarantine), report
