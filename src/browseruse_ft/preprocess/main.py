"""Apply the current dataset admission and normalization policy."""

import json
from collections import defaultdict
from hashlib import sha256

from datasets import Dataset

from browseruse_ft.preprocess.policy import render_policy_messages


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return sha256(payload.encode()).hexdigest()


def preprocess(dataset: Dataset) -> tuple[Dataset, Dataset, dict[str, int]]:
    """Normalize accepted rows and quarantine contradictory supervision."""
    unique_rows = []
    seen_rows = set()
    targets_by_input = defaultdict(set)
    duplicates = 0

    for source_row, row in enumerate(dataset):
        row_key = _fingerprint(row)
        if row_key in seen_rows:
            duplicates += 1
            continue
        seen_rows.add(row_key)

        input_key = _fingerprint(
            [row["problem"], row["history"], row["observation"]]
        )
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
