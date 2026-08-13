"""Build deterministic MiniWoB train/dev/held-out seed manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FAMILIES = (
    "book-flight",
    "click-button-sequence",
    "click-checkboxes-large",
    "email-inbox-forward-nl",
    "form-sequence-3",
    "navigate-tree",
    "social-media-all",
    "use-autocomplete",
)
TRANSFER_FAMILIES = (
    "buy-ticket",
    "email-inbox-star-reply",
    "multi-layouts",
    "phone-book",
)
TRAIN_CASES = (
    ("click-button-sequence", 1001),
    ("click-checkboxes-large", 1001),
    ("email-inbox-forward-nl", 1002),
    ("form-sequence-3", 1001),
    ("navigate-tree", 1001),
    ("navigate-tree", 1002),
    ("social-media-all", 1001),
    ("use-autocomplete", 1001),
)
SYSTEM = (
    "You are a browser policy. Call open_task with the supplied task_id. Read its "
    "goal and accessibility tree, then call act with exactly one allowed BrowserGym "
    "bid action string at a time. The MCP tool name must be act and its JSON "
    'argument must look like {"action":"click(\'12\')"}; never call click, fill, '
    "or another browser action as an MCP tool. Actions use literal Python-style syntax: "
    "click('12'), fill('7', 'text'), select_option('9', 'value'), press('4', 'Enter'), "
    "or scroll(0, 200). Never invent names like click_button_12. Call finish only "
    "after the environment reports done."
)


def _row(split: str, family: str, seed: int) -> dict:
    task_id = f"mw-{split}-{family}-{seed}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"task_id={task_id}"},
        ],
        "input_metadata": {
            "row_id": task_id,
            "session_data": {
                "benchmark_version": "miniwob-v1",
                "family_id": family,
                "template_id": f"{split}-{family}",
                "task_name": family,
                "seed": seed,
            },
        },
    }


def _write(name: str, rows: list[dict]) -> None:
    path = ROOT / "evaluator" / "miniwob" / name
    content = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(content)
    digest = hashlib.sha256(content.encode()).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    print(f"{path.relative_to(ROOT)} rows={len(rows)} sha256={digest}")


def main() -> None:
    _write(
        "train_tasks.jsonl",
        [_row("train", family, seed) for family, seed in TRAIN_CASES],
    )
    _write(
        "dev_tasks.jsonl",
        [_row("dev", family, seed) for family in TRAIN_FAMILIES for seed in (1101, 1102)],
    )
    _write(
        "heldout_tasks.jsonl",
        [
            _row("heldout", family, seed)
            for family in (*TRAIN_FAMILIES, *TRANSFER_FAMILIES)
            for seed in (2001, 2002)
        ],
    )


if __name__ == "__main__":
    main()
