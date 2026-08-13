"""Build deterministic multi-page browser RLVR train/dev/held-out tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You are a browser policy. Call open_task with the supplied task_id, use one "
    "browser tool at a time, and call finish only after the task is complete."
)


def _page(
    title: str,
    key: str,
    value: str,
    next_key: str,
    next_path: str,
    *,
    reverse: bool,
) -> str:
    rows = [
        f"<li>selector={key}-old; value=decoy-{value}; next={next_key[::-1]}</li>",
        f"<li>selector={key}; value=<strong>{value}</strong>; next={next_key}</li>",
        f"<li>selector={key}-draft; value=sample-{value}; next={next_key[1:]}x</li>",
    ]
    if reverse:
        rows.reverse()
    return (
        f"<!doctype html><html><body><main><h1>{title}</h1><ul>{''.join(rows)}</ul>"
        f"<a href='{next_path}'>Continue workflow</a></main></body></html>"
    )


def _task(split: str, index: int) -> dict:
    seed = hashlib.sha256(f"browseruse-long-v2:{split}:{index}".encode()).hexdigest()
    task_id = f"lh-{split}-{index:03d}"
    labels = [
        "Owner",
        "Region",
        "Ticket",
        "Queue",
        "Dataset",
        "Code",
        "Cluster",
        "Window",
        "Reviewer",
        "Checksum",
    ]
    fields = [
        (
            label,
            f"k{seed[position * 3 : position * 3 + 3]}",
            f"{label.lower()}-{seed[30 + position * 3 : 33 + position * 3]}",
        )
        for position, label in enumerate(labels)
    ]
    pages: dict[str, str] = {}
    for position, (label, key, value) in enumerate(fields):
        path = f"/stage-{position + 1}"
        next_path = (
            f"/stage-{position + 2}" if position + 1 < len(fields) else "/review"
        )
        next_key = fields[position + 1][1] if position + 1 < len(fields) else "review"
        pages[path] = _page(
            label, key, value, next_key, next_path, reverse=split == "heldout"
        )
    inputs = "".join(
        f"<label>{label} <input aria-label='{label}'></label>" for label, _, _ in fields
    )
    checks = (
        "<label><input type='checkbox' aria-label='Evidence reviewed'>Evidence reviewed</label>"
        "<label><input type='checkbox' aria-label='Change authorized'>Change authorized</label>"
    )
    assertions = "&&".join(
        f'document.querySelector(`[aria-label="{label}"]`).value==={json.dumps(value)}'
        for label, _, value in fields
    )
    pages["/review"] = (
        "<!doctype html><html><body><main><h1>Final workflow review</h1>"
        f"{checks if split == 'heldout' else inputs}{inputs if split == 'heldout' else checks}"
        "<button type='button'>Submit verified workflow</button><p id='result'></p>"
        "</main><script>document.querySelector('button').onclick=()=>{"
        "const ok=" + assertions + "&&"
        'document.querySelector(`[aria-label="Evidence reviewed"]`).checked&&'
        'document.querySelector(`[aria-label="Change authorized"]`).checked;'
        "const r=document.querySelector('#result');r.textContent=ok?'Completed':'Incorrect';"
        "r.dataset.success=String(ok)}</script></body></html>"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    f"task_id={task_id}\nStart with selector {fields[0][1]}. On every "
                    "workflow page, choose the record whose selector exactly matches the "
                    "current selector, retain its value, and carry its next selector to the "
                    "following page. Enter all ten retained values on the final review, "
                    "enable both required acknowledgements, and submit."
                ),
            },
        ],
        "input_metadata": {
            "row_id": task_id,
            "session_data": {
                "benchmark_version": "long-v2",
                "family_id": "cross-page-selector-chain",
                "template_id": f"long-{split}-template",
                "oracle_steps": 25,
                "pages": pages,
                "start_path": "/stage-1",
                "verify_selector": "#result",
                "verify_attribute": "data-success",
                "verify_value": "true",
            },
        },
    }


def _write(name: str, rows: list[dict]) -> None:
    path = ROOT / "evaluator" / name
    content = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(content)
    digest = hashlib.sha256(content.encode()).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    print(f"{path.relative_to(ROOT)} rows={len(rows)} sha256={digest}")


def main() -> None:
    _write("long_train_tasks.jsonl", [_task("train", i) for i in range(1, 17)])
    _write("long_dev_tasks.jsonl", [_task("dev", i) for i in range(1, 9)])
    _write("long_heldout_tasks.jsonl", [_task("heldout", i) for i in range(1, 25)])


if __name__ == "__main__":
    main()
