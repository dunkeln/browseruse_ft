"""Run model-driven browser rollouts from a declarative config."""

import asyncio
import json
import os
import tomllib
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from fireworks import AsyncFireworks
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from browseruse_ft.data import load_task_split
from browseruse_ft.policy import fireworks_policy
from browseruse_ft.world.rollout import run_rollout
from browseruse_ft.world.task import resolve_task

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_config(path: Path) -> dict:
    with path.open("rb") as source:
        config = tomllib.load(source)
    task_ids = config["dataset"]["task_ids"]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("dataset.task_ids must contain unique task IDs")
    if config["rollout"]["max_steps"] < 1:
        raise ValueError("rollout.max_steps must be positive")
    if config["sampling"]["max_tokens"] < 1:
        raise ValueError("sampling.max_tokens must be positive")
    return config


def _metrics(steps: list[dict]) -> dict:
    generations = [step["generation"] for step in steps if step["generation"]]
    return {
        "requests": len(generations),
        "input_tokens": sum(
            generation["usage"].get("prompt_tokens", 0)
            for generation in generations
        ),
        "output_tokens": sum(
            generation["usage"].get("completion_tokens", 0)
            for generation in generations
        ),
        "generation_latency_ms": sum(
            generation["latency_ms"] for generation in generations
        ),
    }


async def _run(config: dict) -> Path:
    settings = config["fireworks"]
    dataset = load_task_split(config["dataset"]["split"])
    tasks = {task["task_id"]: task for task in dataset}
    missing = set(config["dataset"]["task_ids"]) - tasks.keys()
    if missing:
        raise ValueError(f"task IDs are not in the selected split: {sorted(missing)}")

    output = PROJECT_ROOT / config["output"]["path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    load_dotenv(find_dotenv())
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is required")

    headers = {
        header: value
        for header, variable in {
            "X-Fireworks-Client-Source": "FIREWORKS_CLIENT_SOURCE",
            "X-Fireworks-Session-Id": "FIREWORKS_SESSION_ID",
        }.items()
        if (value := os.environ.get(variable))
    }
    async with AsyncFireworks(
        api_key=api_key,
        account_id=settings["account_id"],
        base_url=settings.get("base_url"),
        default_headers=headers or None,
    ) as client:
        policy = fireworks_policy(client, model=settings["model"], **config["sampling"])
        with output.open("w") as destination, Progress(
            TextColumn("{task.description}"), BarColumn(), TaskProgressColumn()
        ) as progress:
            progress_task = progress.add_task(
                "model rollouts", total=len(config["dataset"]["task_ids"])
            )
            for task_id in config["dataset"]["task_ids"]:
                task, storage_state = resolve_task(tasks[task_id], root=PROJECT_ROOT)
                rollout = await run_rollout(
                    task,
                    policy,
                    storage_state=storage_state,
                    **config["rollout"],
                )
                row = {
                    "model": settings["model"],
                    "sampling": config["sampling"],
                    "task": task,
                    **rollout,
                    "metrics": _metrics(rollout["steps"]),
                }
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                destination.flush()
                progress.advance(progress_task)
    return output


def run_model_rollouts(config_path: Path, *, confirm: bool = False) -> Path | None:
    """Run and persist model-driven browser trajectories."""
    config = _load_config(config_path)
    if not confirm:
        settings = config["fireworks"]
        dataset = config["dataset"]
        rollout = config["rollout"]
        sampling = config["sampling"]
        output = PROJECT_ROOT / config["output"]["path"]
        print(
            "preflight=ok paid_inference=false "
            f"(model={settings['model']}, split={dataset['split']}, "
            f"tasks={len(dataset['task_ids'])}, max_steps={rollout['max_steps']}, "
            f"max_tokens={sampling['max_tokens']}, output={output})"
        )
        return None
    return asyncio.run(_run(config))
