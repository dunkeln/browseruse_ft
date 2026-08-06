"""Run a managed-SFT smoke from declarative config."""

import json
import os
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from fireworks import Fireworks, NotFoundError
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/sft-smoke.toml"
TERMINAL_STATES = {
    "JOB_STATE_CANCELLED",
    "JOB_STATE_COMPLETED",
    "JOB_STATE_DELETED",
    "JOB_STATE_EARLY_STOPPED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_FAILED",
}


def _load_config() -> dict:
    with CONFIG_PATH.open("rb") as source:
        return tomllib.load(source)


def _validate_dataset(config: dict) -> int:
    path = PROJECT_ROOT / config["dataset"]["path"]
    rows = 0
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            roles = [message.get("role") for message in row.get("messages", [])]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(f"line {line_number}: unexpected roles {roles}")
            if any(not message.get("content") for message in row["messages"]):
                raise ValueError(f"line {line_number}: empty message content")
            rows += 1
    expected = config["dataset"]["rows"]
    if rows != expected:
        raise ValueError(f"expected {expected} smoke rows, found {rows}")
    return rows


def _client(config: dict) -> Fireworks:
    load_dotenv(find_dotenv())
    api_key = os.environ.get("FIREWORKS_API_KEY")
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not api_key or not wandb_key:
        raise RuntimeError("FIREWORKS_API_KEY and WANDB_API_KEY are required")

    headers = {
        header: value
        for header, variable in {
            "X-Fireworks-Client-Source": "FIREWORKS_CLIENT_SOURCE",
            "X-Fireworks-Session-Id": "FIREWORKS_SESSION_ID",
        }.items()
        if (value := os.environ.get(variable))
    }
    return Fireworks(
        api_key=api_key,
        account_id=config["fireworks"]["account_id"],
        default_headers=headers or None,
    )


def _ensure_dataset(client: Fireworks, config: dict, rows: int) -> str:
    settings = config["fireworks"]
    dataset_id = settings["dataset_id"]
    try:
        dataset = client.datasets.get(dataset_id)
    except NotFoundError:
        dataset = client.datasets.create(
            dataset_id=dataset_id,
            dataset={
                "display_name": dataset_id,
                "example_count": str(rows),
                "format": "CHAT",
                "user_uploaded": {},
            },
        )
        with (PROJECT_ROOT / config["dataset"]["path"]).open("rb") as source:
            client.datasets.upload(dataset_id, file=source)

    deadline = time.monotonic() + 120
    while dataset.state != "READY" and time.monotonic() < deadline:
        time.sleep(5)
        dataset = client.datasets.get(dataset_id)
    if dataset.state != "READY":
        raise RuntimeError(f"dataset did not become READY: {dataset.state}")
    if dataset.example_count not in {None, str(rows)}:
        raise RuntimeError(
            f"existing dataset has {dataset.example_count} rows, expected {rows}"
        )
    return f"accounts/{settings['account_id']}/datasets/{dataset_id}"


def _create_job(client: Fireworks, config: dict, dataset: str):
    settings = config["fireworks"]
    job_id = settings["job_id"]
    try:
        job = client.supervised_fine_tuning_jobs.get(job_id)
    except NotFoundError:
        return client.supervised_fine_tuning_jobs.create(
            supervised_fine_tuning_job_id=job_id,
            dataset=dataset,
            base_model=settings["base_model"],
            output_model=settings["output_model"],
            **config["training"],
            wandb_config={
                **config["wandb"],
                "api_key": os.environ["WANDB_API_KEY"],
            },
        )

    expected = {
        "dataset": dataset,
        "base_model": settings["base_model"],
        **config["training"],
    }
    mismatches = {
        field: (getattr(job, field), value)
        for field, value in expected.items()
        if getattr(job, field) != value
    }
    if mismatches:
        raise RuntimeError(f"existing job config differs: {mismatches}")
    return job


def _delete_active_job(client: Fireworks, job_id: str) -> None:
    job = client.supervised_fine_tuning_jobs.get(job_id)
    if job.state in TERMINAL_STATES:
        return
    client.supervised_fine_tuning_jobs.delete(job_id)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            state = client.supervised_fine_tuning_jobs.get(job_id).state
        except NotFoundError:
            state = "JOB_STATE_DELETED"
        print(f"{datetime.now(timezone.utc).isoformat()} state={state}")
        if state in TERMINAL_STATES:
            return
        time.sleep(10)
    raise RuntimeError("job deletion was not confirmed within 60 seconds")


def run_smoke(*, confirm: bool = False):
    """Upload, train, observe, and delete the configured managed-SFT smoke."""
    config = _load_config()
    rows = _validate_dataset(config)
    if not confirm:
        raise ValueError("paid Fireworks mutation requires confirm=True")

    job_id = config["fireworks"]["job_id"]
    monitor = config["monitor"]
    with _client(config) as client:
        dataset = _ensure_dataset(client, config, rows)
        job = _create_job(client, config, dataset)
        deadline = time.monotonic() + monitor["seconds"]
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(job.state, total=100)
                while True:
                    job = client.supervised_fine_tuning_jobs.get(job_id)
                    percent = getattr(job.job_progress, "percent", 0) or 0
                    progress.update(task, completed=percent, description=job.state)
                    if job.state in TERMINAL_STATES or time.monotonic() >= deadline:
                        break
                    time.sleep(min(monitor["interval"], deadline - time.monotonic()))
            return job
        finally:
            if job.state not in TERMINAL_STATES:
                _delete_active_job(client, job_id)
