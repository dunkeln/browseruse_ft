"""Run managed SFT from a declarative experiment config."""

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
RUNS_PATH = PROJECT_ROOT / "fireworks-training-runs"
TERMINAL_STATES = {
    "JOB_STATE_CANCELLED",
    "JOB_STATE_COMPLETED",
    "JOB_STATE_DELETED",
    "JOB_STATE_EARLY_STOPPED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_FAILED",
}


def _load_config(path: Path) -> dict:
    with path.open("rb") as source:
        return tomllib.load(source)


def _source_model(config: dict) -> tuple[str, str]:
    settings = config["fireworks"]
    sources = [key for key in ("base_model", "warm_start_from") if settings.get(key)]
    if len(sources) != 1:
        raise ValueError("set exactly one of fireworks.base_model or warm_start_from")
    key = sources[0]
    return key, settings[key]


def _record_run(config: dict, job) -> Path:
    source_kind, source_model = _source_model(config)
    status = getattr(job, "status", None)
    record = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "lineage": {"source_kind": source_kind, "source_model": source_model},
        "job": {
            "name": getattr(job, "name", None),
            "state": job.state,
            "status": getattr(status, "message", None),
            "output_model": getattr(job, "output_model", None),
            "wandb_url": getattr(getattr(job, "wandb_config", None), "url", None),
        },
    }
    if job.state == "JOB_STATE_COMPLETED":
        record["lineage"]["next_warm_start_from"] = job.output_model

    destination = RUNS_PATH / config["fireworks"]["job_id"] / "run.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(destination)
    return destination


def _validate_dataset(dataset: dict) -> int:
    path = PROJECT_ROOT / dataset["path"]
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
    expected = dataset["rows"]
    if rows != expected:
        raise ValueError(f"expected {expected} rows in {path}, found {rows}")
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


def _ensure_dataset(
    client: Fireworks,
    account_id: str,
    dataset_id: str,
    dataset_config: dict,
    rows: int,
) -> str:
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
        with (PROJECT_ROOT / dataset_config["path"]).open("rb") as source:
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
    return f"accounts/{account_id}/datasets/{dataset_id}"


def _create_job(
    client: Fireworks,
    config: dict,
    dataset: str,
    evaluation_dataset: str | None,
):
    settings = config["fireworks"]
    job_id = settings["job_id"]
    source_kind, source_model = _source_model(config)
    try:
        job = client.supervised_fine_tuning_jobs.get(job_id)
    except NotFoundError:
        create = {
            "supervised_fine_tuning_job_id": job_id,
            "dataset": dataset,
            "output_model": settings["output_model"],
            source_kind: source_model,
            **config["training"],
            "wandb_config": {
                **config["wandb"],
                "api_key": os.environ["WANDB_API_KEY"],
            },
        }
        if evaluation_dataset:
            create["evaluation_dataset"] = evaluation_dataset
        return client.supervised_fine_tuning_jobs.create(**create)

    expected = {
        "dataset": dataset,
        source_kind: source_model,
        "output_model": settings["output_model"],
        **config["training"],
    }
    if evaluation_dataset:
        expected["evaluation_dataset"] = evaluation_dataset
    mismatches = {
        field: (getattr(job, field), value)
        for field, value in expected.items()
        if getattr(job, field) != value
    }
    if mismatches:
        raise RuntimeError(f"existing job config differs: {mismatches}")
    return job


def run_sft(config_path: Path, *, confirm: bool = False):
    """Upload, train, and monitor one configured managed SFT run."""
    config = _load_config(config_path)
    train_rows = _validate_dataset(config["dataset"])
    evaluation_rows = (
        _validate_dataset(config["evaluation_dataset"])
        if "evaluation_dataset" in config
        else None
    )
    if not confirm:
        raise ValueError("paid Fireworks mutation requires confirm=True")

    job_id = config["fireworks"]["job_id"]
    monitor = config["monitor"]
    with _client(config) as client:
        settings = config["fireworks"]
        dataset = _ensure_dataset(
            client,
            settings["account_id"],
            settings["dataset_id"],
            config["dataset"],
            train_rows,
        )
        evaluation_dataset = (
            _ensure_dataset(
                client,
                settings["account_id"],
                settings["evaluation_dataset_id"],
                config["evaluation_dataset"],
                evaluation_rows,
            )
            if evaluation_rows is not None
            else None
        )
        job = _create_job(client, config, dataset, evaluation_dataset)
        record_path = _record_run(config, job)
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("epoch {task.fields[epoch]}"),
            TextColumn("requests {task.fields[requests]}"),
            TextColumn("tokens {task.fields[tokens]}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                job.state, total=100, epoch=0, requests="0/0", tokens=0
            )
            if wandb_url := getattr(job.wandb_config, "url", None):
                progress.console.print(f"W&B metrics: {wandb_url}")
            while job.state not in TERMINAL_STATES:
                job = client.supervised_fine_tuning_jobs.get(job_id)
                _record_run(config, job)
                job_progress = job.job_progress
                status = getattr(job.status, "message", "")
                progress.update(
                    task,
                    completed=getattr(job_progress, "percent", 0) or 0,
                    description=status or job.state,
                    epoch=getattr(job_progress, "epoch", 0) or 0,
                    requests=(
                        f"{getattr(job_progress, 'total_processed_requests', 0) or 0}/"
                        f"{getattr(job_progress, 'total_input_requests', 0) or 0}"
                    ),
                    tokens=getattr(job_progress, "input_tokens", 0) or 0,
                )
                if job.state not in TERMINAL_STATES:
                    time.sleep(monitor["interval"])
        _record_run(config, job)
        print(f"Run record: {record_path}")
        return job
