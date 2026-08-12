"""Validate, launch, and watch one managed Fireworks RFT config."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import tomllib
import warnings
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

warnings.filterwarnings(
    "ignore",
    message="Support for class-based `config` is deprecated.*",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TERMINAL_STATES = {
    "JOB_STATE_COMPLETED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

SECTIONS = {
    "run": {"job_id"},
    "fireworks": {"account_id"},
    "evaluator": {"id", "entry", "secret_env"},
    "dataset": {"path", "rows"},
    "evaluation": {"path", "rows"},
    "model": {"base_model", "output_model"},
    "rollout": {
        "chunk_size",
        "max_concurrent_evaluations",
        "max_concurrent_rollouts",
        "max_output_tokens",
        "response_candidates_count",
        "temperature",
    },
    "training": {"epochs", "lora_rank", "loss_method", "max_context_length"},
    "wandb": {"api_key_env", "enabled", "entity", "project"},
    "monitor": {"interval_seconds", "no_progress_seconds"},
}


def _positive(config: dict, section: str, key: str) -> int:
    value = config[section][key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{section}.{key} must be a positive integer")
    return value


def _load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            metadata = row.get("input_metadata")
            session = (
                metadata.get("session_data") if isinstance(metadata, dict) else None
            )
            if not isinstance(row.get("messages"), list) or not isinstance(
                session, dict
            ):
                raise TypeError(
                    f"invalid browser task contract at {path}:{line_number}"
                )
            required = {"html", "verify_selector", "verify_attribute", "verify_value"}
            if not isinstance(metadata.get("row_id"), str) or required - session.keys():
                raise ValueError(
                    f"incomplete browser task contract at {path}:{line_number}"
                )
            rows.append(row)
    row_ids = [row["input_metadata"]["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("dataset input_metadata.row_id values must be unique")
    return rows


def _load_evaluator(entry: str):
    entry_path, separator, function_name = entry.partition("::")
    path = PROJECT_ROOT / entry_path
    if not separator or not function_name or not path.is_file():
        raise ValueError("evaluator.entry must be an existing path::function")
    spec = importlib.util.spec_from_file_location("browseruse_rft_evaluator", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(module)
    evaluator = getattr(module, function_name, None)
    if evaluator is None:
        raise ValueError(f"evaluator function not found: {entry}")
    return module, evaluator


def _validate_evaluator(config: dict) -> None:
    module, evaluator = _load_evaluator(config["evaluator"]["entry"])
    params = getattr(evaluator, "__ep_params__", None)
    completions = getattr(params, "completion_params", None)
    if not params or not isinstance(completions, list) or len(completions) != 1:
        raise ValueError("evaluator must declare exactly one completion configuration")
    completion = completions[0]
    rollout = config["rollout"]
    expected_model = f"fireworks_ai/{config['model']['base_model']}"
    expected = {
        "model": expected_model,
        "max_tokens": rollout["max_output_tokens"],
        "temperature": rollout["temperature"],
    }
    if any(completion.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "evaluator completion model/tokens/temperature drifted from TOML"
        )
    if getattr(params, "input_dataset", None) != [config["dataset"]["path"]]:
        raise ValueError("evaluator input_dataset drifted from TOML")
    if (
        getattr(params, "max_concurrent_rollouts", None)
        != rollout["max_concurrent_rollouts"]
    ):
        raise ValueError("evaluator max_concurrent_rollouts drifted from TOML")
    if (
        getattr(params, "max_concurrent_evaluations", None)
        != rollout["max_concurrent_evaluations"]
    ):
        raise ValueError("evaluator max_concurrent_evaluations drifted from TOML")
    reward_contract = getattr(module, "reward_contract", None)
    if not callable(reward_contract):
        raise TypeError("evaluator must expose reward_contract()")
    scores = reward_contract()
    if set(scores.values()) != {0.0, 1.0}:
        raise ValueError("reward calibration must contain both zero and full credit")


def load_rft_config(path: Path) -> dict:
    """Load the executable config and reject drift before any mutation."""
    with path.open("rb") as source:
        config = tomllib.load(source)
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    unknown_sections = set(config) - {"schema_version", *SECTIONS}
    if unknown_sections:
        raise ValueError(f"unknown config sections: {sorted(unknown_sections)}")
    for section, allowed in SECTIONS.items():
        values = config.get(section)
        if not isinstance(values, dict):
            raise TypeError(f"missing [{section}] section")
        unknown = set(values) - allowed
        missing = allowed - set(values)
        if unknown or missing:
            raise ValueError(
                f"[{section}] missing={sorted(missing)} unknown={sorted(unknown)}"
            )

    dataset = PROJECT_ROOT / config["dataset"]["path"]
    if not dataset.is_file():
        raise ValueError(f"dataset not found: {dataset}")
    rows = _load_dataset(dataset)
    configured_rows = _positive(config, "dataset", "rows")
    if len(rows) != configured_rows:
        raise ValueError(f"dataset.rows is {configured_rows}, found {len(rows)}")
    evaluation = PROJECT_ROOT / config["evaluation"]["path"]
    if not evaluation.is_file():
        raise ValueError(f"evaluation dataset not found: {evaluation}")
    evaluation_rows = _load_dataset(evaluation)
    configured_evaluation_rows = _positive(config, "evaluation", "rows")
    if len(evaluation_rows) != configured_evaluation_rows:
        raise ValueError(
            f"evaluation.rows is {configured_evaluation_rows}, found {len(evaluation_rows)}"
        )
    training_ids = {row["input_metadata"]["row_id"] for row in rows}
    evaluation_ids = {row["input_metadata"]["row_id"] for row in evaluation_rows}
    if training_ids & evaluation_ids:
        raise ValueError("training and evaluation row IDs must be disjoint")

    for section, key in (
        ("rollout", "chunk_size"),
        ("rollout", "max_concurrent_evaluations"),
        ("rollout", "max_concurrent_rollouts"),
        ("rollout", "max_output_tokens"),
        ("rollout", "response_candidates_count"),
        ("training", "epochs"),
        ("training", "lora_rank"),
        ("training", "max_context_length"),
        ("monitor", "interval_seconds"),
        ("monitor", "no_progress_seconds"),
    ):
        _positive(config, section, key)
    if config["rollout"]["chunk_size"] > configured_rows:
        raise ValueError("rollout.chunk_size cannot exceed dataset.rows")
    if config["rollout"]["response_candidates_count"] < 2:
        raise ValueError("rollout.response_candidates_count must be at least 2")
    temperature = config["rollout"]["temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= temperature <= 2
    ):
        raise ValueError("rollout.temperature must be between 0 and 2")
    if config["training"]["loss_method"] not in {"grpo", "dapo", "gspo-token"}:
        raise ValueError("training.loss_method must be grpo, dapo, or gspo-token")

    identifier = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
    for section, key in (
        ("run", "job_id"),
        ("evaluator", "id"),
        ("model", "output_model"),
    ):
        if not identifier.fullmatch(config[section][key]):
            raise ValueError(f"{section}.{key} must be a lowercase Fireworks ID")
    if len(f"{config['run']['job_id']}-wandb-key") > 63:
        raise ValueError("run.job_id is too long for Fireworks' derived W&B secret ID")
    if not re.fullmatch(
        r"accounts/fireworks/models/[a-z0-9-]+", config["model"]["base_model"]
    ):
        raise ValueError("model.base_model must be a full public Fireworks model path")
    if not re.fullmatch(r"[a-z0-9-]+", config["fireworks"]["account_id"]):
        raise ValueError("fireworks.account_id is invalid")
    if config["evaluator"]["secret_env"] not in {
        "BROWSERUSE_API_KEY",
        "BROWSER_USE_API_KEY",
    }:
        raise ValueError("evaluator.secret_env must name the Browser Use API key")
    wandb = config["wandb"]
    if wandb["enabled"] is not True or wandb["api_key_env"] != "WANDB_API_KEY":
        raise ValueError('W&B must be enabled with api_key_env="WANDB_API_KEY"')
    if not all(
        isinstance(wandb[key], str) and wandb[key] for key in ("entity", "project")
    ):
        raise ValueError("wandb.entity and wandb.project are required")
    _validate_evaluator(config)
    return config


def build_rft_command(
    config: dict,
    env_file: str = "<browser-key-env>",
    wandb_api_key: str = "<wandb-key>",
) -> list[str]:
    """Map reviewed TOML values directly to the installed Eval Protocol CLI."""
    rollout = config["rollout"]
    training = config["training"]
    wandb = config["wandb"]
    return [
        shutil.which("ep") or "ep",
        "create",
        "rft",
        "--evaluator",
        config["evaluator"]["id"],
        "--job-id",
        config["run"]["job_id"],
        "--base-model",
        config["model"]["base_model"],
        "--output-model",
        config["model"]["output_model"],
        "--epochs",
        str(training["epochs"]),
        "--lora-rank",
        str(training["lora_rank"]),
        "--max-context-length",
        str(training["max_context_length"]),
        "--method",
        training["loss_method"],
        "--chunk-size",
        str(rollout["chunk_size"]),
        "--max-concurrent-evaluations",
        str(rollout["max_concurrent_evaluations"]),
        "--max-concurrent-rollouts",
        str(rollout["max_concurrent_rollouts"]),
        "--max-output-tokens",
        str(rollout["max_output_tokens"]),
        "--response-candidates-count",
        str(rollout["response_candidates_count"]),
        "--temperature",
        str(rollout["temperature"]),
        "--wandb",
        "--wandb-api-key",
        wandb_api_key,
        "--wandb-entity",
        wandb["entity"],
        "--wandb-project",
        wandb["project"],
        "--env-file",
        env_file,
    ]


def build_evaluation_command(config: dict) -> list[str]:
    return [
        shutil.which("ep") or "ep",
        "local-test",
        "--entry",
        config["evaluator"]["entry"],
        "--yes",
    ]


def _run_json(command: list[str]) -> dict:
    result = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "command failed"
        raise ValueError(detail)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"expected JSON from {command[0]}") from error


def _live_preflight(config: dict) -> None:
    firectl = shutil.which("firectl")
    if not firectl:
        raise ValueError("firectl is required for live model/account preflight")
    whoami = subprocess.run(
        [firectl, "whoami"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    expected_account = config["fireworks"]["account_id"]
    if whoami.returncode or f"Account ID: {expected_account}" not in whoami.stdout:
        raise ValueError(f"firectl is not signed into account {expected_account}")
    model_id = config["model"]["base_model"].rsplit("/", 1)[-1]
    model = _run_json(
        [firectl, "model", "get", "-a", "fireworks", model_id, "-o", "json"]
    )
    if (
        model.get("name") != config["model"]["base_model"]
        or model.get("state") != "READY"
    ):
        raise ValueError("configured base model is not READY")
    if not (model.get("rl_lora_tunable") or model.get("rl_tunable")):
        raise ValueError("configured base model is not RL LoRA tunable")


def _field(values: dict, snake: str, camel: str, default="—"):
    return values.get(snake, values.get(camel, default))


def _job_table(job: dict) -> tuple[Table, float, tuple]:
    progress = _field(job, "job_progress", "jobProgress", {})
    wandb = _field(job, "wandb_config", "wandbConfig", {})
    state = job.get("state", "UNKNOWN")
    percent = float(_field(progress, "percent", "percent", 0) or 0)
    processed = _field(
        progress, "total_processed_requests", "totalProcessedRequests", 0
    )
    total = _field(progress, "total_input_requests", "totalInputRequests", 0)
    input_tokens = _field(progress, "input_tokens", "inputTokens", 0)
    output_tokens = _field(progress, "output_tokens", "outputTokens", 0)
    table = Table(title="Fireworks managed RFT", show_header=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("State", str(state))
    table.add_row("Epoch", str(_field(progress, "epoch", "epoch")))
    table.add_row("Requests", f"{processed}/{total}")
    table.add_row("Tokens", f"{input_tokens} in / {output_tokens} out")
    table.add_row("W&B", str(_field(wandb, "url", "url")))
    metrics = _field(job, "output_metrics", "outputMetrics", "")
    if metrics:
        table.add_row("Metrics", str(metrics)[:180])
    signature = (state, percent, processed, total, input_tokens, output_tokens, metrics)
    return table, percent, signature


def watch_rft(path: Path, *, once: bool = False) -> dict:
    """Render authoritative Fireworks state until terminal or no progress."""
    config = load_rft_config(path)
    load_dotenv(PROJECT_ROOT / ".env")
    firectl = shutil.which("firectl")
    if not firectl:
        raise ValueError("firectl is required to watch RFT progress")
    command = [
        firectl,
        "rftj",
        "get",
        "-a",
        config["fireworks"]["account_id"],
        config["run"]["job_id"],
        "-o",
        "json",
    ]
    monitor = config["monitor"]
    rich_progress = Progress(
        TextColumn("[bold]Job progress"), BarColumn(), TaskProgressColumn()
    )
    task = rich_progress.add_task("rft", total=100)
    console = Console()
    last_signature = None
    last_change = time.monotonic()
    with Live(console=console, refresh_per_second=4) as live:
        while True:
            job = _run_json(command)
            table, percent, signature = _job_table(job)
            rich_progress.update(task, completed=max(0, min(percent, 100)))
            live.update(Group(table, rich_progress))
            if signature != last_signature:
                last_signature = signature
                last_change = time.monotonic()
            if once or job.get("state") in TERMINAL_STATES:
                return job
            if time.monotonic() - last_change >= monitor["no_progress_seconds"]:
                console.print(
                    f"[yellow]No authoritative progress for {monitor['no_progress_seconds']}s; "
                    "job was not cancelled.[/yellow]"
                )
                return job
            time.sleep(monitor["interval_seconds"])


def evaluate_rft(path: Path, *, model: str, confirm: bool) -> None:
    """Plan or run the same held-out verifier against base or tuned policy."""
    config = load_rft_config(path)
    load_dotenv(PROJECT_ROOT / ".env")
    model_path = (
        config["model"]["base_model"]
        if model == "base"
        else f"accounts/{config['fireworks']['account_id']}/models/{config['model']['output_model']}"
    )
    command = build_evaluation_command(config)
    print(
        f"model={model_path} heldout={config['evaluation']['path']} "
        f"rows={config['evaluation']['rows']} metric=mean_binary_reward"
    )
    print(shlex.join(command))
    if not confirm:
        print("plan only; pass --confirm to run paid held-out model evaluation")
        return
    required = ("FIREWORKS_API_KEY", config["evaluator"]["secret_env"])
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise ValueError(f"missing required secrets: {', '.join(missing)}")
    environment = os.environ.copy()
    environment.update(
        {
            "BROWSERUSE_EVAL_DATASET": config["evaluation"]["path"],
            "BROWSERUSE_TASKS_PATH": config["evaluation"]["path"],
            "BROWSERUSE_EVAL_MODEL": f"fireworks_ai/{model_path}",
        }
    )
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
    if result.returncode:
        raise ValueError(
            f"held-out evaluation failed with exit code {result.returncode}"
        )


def run_rft(path: Path, *, confirm: bool, watch: bool = True) -> None:
    """Print a redacted plan or execute it after explicit confirmation."""
    config = load_rft_config(path)
    load_dotenv(PROJECT_ROOT / ".env")
    rollout_count = (
        config["dataset"]["rows"] * config["rollout"]["response_candidates_count"]
    )
    secret_names = (
        "FIREWORKS_API_KEY",
        config["evaluator"]["secret_env"],
        config["wandb"]["api_key_env"],
    )
    print(
        f"account={config['fireworks']['account_id']} model={config['model']['base_model']}"
    )
    print(
        f"dataset={config['dataset']['path']} rows={config['dataset']['rows']} "
        f"bounded_rollouts={rollout_count}"
    )
    print(
        f"evaluator={config['evaluator']['entry']} reward=binary-terminal calibrated=0,1"
    )
    print(f"wandb={config['wandb']['entity']}/{config['wandb']['project']}")
    print(
        "secrets="
        + ",".join(
            f"{name}:{'set' if os.getenv(name) else 'missing'}" for name in secret_names
        )
    )
    print(shlex.join(build_rft_command(config)))
    if not confirm:
        print("plan only; pass --confirm to upload evaluator/dataset and start RFT")
        return
    missing = [name for name in secret_names if not os.getenv(name)]
    if missing:
        raise ValueError(f"missing required secrets: {', '.join(missing)}")
    _live_preflight(config)
    browser_key = os.environ[config["evaluator"]["secret_env"]]
    wandb_key = os.environ[config["wandb"]["api_key_env"]]
    launch_environment = os.environ.copy()
    for name in (
        "BROWSERUSE_EVAL_DATASET",
        "BROWSERUSE_EVAL_MODEL",
        "BROWSERUSE_TASKS_PATH",
    ):
        launch_environment.pop(name, None)
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="browseruse-rft-", suffix=".env"
    ) as env_file:
        env_file.write(f"BROWSER_USE_API_KEY={browser_key}\n")
        env_file.flush()
        result = subprocess.run(
            build_rft_command(config, env_file.name, wandb_key),
            cwd=PROJECT_ROOT,
            env=launch_environment,
            check=False,
        )
    if result.returncode:
        raise ValueError(
            f"Eval Protocol create rft failed with exit code {result.returncode}"
        )
    if watch:
        watch_rft(path)
