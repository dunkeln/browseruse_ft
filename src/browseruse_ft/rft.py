"""Validate, launch, and watch one managed Fireworks RFT config."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import random
import shlex
import shutil
import subprocess
import sys
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
    "evaluator": {"id", "entry"},
    "dataset": {"path", "rows"},
    "evaluation": {"id", "max_concurrent_rollouts", "num_runs", "path", "rows"},
    "model": {"base_model", "output_model"},
    "rollout": {
        "chunk_size",
        "max_concurrent_evaluations",
        "max_concurrent_rollouts",
        "max_output_tokens",
        "response_candidates_count",
        "steps",
        "temperature",
    },
    "training": {"epochs", "lora_rank", "loss_method", "max_context_length"},
    "wandb": {"api_key_env", "enabled", "entity", "project"},
    "monitor": {"interval_seconds", "no_progress_seconds"},
}
OPTIONAL_FIELDS = {"dataset": {"remote_id"}, "evaluator": {"secret_env"}}


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
            required = {"verify_selector", "verify_attribute", "verify_value"}
            has_content = isinstance(session.get("html"), str) or (
                isinstance(session.get("pages"), dict)
                and isinstance(session.get("start_path"), str)
                and session["start_path"] in session["pages"]
            )
            is_miniwob = isinstance(session.get("task_name"), str) and isinstance(
                session.get("seed"), int
            )
            if (
                not isinstance(metadata.get("row_id"), str)
                or (not is_miniwob and (required - session.keys() or not has_content))
            ):
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
    prefix = (
        "MINIWOB"
        if config["evaluator"]["entry"].startswith("evaluator/miniwob/")
        else "BROWSERUSE"
    )
    overrides = {
        f"{prefix}_EVAL_DATASET": config["dataset"]["path"],
        f"{prefix}_MAX_STEPS": str(config["rollout"]["steps"]),
    }
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        module, evaluator = _load_evaluator(config["evaluator"]["entry"])
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
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
    if getattr(params, "steps", None) != rollout["steps"]:
        raise ValueError("evaluator step budget drifted from TOML")
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
        unknown = set(values) - allowed - OPTIONAL_FIELDS.get(section, set())
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
    training_templates = {
        row["input_metadata"]["session_data"].get(
            "template_id", row["input_metadata"]["row_id"]
        )
        for row in rows
    }
    evaluation_templates = {
        row["input_metadata"]["session_data"].get(
            "template_id", row["input_metadata"]["row_id"]
        )
        for row in evaluation_rows
    }
    if training_templates & evaluation_templates:
        raise ValueError("training and evaluation template IDs must be disjoint")

    for section, key in (
        ("rollout", "chunk_size"),
        ("rollout", "max_concurrent_evaluations"),
        ("rollout", "max_concurrent_rollouts"),
        ("rollout", "max_output_tokens"),
        ("rollout", "response_candidates_count"),
        ("rollout", "steps"),
        ("evaluation", "max_concurrent_rollouts"),
        ("evaluation", "num_runs"),
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
    if config["training"]["loss_method"] != "platform-default":
        raise ValueError("training.loss_method must be platform-default")

    identifier = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
    for section, key in (
        ("run", "job_id"),
        ("evaluator", "id"),
        ("model", "output_model"),
        ("evaluation", "id"),
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
    if config["evaluator"].get("secret_env") not in {
        None,
        "BROWSERUSE_API_KEY",
        "BROWSER_USE_API_KEY",
    }:
        raise ValueError("evaluator.secret_env must be omitted or name the Browser Use API key")
    wandb = config["wandb"]
    if wandb["enabled"] is not True or wandb["api_key_env"] != "WANDB_API_KEY":
        raise ValueError('W&B must be enabled with api_key_env="WANDB_API_KEY"')
    if not all(
        isinstance(wandb[key], str) and wandb[key] for key in ("entity", "project")
    ):
        raise ValueError("wandb.entity and wandb.project are required")
    remote_dataset = config["dataset"].get("remote_id")
    expected_dataset_prefix = f"accounts/{config['fireworks']['account_id']}/datasets/"
    if remote_dataset and not remote_dataset.startswith(expected_dataset_prefix):
        raise ValueError(f"dataset.remote_id must start with {expected_dataset_prefix}")
    _validate_evaluator(config)
    return config


def build_rft_command(
    config: dict,
    env_file: str | None = "<browser-key-env>",
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
        "--skip-validation",
        *(
            ["--dataset", config["dataset"]["remote_id"]]
            if config["dataset"].get("remote_id")
            else []
        ),
        "--evaluator",
        config["evaluator"]["id"],
        "--job-id",
        config["run"]["job_id"],
        "--base-model",
        config["model"]["base_model"],
        "--output-model",
        f"accounts/{config['fireworks']['account_id']}/models/{config['model']['output_model']}",
        "--epochs",
        str(training["epochs"]),
        "--lora-rank",
        str(training["lora_rank"]),
        "--max-context-length",
        str(training["max_context_length"]),
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
        *(
            ["--env-file", env_file]
            if config["evaluator"].get("secret_env") and env_file
            else []
        ),
    ]


def build_evaluation_command(config: dict) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "--ep-success-threshold=-1e9",
        config["evaluator"]["entry"],
        "-vs",
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


def _failure_category(row: dict, score: float, step_count: int) -> str:
    if score == 1.0:
        return "success"
    calls = [
        (call["function"]["name"], call["function"].get("arguments", ""))
        for message in row["messages"]
        for call in message.get("tool_calls") or []
    ]
    if any(calls[index : index + 3] == [calls[index]] * 3 for index in range(len(calls) - 2)):
        return "repeated_action_loop"
    tool_text = " ".join(
        str(message.get("content", "")).lower()
        for message in row["messages"]
        if message.get("role") == "tool"
    )
    if "unknown element_id" in tool_text or "error" in tool_text:
        return "invalid_action"
    terminal = '"terminal": true' in tool_text or '\\"terminal\\": true' in tool_text
    oracle = row["input_metadata"].get("session_data", {}).get("oracle_steps", 0)
    if terminal and step_count < oracle:
        return "premature_finish"
    if terminal:
        return "verifier_failure"
    return (
        "step_budget_exhaustion"
        if row.get("rollout_status", {}).get("code") == 100
        else "runtime_failure"
    )


def _evaluation_metrics(output: Path) -> tuple[dict, list[list]]:
    rows = [
        json.loads(line)
        for result in (output / "experiment_results").glob("*.jsonl")
        for line in result.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("held-out evaluation produced no result rows")
    scores = [float(row["evals"]["score"]) for row in rows]
    durations = [row["execution_metadata"]["rollout_duration_seconds"] for row in rows]
    steps = [
        sum(len(message.get("tool_calls") or []) for message in row["messages"])
        for row in rows
    ]
    usages = [row["execution_metadata"].get("usage", {}) for row in rows]
    failures = [
        _failure_category(row, score, step_count)
        for row, score, step_count in zip(rows, scores, steps)
    ]
    costs = [
        row["execution_metadata"].get("cost_metrics", {}).get("total_cost_dollar") or 0
        for row in rows
    ]
    metrics = {
        "eval/success_rate": sum(scores) / len(scores),
        "eval/successes": sum(scores),
        "eval/rollouts": len(scores),
        "eval/mean_steps": sum(steps) / len(steps),
        "eval/mean_rollout_seconds": sum(durations) / len(durations),
        "eval/mean_prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages)
        / len(usages),
        "eval/mean_completion_tokens": sum(
            u.get("completion_tokens", 0) for u in usages
        )
        / len(usages),
        "eval/invalid_action_rate": failures.count("invalid_action") / len(failures),
        "eval/inference_cost_dollars": sum(costs),
    }
    table = []
    for row, score, step_count, duration, failure, cost in zip(
        rows, scores, steps, durations, failures, costs
    ):
        usage = row["execution_metadata"].get("usage", {})
        session = row["input_metadata"].get("session_data", {})
        table.append(
            [
                row["input_metadata"]["row_id"],
                session.get("family_id", row["input_metadata"]["row_id"]),
                session.get("oracle_steps"),
                score,
                step_count,
                failure,
                duration,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                cost,
            ]
        )
    for failure in set(failures):
        metrics[f"eval/failure/{failure}"] = failures.count(failure)
    for task_id in {row[0] for row in table}:
        task_scores = [row[3] for row in table if row[0] == task_id]
        metrics[f"eval/task/{task_id}/success_rate"] = sum(task_scores) / len(
            task_scores
        )
    return metrics, table


def compare_evaluations(base: Path, tuned: Path) -> dict:
    """Apply the pre-registered task-clustered held-out win rule."""
    base_metrics, base_rows = _evaluation_metrics(base)
    tuned_metrics, tuned_rows = _evaluation_metrics(tuned)

    def task_scores(rows: list[list]) -> dict[str, float]:
        task_ids = {row[0] for row in rows}
        return {
            task_id: sum(row[3] for row in rows if row[0] == task_id)
            / sum(row[0] == task_id for row in rows)
            for task_id in task_ids
        }

    base_scores, tuned_scores = task_scores(base_rows), task_scores(tuned_rows)
    if base_scores.keys() != tuned_scores.keys():
        raise ValueError("base and tuned evaluations must contain identical task IDs")
    tasks = sorted(base_scores)
    deltas = [tuned_scores[task] - base_scores[task] for task in tasks]
    rng = random.Random(0)
    bootstrap = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(10_000)
    )
    delta = sum(deltas) / len(deltas)
    ci = [bootstrap[249], bootstrap[9749]]
    families = {row[1] for row in base_rows}
    family_deltas = {
        family: sum(
            tuned_scores[row[0]] - base_scores[row[0]]
            for row in base_rows
            if row[1] == family
        )
        / sum(row[1] == family for row in base_rows)
        for family in families
    }
    result = {
        "base_success_rate": base_metrics["eval/success_rate"],
        "tuned_success_rate": tuned_metrics["eval/success_rate"],
        "success_delta": delta,
        "task_clustered_95_ci": ci,
        "family_deltas": family_deltas,
        "base_invalid_action_rate": base_metrics["eval/invalid_action_rate"],
        "tuned_invalid_action_rate": tuned_metrics["eval/invalid_action_rate"],
    }
    result["win"] = (
        delta >= 0.10
        and ci[0] > 0
        and min(family_deltas.values()) >= -0.10
        and result["tuned_invalid_action_rate"]
        <= result["base_invalid_action_rate"]
    )
    return result


def _log_wandb_evaluation(
    config: dict, *, model: str, model_path: str, deployment: str, output: Path
) -> dict:
    try:
        import wandb
    except ImportError as error:
        raise ValueError("wandb is required to log held-out evaluation") from error
    metrics, rows = _evaluation_metrics(output)
    evaluation = config["evaluation"]
    expected_rollouts = evaluation["rows"] * evaluation["num_runs"]
    if metrics["eval/rollouts"] != expected_rollouts:
        raise ValueError(
            f"expected {expected_rollouts} held-out rollouts, "
            f"found {metrics['eval/rollouts']}"
        )
    run = wandb.init(
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"],
        group=evaluation["id"],
        job_type="heldout-evaluation",
        id=f"{evaluation['id']}-{model}-{output.name}",
        name=f"{evaluation['id']}-{model}",
        resume="allow",
        config={
            "model_variant": model,
            "model": model_path,
            "deployment": deployment,
            "dataset": evaluation["path"],
            "rows": evaluation["rows"],
            "num_runs": evaluation["num_runs"],
            "max_concurrent_rollouts": evaluation["max_concurrent_rollouts"],
            "temperature": config["rollout"]["temperature"],
            "max_output_tokens": config["rollout"]["max_output_tokens"],
            "steps": config["rollout"]["steps"],
        },
        tags=["browser-use", "heldout", model],
    )
    run.log(
        {
            **metrics,
            "eval/results": wandb.Table(
                columns=[
                    "task_id",
                    "family_id",
                    "oracle_steps",
                    "score",
                    "steps",
                    "failure_category",
                    "rollout_seconds",
                    "prompt_tokens",
                    "completion_tokens",
                    "inference_cost_dollars",
                ],
                data=rows,
            ),
        }
    )
    run.summary.update(metrics)
    run.finish()
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def evaluate_rft(
    path: Path, *, model: str, deployment: str | None, confirm: bool
) -> None:
    """Plan or run the same held-out verifier against base or tuned policy."""
    config = load_rft_config(path)
    load_dotenv(PROJECT_ROOT / ".env")
    model_path = (
        config["model"]["base_model"]
        if model == "base"
        else f"accounts/{config['fireworks']['account_id']}/models/{config['model']['output_model']}"
    )
    evaluation = config["evaluation"]
    if deployment and not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", deployment):
        raise ValueError("--deployment must be a lowercase Fireworks ID")
    deployment_path = (
        f"accounts/{config['fireworks']['account_id']}/deployments/{deployment}"
        if deployment
        else "serverless"
    )
    served_model = f"{model_path}#{deployment_path}" if deployment else model_path
    command = build_evaluation_command(config)
    print(
        f"model={model_path} deployment={deployment_path} "
        f"heldout={evaluation['path']} rows={evaluation['rows']} "
        f"runs={evaluation['num_runs']} metric=mean_binary_reward"
    )
    print(shlex.join(command))
    if not confirm:
        print("plan only; pass --confirm to run paid held-out model evaluation")
        return
    required = tuple(
        name
        for name in ("FIREWORKS_API_KEY", config["evaluator"].get("secret_env"))
        if name
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise ValueError(f"missing required secrets: {', '.join(missing)}")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = (
        PROJECT_ROOT / "fireworks-training-runs" / evaluation["id"] / model / timestamp
    )
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if config["evaluator"]["entry"].startswith("evaluator/miniwob/"):
        evaluator_environment = {
            "MINIWOB_EVAL_DATASET": config["evaluation"]["path"],
            "MINIWOB_TASKS_PATH": config["evaluation"]["path"],
            "MINIWOB_MAX_STEPS": str(config["rollout"]["steps"]),
            "MINIWOB_EVAL_MODEL": f"fireworks_ai/{served_model}",
        }
    else:
        evaluator_environment = {
            "BROWSERUSE_EVAL_DATASET": config["evaluation"]["path"],
            "BROWSERUSE_TASKS_PATH": config["evaluation"]["path"],
            "BROWSERUSE_MAX_STEPS": str(config["rollout"]["steps"]),
            "BROWSERUSE_EVAL_MODEL": f"fireworks_ai/{served_model}",
        }
    environment.update(
        {
            **evaluator_environment,
            "EP_DISABLE_AUTO_BROWSER": "1",
            "EP_MAX_CONCURRENT_EVALUATIONS": str(evaluation["max_concurrent_rollouts"]),
            "EP_MAX_CONCURRENT_ROLLOUTS": str(evaluation["max_concurrent_rollouts"]),
            "EP_NO_UPLOAD": "1",
            "EP_NUM_RUNS": str(evaluation["num_runs"]),
            "EP_OUTPUT_DIR": str(output),
            "EP_PRINT_SUMMARY": "1",
            "EP_SUMMARY_JSON": str(output / "summary.json"),
        }
    )
    if config["evaluator"]["entry"].startswith("evaluator/miniwob/"):
        environment["MINIWOB_PYTHON"] = str(
            PROJECT_ROOT / ".venv-browsergym" / "bin" / "python"
        )
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
    if result.returncode:
        raise ValueError(
            f"held-out evaluation failed with exit code {result.returncode}"
        )
    metrics = _log_wandb_evaluation(
        config,
        model=model,
        model_path=model_path,
        deployment=deployment_path,
        output=output,
    )
    print(json.dumps(metrics, indent=2))


def run_rft(path: Path, *, confirm: bool, watch: bool = True) -> None:
    """Print a redacted plan or execute it after explicit confirmation."""
    config = load_rft_config(path)
    load_dotenv(PROJECT_ROOT / ".env")
    rollout_count = (
        config["dataset"]["rows"] * config["rollout"]["response_candidates_count"]
    )
    secret_names = tuple(
        name
        for name in (
            "FIREWORKS_API_KEY",
            config["evaluator"].get("secret_env"),
            config["wandb"]["api_key_env"],
        )
        if name
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
    evaluator_secret = config["evaluator"].get("secret_env")
    wandb_key = os.environ[config["wandb"]["api_key_env"]]
    launch_environment = os.environ.copy()
    for name in (
        "BROWSERUSE_EVAL_DATASET",
        "BROWSERUSE_EVAL_MODEL",
        "BROWSERUSE_TASKS_PATH",
        "MINIWOB_EVAL_DATASET",
        "MINIWOB_EVAL_MODEL",
        "MINIWOB_TASKS_PATH",
    ):
        launch_environment.pop(name, None)
    if config["evaluator"]["entry"].startswith("evaluator/miniwob/"):
        evaluator_environment = {
            "MINIWOB_EVAL_DATASET": str(PROJECT_ROOT / config["dataset"]["path"]),
            "MINIWOB_MAX_STEPS": str(config["rollout"]["steps"]),
            "MINIWOB_TASKS_PATH": str(PROJECT_ROOT / config["dataset"]["path"]),
            "MINIWOB_PYTHON": str(
                PROJECT_ROOT / ".venv-browsergym" / "bin" / "python"
            ),
        }
    else:
        evaluator_environment = {
            "BROWSERUSE_EVAL_DATASET": config["dataset"]["path"],
            "BROWSERUSE_MAX_STEPS": str(config["rollout"]["steps"]),
            "BROWSERUSE_TASKS_PATH": config["dataset"]["path"],
        }
    launch_environment.update(evaluator_environment)
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="browser-rft-", suffix=".env"
    ) as env_file:
        if evaluator_secret:
            env_file.write(f"{evaluator_secret}={os.environ[evaluator_secret]}\n")
        env_file.flush()
        result = subprocess.run(
            build_rft_command(
                config, env_file.name if evaluator_secret else None, wandb_key
            ),
            cwd=(
                PROJECT_ROOT / "evaluator" / "miniwob"
                if config["evaluator"]["entry"].startswith("evaluator/miniwob/")
                else PROJECT_ROOT
            ),
            env=launch_environment,
            check=False,
        )
    if result.returncode:
        raise ValueError(
            f"Eval Protocol create rft failed with exit code {result.returncode}"
        )
    if watch:
        watch_rft(path)
