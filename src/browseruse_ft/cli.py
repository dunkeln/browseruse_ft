"""Terminal entry points for the fine-tuning flywheel."""

import asyncio
from pathlib import Path

import click

from browseruse_ft.data import (
    DATASET_CARDS,
    TASK_DATASETS,
    load_hub_dataset,
    load_task_split,
    write_jsonl,
)
from browseruse_ft.managed_sft import run_sft
from browseruse_ft.model_rollout import run_model_rollouts
from browseruse_ft.preprocess.main import preprocess_sft
from browseruse_ft.world import capture_storage_state, observe_url
from browseruse_ft.world.smoke import run_calibration


@click.command()
@click.argument("source", type=click.Choice((*DATASET_CARDS, *TASK_DATASETS)))
@click.option("--split", type=click.Choice(("train", "test")), default="train")
@click.option(
    "--rows",
    type=click.IntRange(min=1),
    help="Select this many rows after the complete split is cached.",
)
def download_data(source: str, split: str, rows: int | None) -> None:
    """Download/cache a reviewed dataset and report what was loaded."""
    try:
        dataset = (
            load_hub_dataset(source, split=split, rows=rows)
            if source in DATASET_CARDS
            else load_task_split(split=split, rows=rows)
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        f"source={source} split={split} rows={dataset.num_rows:,} "
        f"columns={len(dataset.column_names)}"
    )


@click.command()
@click.option(
    "--split",
    type=click.Choice(("train", "validation", "test")),
    default="train",
)
@click.option("--fold", type=click.IntRange(min=0, max=2))
@click.option("--rows", type=click.IntRange(min=1))
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
)
def prepare_sft(split: str, fold: int | None, rows: int | None, output: Path) -> None:
    """Render WebArena-Lite examples as Fireworks-compatible JSONL."""
    try:
        dataset = load_hub_dataset(
            "webarena_lite_sft", split=split, rows=rows, fold=fold
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    prepared, quarantine, report = preprocess_sft(dataset)
    destination = write_jsonl(prepared, output)
    quarantine_path = output.with_name(f"{output.stem}.quarantine.jsonl")
    if quarantine.num_rows:
        write_jsonl(quarantine, quarantine_path)
    click.echo(
        f"split={split} fold={fold if fold is not None else 'none'} "
        f"input={report['input_rows']:,} "
        f"output={report['output_rows']:,} "
        f"duplicates={report['exact_duplicates_removed']:,} "
        f"quarantined={report['conflicting_rows_quarantined']:,} "
        f"artifact={destination} "
        f"quarantine={quarantine_path if quarantine.num_rows else 'none'}"
    )


@click.command()
@click.argument(
    "config",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--confirm",
    is_flag=True,
    help="Confirm dataset uploads and paid Fireworks training.",
)
def train_sft(config: Path, confirm: bool) -> None:
    """Run one managed-SFT experiment defined by CONFIG."""
    try:
        run_sft(config.resolve(), confirm=confirm)
    except ValueError as error:
        raise click.ClickException(str(error)) from error


@click.command()
@click.argument("url")
@click.option("--headed", is_flag=True, help="Show the Chromium window.")
@click.option(
    "--storage-state",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Playwright authentication state JSON.",
)
def world_smoke(url: str, headed: bool, storage_state: Path | None) -> None:
    """Open URL in an isolated Playwright world and print its observation."""
    try:
        click.echo(
            asyncio.run(observe_url(url, headed=headed, storage_state=storage_state))
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error


@click.command()
@click.argument("url")
@click.argument("output", type=click.Path(path_type=Path, dir_okay=False))
def save_auth(url: str, output: Path) -> None:
    """Open a login page and save Playwright authentication state."""
    try:
        click.echo(asyncio.run(capture_storage_state(url, output.resolve())))
    except ValueError as error:
        raise click.ClickException(str(error)) from error


@click.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("data/eval/reward-distribution.jsonl"),
    show_default=True,
)
def rft_smoke(output: Path) -> None:
    """Run local browser rollouts and materialize reward calibration rows."""
    rows = run_calibration(output.resolve())
    click.echo(f"rollouts={len(rows)} artifact={output.resolve()}")


@click.command()
@click.argument(
    "config", type=click.Path(path_type=Path, exists=True, dir_okay=False)
)
@click.option("--confirm", is_flag=True, help="Confirm paid Fireworks inference.")
def model_rollout(config: Path, confirm: bool) -> None:
    """Run model-driven browser rollouts defined by CONFIG."""
    try:
        output = run_model_rollouts(config.resolve(), confirm=confirm)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    if output is not None:
        click.echo(f"artifact={output.resolve()}")
