"""Terminal entry points for the browser fine-tuning flywheel."""

from pathlib import Path

import click

from browseruse_ft.data import (
    DATASET_CARDS,
    TASK_DATASETS,
    load_hub_dataset,
    load_task_split,
    write_jsonl,
)
from browseruse_ft.preprocess.main import preprocess_sft
from browseruse_ft.rft import evaluate_rft, run_rft, watch_rft


@click.command()
@click.argument("source", type=click.Choice((*DATASET_CARDS, *TASK_DATASETS)))
@click.option("--split", type=click.Choice(("train", "test")), default="train")
@click.option("--rows", type=click.IntRange(min=1))
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
    "--output", type=click.Path(path_type=Path, dir_okay=False), required=True
)
def prepare_sft(split: str, fold: int | None, rows: int | None, output: Path) -> None:
    """Render WebArena-Lite examples as reviewable SFT JSONL."""
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
        f"input={report['input_rows']:,} output={report['output_rows']:,} "
        f"duplicates={report['exact_duplicates_removed']:,} "
        f"quarantined={report['conflicting_rows_quarantined']:,} "
        f"artifact={destination} "
        f"quarantine={quarantine_path if quarantine.num_rows else 'none'}"
    )


@click.command()
@click.argument("config", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option(
    "--confirm",
    is_flag=True,
    help="Upload evaluator/dataset and start the configured RFT job.",
)
@click.option(
    "--watch/--no-watch",
    default=True,
    help="Show live Fireworks progress after a confirmed launch.",
)
def train_rft(config: Path, confirm: bool, watch: bool) -> None:
    """Validate and run one declarative hosted-evaluator RFT config."""
    try:
        run_rft(config.resolve(), confirm=confirm, watch=watch)
    except (OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error


@click.command()
@click.argument("config", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--once", is_flag=True, help="Read one authoritative status snapshot.")
def watch_rft_job(config: Path, once: bool) -> None:
    """Resume rich progress monitoring for a configured RFT job."""
    try:
        watch_rft(config.resolve(), once=once)
    except (OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error


@click.command()
@click.argument("config", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--model", type=click.Choice(("base", "tuned")), required=True)
@click.option("--confirm", is_flag=True, help="Run paid held-out model evaluation.")
def evaluate_rft_model(config: Path, model: str, confirm: bool) -> None:
    """Evaluate base or tuned policy on the disjoint held-out browser tasks."""
    try:
        evaluate_rft(config.resolve(), model=model, confirm=confirm)
    except (OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
