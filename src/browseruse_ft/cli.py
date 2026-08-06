"""Terminal entry points for the data flywheel."""

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
@click.option("--split", type=click.Choice(("train", "test")), default="train")
@click.option("--rows", type=click.IntRange(min=1))
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
)
def prepare_sft(split: str, rows: int | None, output: Path) -> None:
    """Render WebArena-Lite examples as Fireworks-compatible JSONL."""
    dataset = load_hub_dataset("webarena_lite_sft", split=split, rows=rows)
    prepared, quarantine, report = preprocess_sft(dataset)
    destination = write_jsonl(prepared, output)
    quarantine_path = output.with_name(f"{output.stem}.quarantine.jsonl")
    if quarantine.num_rows:
        write_jsonl(quarantine, quarantine_path)
    click.echo(
        f"split={split} input={report['input_rows']:,} "
        f"output={report['output_rows']:,} "
        f"duplicates={report['exact_duplicates_removed']:,} "
        f"quarantined={report['conflicting_rows_quarantined']:,} "
        f"artifact={destination} "
        f"quarantine={quarantine_path if quarantine.num_rows else 'none'}"
    )
