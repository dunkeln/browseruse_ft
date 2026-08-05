"""Terminal entry points for the data flywheel."""

import click

from browseruse_ft.data import (
    DATASET_CARDS,
    TASK_DATASETS,
    load_hub_dataset,
    load_task_split,
)


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
