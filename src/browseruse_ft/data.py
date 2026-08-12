"""Load, inspect, and materialize browser-training datasets."""

from hashlib import sha256
from math import ceil
from pathlib import Path

import polars as pl
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

DATASET_CARDS = {
    "webarena_lite_sft": {
        "repo_id": "weizhepei/WebArena-Lite-SFT",
        "split": "train",
        "data_file": "train.json",
        "role": "Primary HTML behavior-cloning data used by WebAgent-R1.",
        "modality": "HTML/text",
        "size": "~96 MB",
        "revision": "b560a725989c99e99232ab4f33ea1565af250612",
    },
}

TASK_DATASETS = {
    "webarena_lite_tasks": "WebArena tasks with paper and project holdouts.",
}

SFT_TEST_FRACTION = 0.2
SFT_FOLDS = 3
EXTRA_TEST_TASKS = 130

WEBARENA_TASKS_URL = (
    "https://raw.githubusercontent.com/web-arena-x/webarena/"
    "dce04686a56253aefba7b18a4fa0937cf1dc987b/config_files/test.raw.json"
)
WEBARENA_LITE_EVAL_URL = (
    "https://raw.githubusercontent.com/THUDM/VisualAgentBench/"
    "9055fc299c366ef34700d1710215fb60a0d8c35e/"
    "VAB-WebArena-Lite/new/test_webarena_lite.raw.json"
)


def _held_out_values(values: list, count: int) -> set:
    unique = set(values)
    return set(sorted(unique, key=lambda value: sha256(str(value).encode()).digest())[:count])


def load_hub_dataset(
    card: str,
    split: str = "train",
    rows: int | None = None,
    fold: int | None = None,
) -> Dataset:
    """Load a deterministic, problem-disjoint SFT split or training fold."""
    if card not in DATASET_CARDS:
        raise ValueError(f"Unknown dataset card: {card}")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if rows is not None and rows < 1:
        raise ValueError("rows must be positive")
    if split == "validation" and fold is None:
        raise ValueError("validation requires a fold")
    if fold is not None and (split == "test" or fold not in range(SFT_FOLDS)):
        raise ValueError(f"fold must be 0-{SFT_FOLDS - 1} for train/validation")

    config = DATASET_CARDS[card]
    source = load_dataset(
        config["repo_id"],
        data_files={config["split"]: config["data_file"]}
        if config.get("data_file")
        else None,
        split=config["split"],
        revision=config.get("revision"),
    )
    test_problems = _held_out_values(
        source["problem"], ceil(len(set(source["problem"])) * SFT_TEST_FRACTION)
    )
    training_pool = source.select(
        [index for index, problem in enumerate(source["problem"]) if problem not in test_problems]
    )
    if split == "test":
        dataset = source.select(
            [index for index, problem in enumerate(source["problem"]) if problem in test_problems]
        )
    elif fold is None:
        dataset = training_pool
    else:
        problems = sorted(
            set(training_pool["problem"]),
            key=lambda problem: sha256(problem.encode()).digest(),
        )
        validation_problems = set(problems[fold::SFT_FOLDS])
        dataset = training_pool.select(
            [
                index
                for index, problem in enumerate(training_pool["problem"])
                if (problem in validation_problems) == (split == "validation")
            ]
        )
    return dataset if rows is None else dataset.select(range(min(rows, dataset.num_rows)))


def load_task_split(split: str = "train", rows: int | None = None) -> Dataset:
    """Load disjoint RFT train or expanded project test tasks and verifiers."""
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if rows is not None and rows < 1:
        raise ValueError("rows must be positive")

    evaluation = load_dataset("json", data_files=WEBARENA_LITE_EVAL_URL, split="train")
    held_out_ids = set(evaluation["old_task_id"])
    if len(held_out_ids) != 165:
        raise RuntimeError("Expected 165 unique held-out WebArena-Lite task IDs")

    all_tasks = load_dataset("json", data_files=WEBARENA_TASKS_URL, split="train")
    rft_tasks = all_tasks.select(
        [index for index, task_id in enumerate(all_tasks["task_id"]) if task_id not in held_out_ids]
    )
    extra_test_ids = _held_out_values(rft_tasks["task_id"], EXTRA_TEST_TASKS)

    if split == "train":
        dataset = rft_tasks.select(
            [index for index, task_id in enumerate(rft_tasks["task_id"]) if task_id not in extra_test_ids]
        )
        dataset = dataset.add_column("source_task_id", dataset["task_id"])
        dataset = dataset.add_column("project_subset", ["rft_train"] * len(dataset))
    else:
        extra_test = rft_tasks.select(
            [index for index, task_id in enumerate(rft_tasks["task_id"]) if task_id in extra_test_ids]
        )
        evaluation = evaluation.add_column("source_task_id", evaluation["old_task_id"])
        evaluation = evaluation.add_column("project_subset", ["official_eval"] * len(evaluation))
        extra_test = extra_test.add_column("source_task_id", extra_test["task_id"])
        extra_test = extra_test.add_column("project_subset", ["extra_holdout"] * len(extra_test))
        dataset = concatenate_datasets([evaluation, extra_test])

    expected_rows = 517 if split == "train" else 295
    if dataset.num_rows != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} WebArena-Lite {split} tasks")
    return dataset if rows is None else dataset.select(range(min(rows, dataset.num_rows)))


def load_jsonl(path: str | Path) -> Dataset:
    """Load one immutable JSONL source file as a Hugging Face Dataset."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return load_dataset("json", data_files=str(source), split="train")


def audit(dataset: Dataset) -> pl.DataFrame:
    """Return a compact, notebook-friendly column audit."""
    frame = dataset.to_polars()
    return pl.DataFrame(
        {
            "column": frame.columns,
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "null_count": [frame[column].null_count() for column in frame.columns],
        }
    )


def write_jsonl(dataset: Dataset, path: str | Path) -> Path:
    """Materialize a prepared Dataset as JSONL for review or upload."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(destination, orient="records", lines=True, force_ascii=False)
    return destination
