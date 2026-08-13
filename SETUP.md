# Setup and execution

## Install

```bash
uv sync
```

Create `.env` with the credentials used by the selected TOML:

```dotenv
FIREWORKS_API_KEY=...
BROWSERUSE_API_KEY=...
WANDB_API_KEY=...
```

`BROWSER_USE_API_KEY` is also accepted as the canonical Browser Use variable.
When uploading the hosted evaluator, select only the Browser Use key from `.env`.

MiniWoB uses an isolated BrowserGym environment because its Playwright version
conflicts with Browser Use's runtime:

```bash
./scripts/setup_browsergym.sh
```

## Inspect and preprocess data

```bash
uv run browseruse-data webarena_lite_sft --split train --rows 10
uv run browseruse-sft --split train --fold 0 --rows 10 \
  --output data/processed/sft-smoke.jsonl
uv run marimo edit notebooks/data_explore.py
```

Materialize provider rollouts as execution-backed trajectories:

```bash
uv run browseruse-trajectories <experiment-results-directory> \
  --heldout evaluator/miniwob/heldout_tasks.jsonl \
  --max-context-chars 65536 \
  --output data/processed/verified-trajectories.jsonl
```

## Train

Without `--confirm`, the CLI validates the TOML and prints the provider command
without launching a paid job.

Browser Use Cloud:

```bash
uv run browseruse-rft configs/rft/001-browseruse-cloud-rft.toml
uv run browseruse-rft configs/rft/001-browseruse-cloud-rft.toml --confirm
uv run browseruse-rft-watch configs/rft/001-browseruse-cloud-rft.toml
```

MiniWoB:

```bash
uv run python scripts/build_miniwob_tasks.py
uv run browseruse-rft configs/rft/004-miniwob-rft.toml
uv run browseruse-rft configs/rft/004-miniwob-rft.toml --confirm
uv run browseruse-rft-watch configs/rft/004-miniwob-rft.toml --once
```

Long-horizon tasks:

```bash
uv run python scripts/build_long_horizon_tasks.py
uv run browseruse-rft configs/rft/002-browseruse-long-horizon-rft.toml
```

## Evaluate

Base models use serverless inference by default. Pass an explicit ready
deployment for models that are not serverless.

```bash
uv run browseruse-rft-eval configs/rft/004-miniwob-rft.toml \
  --model base --deployment <base-deployment> --confirm
uv run browseruse-rft-eval configs/rft/004-miniwob-rft.toml \
  --model tuned --deployment <tuned-deployment> --confirm
uv run browseruse-rft-compare <base-artifact-dir> <tuned-artifact-dir>
```

## Prove the evaluator locally

Exercise the browser layer without model inference:

```bash
uv run python evaluator/browser.py --smoke
```

Exercise the complete evaluator with model inference:

```bash
uv run ep local-test \
  --entry evaluator/test_browseruse.py::test_browseruse \
  --ignore-docker \
  --yes
```

Upload the hosted evaluator:

```bash
uv run ep upload \
  --entry evaluator/test_browseruse.py::test_browseruse \
  --id browseruse-cloud \
  --display-name "Browser Use Cloud" \
  --env-file .env
```

Add Browser Use tasks by appending rows to `evaluator/tasks.jsonl`; browser
execution does not change.
