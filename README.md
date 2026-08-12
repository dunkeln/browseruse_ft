# Browser-use RFT

One hosted evaluator runs the whole browser RLVR loop:

`Fireworks model -> Eval Protocol agent -> browser tools -> Browser Use Cloud -> 0/1 reward`

There is no remote rollout service, VM, Vercel deployment, Magento instance,
or website profile. A reviewed TOML is the executable run contract.

## Files

- `evaluator/test_browseruse.py` — hosted evaluator and terminal reward.
- `evaluator/browser.py` — constrained MCP tools backed by Browser Use Cloud.
- `evaluator/tasks.jsonl` — prompts, isolated task pages, and verifier contracts.
- `evaluator/mcp.json` — launches the bundled browser tool server.
- `src/browseruse_ft/` — reusable dataset loading, splits, admission, and rendering.
- `notebooks/data_explore.py` — Marimo inspection surface; it does not orchestrate runs.

## Setup

```bash
uv sync
```

The ELT CLI remains available independently of browser execution:

```bash
uv run browseruse-data webarena_lite_sft --split train --rows 10
uv run browseruse-sft --split train --fold 0 --rows 10 \
  --output data/processed/sft-smoke.jsonl
uv run marimo edit notebooks/data_explore.py
```

The hosted RFT run is declarative again. Without `--confirm`, it only validates
the TOML and prints the exact command:

```bash
uv run browseruse-rft configs/rft/001-browseruse-cloud-smoke.toml
uv run browseruse-rft configs/rft/001-browseruse-cloud-smoke.toml --confirm
uv run browseruse-rft-watch configs/rft/001-browseruse-cloud-smoke.toml
uv run browseruse-rft-eval configs/rft/001-browseruse-cloud-smoke.toml --model base
uv run browseruse-rft-eval configs/rft/001-browseruse-cloud-smoke.toml --model tuned
```

The confirmed path remains interactive when selecting evaluator secrets. Select
only the Browser Use key; do not upload the Fireworks API key as an evaluator
secret.

Set either `BROWSERUSE_API_KEY` or `BROWSER_USE_API_KEY` in `.env`. The former
is kept for compatibility with this workspace; the latter is Browser Use's
canonical name. Also set `FIREWORKS_API_KEY` and `WANDB_API_KEY`; the TOML pins
the W&B entity/project without storing any secret.

The smoke is bounded to 8 browser tasks × 2 candidates = 16 trajectories,
chunked in groups of 4 with concurrency 1. The exact previous browser RFT model,
`accounts/fireworks/models/qwen3-vl-8b-instruct`, is pinned in both the TOML and
the evaluator. W&B receives trainer loss and reward metrics after rollouts reach
the optimizer; because reward is binary terminal task success, mean reward is
the training success rate—not a separate classification accuracy metric. The
evaluation CLI uses two disjoint held-out browser tasks and is also plan-only
until `--confirm` is supplied.

## Prove the browser layer

This creates one Browser Use Cloud session, executes the known-good actions,
checks reward `1.0`, and stops the session:

```bash
uv run python evaluator/browser.py --smoke
```

## Prove the complete evaluator locally

This adds model inference to the same loop:

```bash
uv run ep local-test \
  --entry evaluator/test_browseruse.py::test_browseruse \
  --yes
```

## Upload the hosted evaluator

The upload command is intentionally interactive so only the browser key is
selected from `.env` and existing secrets are not overwritten accidentally:

```bash
uv run ep upload \
  --entry evaluator/test_browseruse.py::test_browseruse \
  --id browseruse-cloud \
  --display-name "Browser Use Cloud" \
  --env-file .env
```

Add tasks by appending rows to `evaluator/tasks.jsonl`. Every row owns its page
and verifier; browser execution remains unchanged.
