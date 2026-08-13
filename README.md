# Browser policy RFT

An end-to-end flywheel for training small vision-language models as browser
automation policies:

`Fireworks model -> Eval Protocol agent -> constrained browser tools -> task environment -> terminal reward`

Browser Use Cloud and MiniWoB++ are interchangeable environment backends behind
the same policy and evaluator boundary. A reviewed TOML defines each dataset,
model, rollout budget, training run, and evaluation.

## What it does

- Runs managed RLVR for Qwen3-VL-8B with executable browser actions.
- Preserves observations, actions, tool results, usage, and final outcomes.
- Rejects unfinished, malformed, leaked, and oversized trajectories.
- Freezes train, development, and held-out task families before training.
- Compares base and tuned policies under identical serving and rollout budgets.
- Tears down temporary deployments after evaluation.

## Training improvements

Two managed RFT runs produced positive training-task reward movement:

- Browser Use: `84.38% -> 100%` (`+15.63` percentage points).
- MiniWoB: `53.13% -> 57.81%` (`+4.69` percentage points).

![Training-task terminal reward improved across both managed RFT runs](docs/training-reward-improvement.svg)

The MiniWoB run completed 128 rollouts across 2 epochs. Its preprocessing path
materialized 111 complete execution-backed trajectories while quarantining 17
incomplete trajectories, with serialized contexts remaining below the configured
65,536-character admission ceiling.

The resulting pipeline is resume-safe and reproducible: managed training,
execution-backed trajectory admission, native task evaluation, family-level
splits, matched base-versus-tuned evaluation, and deployment cleanup all share
one declarative CLI surface.

## Project map

- `configs/rft/` — executable run contracts.
- `evaluator/` — hosted evaluator, browser tools, tasks, and reward contracts.
- `src/browseruse_ft/` — dataset, preprocessing, training, and evaluation CLI.
- `scripts/` — reproducible task generation and environment setup.
- `notebooks/data_explore.py` — Marimo data inspection surface.

See [SETUP.md](SETUP.md) for installation, configuration, training, evaluation,
and local proof commands.
