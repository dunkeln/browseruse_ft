# Browser-use RFT

One hosted evaluator runs the whole browser RLVR loop:

`Fireworks model -> Eval Protocol agent -> browser tools -> Browser Use Cloud -> 0/1 reward`

There is no remote rollout service, VM, Vercel deployment, Magento instance,
website profile, or executable run config.

## Files

- `evaluator/test_browseruse.py` — hosted evaluator and terminal reward.
- `evaluator/browser.py` — constrained MCP tools backed by Browser Use Cloud.
- `evaluator/tasks.jsonl` — prompts, isolated task pages, and verifier contracts.
- `evaluator/mcp.json` — launches the bundled browser tool server.

## Setup

```bash
uv sync
```

Set either `BROWSERUSE_API_KEY` or `BROWSER_USE_API_KEY` in `.env`. The former
is kept for compatibility with this workspace; the latter is Browser Use's
canonical name.

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
