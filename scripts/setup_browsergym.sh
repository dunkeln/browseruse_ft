#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runtime="$root/.venv-browsergym"
miniwob="$root/evaluator/miniwob/html/miniwob"

uv venv "$runtime" --python 3.12 --allow-existing --no-project
uv pip install --python "$runtime/bin/python" browsergym-miniwob==0.14.3 mcp==1.29.0
"$runtime/bin/playwright" install chromium

MINIWOB_URL="file://$miniwob/" \
  "$runtime/bin/python" - <<'PY'
import gymnasium as gym
import browsergym.miniwob

env = gym.make("browsergym/miniwob.click-button-sequence", headless=True)
try:
    _, _ = env.reset(seed=1)
    _, _, _, _, _ = env.step("click('12')")
    _, reward, terminated, _, _ = env.step("click('13')")
    assert reward == 1 and terminated
finally:
    env.close()
print("MiniWoB verifier smoke passed")
PY
