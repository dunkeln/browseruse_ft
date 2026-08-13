"""Constrained BrowserGym tools for executable MiniWoB tasks."""

from __future__ import annotations

import json
import os
from asyncio import get_running_loop
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/ms-playwright")

import browsergym.miniwob  # noqa: F401
import gymnasium as gym
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str
from mcp.server.fastmcp import FastMCP

TASKS_PATH = Path(
    os.environ.get(
        "MINIWOB_TASKS_PATH", Path(__file__).with_name("train_tasks.jsonl")
    )
)
TASKS = {
    row["input_metadata"]["row_id"]: row["input_metadata"]["session_data"]
    for line in TASKS_PATH.read_text().splitlines()
    if (row := json.loads(line))
}
MINIWOB_URL = f"file://{Path(__file__).with_name('html') / 'miniwob'}/"
ACTION_SET = HighLevelActionSet(subsets=["bid"], multiaction=False, strict=True)

_env = None
_task_id: str | None = None
_observation: dict | None = None
_reward = 0.0
_terminated = False
_last_result: dict | None = None
_browser_thread = ThreadPoolExecutor(max_workers=1)


def _close() -> None:
    global _env, _observation, _task_id
    if _env is not None:
        _env.close()
    _env = _observation = _task_id = None


def _observe() -> dict:
    if _observation is None:
        raise RuntimeError("call open_task before using browser tools")
    return {
        "task_id": _task_id,
        "goal": _observation["goal"],
        "axtree": flatten_axtree_to_str(
            _observation["axtree_object"],
            _observation["extra_element_properties"],
        )[:8000],
        "last_action_error": _observation["last_action_error"],
        "done": _terminated,
        "reward": _reward,
    }


def _open_task(task_id: str) -> str:
    """Open a seeded MiniWoB task and return its goal and accessibility tree."""
    global _env, _last_result, _observation, _reward, _task_id, _terminated
    task = TASKS.get(task_id)
    if task is None:
        raise ValueError(f"unknown task_id: {task_id}")
    _close()
    _last_result = None
    _reward = 0.0
    _terminated = False
    _task_id = task_id
    _env = gym.make(
        f"browsergym/miniwob.{task['task_name']}",
        headless=True,
        action_mapping=ACTION_SET.to_python_code,
        task_kwargs={"base_url": MINIWOB_URL},
    )
    _observation, _ = _env.reset(seed=task["seed"])
    return json.dumps(_observe())


def _act(action: str) -> str:
    """Execute one BrowserGym bid action, such as click('12') or fill('7', 'text')."""
    global _observation, _reward, _terminated
    if _env is None or _terminated:
        raise RuntimeError("open a nonterminal task before acting")
    _observation, _reward, _terminated, truncated, _ = _env.step(action)
    _terminated = _terminated or truncated
    return json.dumps(_observe())


def _finish() -> str:
    """Return native terminal reward and close the browser."""
    global _last_result
    if _last_result is not None:
        return json.dumps(_last_result)
    if _env is None:
        raise RuntimeError("call open_task before finish")
    _last_result = {
        "terminal": True,
        "done": True,
        "task_id": _task_id,
        "reward": float(_terminated and _reward == 1.0),
    }
    _close()
    return json.dumps(_last_result)


async def _in_browser_thread(function, *args) -> str:
    return await get_running_loop().run_in_executor(
        _browser_thread, partial(function, *args)
    )


async def open_task(task_id: str) -> str:
    """Open a seeded MiniWoB task and return its goal and accessibility tree."""
    return await _in_browser_thread(_open_task, task_id)


async def act(action: str) -> str:
    """Execute one BrowserGym bid action, such as click('12') or fill('7', 'text')."""
    return await _in_browser_thread(_act, action)


async def finish() -> str:
    """Return native terminal reward and close the browser."""
    return await _in_browser_thread(_finish)


mcp = FastMCP("miniwob")
for tool in (open_task, act, finish):
    mcp.tool()(tool)


if __name__ == "__main__":
    mcp.run()
