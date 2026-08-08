"""Execute bounded browser-policy rollouts."""

import ast
from collections.abc import Awaitable, Callable
from pathlib import Path

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page

from .observation import open_page, render_observation
from .reward import REWARD_POLICY, rollout_rewards, verify_task

Policy = Callable[[str, list[dict], str], Awaitable[dict]]


def parse_action(source: str) -> tuple[str, dict[str, str]]:
    """Parse the narrow function-call action language without executing Python."""
    try:
        call = ast.parse(source.strip(), mode="eval").body
    except SyntaxError as error:
        raise ValueError("action must be one function call") from error
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.args
    ):
        raise ValueError("action must be one function call with keyword arguments")
    values = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("expanded action arguments are not allowed")
        value = ast.literal_eval(keyword.value)
        if not isinstance(value, str):
            raise TypeError("action arguments must be strings")
        values[keyword.arg] = value
    return call.func.id, values


async def execute_action(page: Page, source: str) -> dict:
    """Execute one parsed action and return a non-throwing result contract."""
    try:
        function, values = parse_action(source)
        if function in {"exit", "quote"}:
            output = values.get("message", values.get("content"))
            if output is None:
                raise ValueError(f"{function} requires an output")
            return {
                "success": True,
                "terminal": True,
                "output": output,
                "message": function,
            }
        if function == "go_backward":
            await page.go_back(wait_until="domcontentloaded")
            return {
                "success": True,
                "terminal": False,
                "output": None,
                "message": function,
            }
        if function != "do":
            raise ValueError(f"unsupported action function: {function}")

        action = values.get("action")
        element = values.get("element")
        locator = (
            page.locator(f'[data-browseruse-id="{element}"]')
            if element is not None
            else None
        )
        if (
            action in {"Click", "Type", "Search", "Select Dropdown Option", "Hover"}
            and locator is None
        ):
            raise ValueError(f"{action} requires an element")
        if locator is not None and await locator.count() != 1:
            raise ValueError(f"element {element!r} is not available")
        if action == "Click":
            await locator.click()
        elif action == "Type":
            await locator.fill(values.get("argument", ""))
        elif action == "Search":
            await locator.fill(values.get("argument", ""))
            await locator.press("Enter")
        elif action == "Select Dropdown Option":
            await locator.select_option(label=values.get("argument", ""))
        elif action == "Hover":
            await locator.hover()
        elif action == "Scroll Down":
            await page.mouse.wheel(0, 600)
        elif action == "Scroll Up":
            await page.mouse.wheel(0, -600)
        elif action == "Press Enter":
            await (
                locator.press("Enter")
                if locator is not None
                else page.keyboard.press("Enter")
            )
        elif action == "Wait":
            await page.wait_for_timeout(1000)
        else:
            raise ValueError(f"unsupported do action: {action}")
        return {"success": True, "terminal": False, "output": None, "message": action}
    except (PlaywrightError, TypeError, ValueError) as error:
        return {
            "success": False,
            "terminal": False,
            "output": None,
            "message": str(error),
        }


async def run_rollout(
    task: dict,
    policy: Policy,
    *,
    storage_state: Path | None = None,
    max_steps: int = 12,
    headed: bool = False,
) -> dict:
    """Run observation -> action -> result until exit or the step bound."""
    steps: list[dict] = []
    answer = None
    async with open_page(
        task["start_url"], headed=headed, storage_state=storage_state
    ) as page:
        observation = await render_observation(page)
        for _ in range(max_steps):
            decision = await policy(task["intent"], steps, observation)
            action = decision["action"]
            result = await execute_action(page, action)
            next_observation = (
                observation if result["terminal"] else await render_observation(page)
            )
            steps.append(
                {
                    "observation": observation,
                    "action": action,
                    "generation": decision.get("generation"),
                    "result": result,
                    "next_observation": next_observation,
                }
            )
            observation = next_observation
            if result["terminal"]:
                answer = result["output"]
                break
        terminal_score = await verify_task(task, page, answer)
    rewards = rollout_rewards(terminal_score, steps, max_steps)
    return {
        "task_id": task["task_id"],
        "answer": answer,
        "steps": steps,
        "terminal_score": terminal_score,
        "reward_policy": REWARD_POLICY,
        "reward": rewards[REWARD_POLICY],
        "rewards": rewards,
    }


def scripted_policy(actions: list[str]) -> Policy:
    """Return a deterministic policy for local evaluator calibration."""

    async def policy(_intent: str, history: list[dict], _observation: str) -> dict:
        return {"action": actions[min(len(history), len(actions) - 1)]}

    return policy
