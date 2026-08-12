"""Constrained Browser Use Cloud tools for the hosted evaluator."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from browser_use_sdk.v3 import AsyncBrowserUse
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from playwright.async_api import Browser, Page, Playwright, async_playwright

load_dotenv()

TASKS_PATH = Path(__file__).with_name("tasks.jsonl")
TASKS = {
    row["input_metadata"]["row_id"]: row["input_metadata"]["session_data"]
    for line in TASKS_PATH.read_text().splitlines()
    if (row := json.loads(line))
}

_client: AsyncBrowserUse | None = None
_playwright: Playwright | None = None
_browser: Browser | None = None
_page: Page | None = None
_session_id: str | None = None
_task_id: str | None = None
_last_result: dict | None = None
_lock = asyncio.Lock()


def _api_key() -> str:
    key = os.environ.get("BROWSER_USE_API_KEY") or os.environ.get(
        "BROWSERUSE_API_KEY"
    )
    if not key:
        raise RuntimeError("BROWSER_USE_API_KEY or BROWSERUSE_API_KEY is required")
    return key


async def _stop() -> None:
    """Disconnect Playwright and stop the billable managed browser."""
    global _browser, _client, _page, _playwright, _session_id, _task_id
    browser, client, playwright, session_id = (
        _browser,
        _client,
        _playwright,
        _session_id,
    )
    _browser = _client = _page = _playwright = _session_id = _task_id = None
    try:
        if browser:
            await browser.close()
    finally:
        try:
            if playwright:
                await playwright.stop()
        finally:
            try:
                if client and session_id:
                    await client.browsers.stop(session_id)
            finally:
                if client:
                    await client.close()


def _active_page() -> Page:
    if _page is None:
        raise RuntimeError("call open_task before using browser tools")
    return _page


async def _observe() -> dict:
    page = _active_page()
    elements = await page.locator(
        "a[href], button, input, select, textarea"
    ).evaluate_all(
        """nodes => nodes.map((node, index) => {
          node.dataset.rlId = String(index);
          return {
            id: String(index),
            tag: node.tagName.toLowerCase(),
            label: node.getAttribute('aria-label') || node.innerText || node.name || '',
            type: node.getAttribute('type') || '',
            value: node.value || ''
          };
        })"""
    )
    return {
        "text": (await page.locator("body").inner_text())[:4000],
        "elements": elements,
    }


async def open_task(task_id: str) -> str:
    """Start an isolated managed browser and load a bundled task by ID."""
    global _browser, _client, _last_result, _page, _playwright, _session_id, _task_id
    async with _lock:
        task = TASKS.get(task_id)
        if task is None:
            raise ValueError(f"unknown task_id: {task_id}")
        await _stop()
        _last_result = None
        _client = AsyncBrowserUse(api_key=_api_key())
        session = await _client.browsers.create(
            proxy_country_code=None,
            timeout=5,
            enable_recording=False,
        )
        _session_id = str(session.id)
        if not session.cdp_url:
            await _stop()
            raise RuntimeError("Browser Use did not return a CDP URL")
        try:
            _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.connect_over_cdp(session.cdp_url)
            context = _browser.contexts[0]
            _page = context.pages[0] if context.pages else await context.new_page()
            await _page.set_content(task["html"])
            _task_id = task_id
            return json.dumps(await _observe())
        except BaseException:
            await _stop()
            raise


async def observe() -> str:
    """Return visible text and stable IDs for interactive elements."""
    async with _lock:
        return json.dumps(await _observe())


async def click(element_id: str) -> str:
    """Click one observed element by ID and return the new observation."""
    async with _lock:
        locator = _active_page().locator(f'[data-rl-id="{element_id}"]')
        if await locator.count() != 1:
            raise ValueError(f"unknown element_id: {element_id}")
        await locator.click()
        return json.dumps(await _observe())


async def type_text(element_id: str, text: str) -> str:
    """Replace one observed input's value and return the new observation."""
    async with _lock:
        locator = _active_page().locator(f'[data-rl-id="{element_id}"]')
        if await locator.count() != 1:
            raise ValueError(f"unknown element_id: {element_id}")
        await locator.fill(text)
        return json.dumps(await _observe())


async def finish() -> str:
    """Verify terminal browser state, stop the session, and return reward."""
    global _last_result
    async with _lock:
        if _last_result is not None:
            return json.dumps(_last_result)
        page, task_id = _active_page(), _task_id
        task = TASKS[task_id]
        actual = await page.locator(task["verify_selector"]).get_attribute(
            task["verify_attribute"]
        )
        _last_result = {
            "done": True,
            "task_id": task_id,
            "reward": float(actual == task["verify_value"]),
        }
        await _stop()
        return json.dumps(_last_result)


@asynccontextmanager
async def lifespan(_server: FastMCP):
    try:
        yield
    finally:
        await _stop()


mcp = FastMCP("browser-use-cloud", lifespan=lifespan)
for tool in (open_task, observe, click, type_text, finish):
    mcp.tool()(tool)


async def _smoke() -> None:
    try:
        await open_task("form-submit-001")
        await type_text("0", "violet-731")
        await click("1")
        result = json.loads(await finish())
        assert result["reward"] == 1.0, result
        print(json.dumps(result))
    finally:
        await _stop()


if __name__ == "__main__":
    if sys.argv[1:] == ["--smoke"]:
        asyncio.run(_smoke())
    else:
        mcp.run()
