"""Launch and render the browser state seen by the policy."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright


async def render_observation(page: Page) -> str:
    """Return visible text plus numbered elements understood by the action executor."""
    snapshot = await page.evaluate(
        """() => {
            document.querySelectorAll('[data-browseruse-id]').forEach(
                element => element.removeAttribute('data-browseruse-id')
            );
            const selector = 'a,button,input,textarea,select,[role="button"],[contenteditable="true"]';
            const visible = [...document.querySelectorAll(selector)].filter(element => {
                const style = getComputedStyle(element);
                const box = element.getBoundingClientRect();
                return style.visibility !== 'hidden' && style.display !== 'none' && box.width && box.height;
            });
            const escape = value => String(value ?? '')
                .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;');
            const elements = visible.map((element, index) => {
                element.setAttribute('data-browseruse-id', String(index));
                const tag = element.tagName.toLowerCase();
                const label = element.innerText || element.value || element.getAttribute('aria-label')
                    || element.getAttribute('title') || element.getAttribute('placeholder') || '';
                return `<${tag} id="${index}">${escape(label.trim())}</${tag}>`;
            });
            return {text: document.body.innerText.trim(), elements};
        }"""
    )
    return (
        f"URL: {page.url}\n\n"
        f"<html><body>\n{snapshot['text']}\n\n"
        f"<!-- interactive elements -->\n{'\n'.join(snapshot['elements'])}\n"
        "</body></html>"
    )


@asynccontextmanager
async def open_page(
    url: str, *, headed: bool = False, storage_state: Path | None = None
):
    """Open one isolated browser page and close its process afterward."""
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("browser world only accepts http(s) URLs")
    if storage_state is not None and not storage_state.is_file():
        raise ValueError(f"storage state does not exist: {storage_state}")
    playwright: Playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=not headed)
    try:
        context: BrowserContext = await browser.new_context(
            storage_state=str(storage_state) if storage_state else None
        )
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        yield page
    finally:
        await browser.close()
        await playwright.stop()


async def observe_url(
    url: str, *, headed: bool = False, storage_state: Path | None = None
) -> str:
    """Open one isolated browser world and return its initial observation."""
    async with open_page(url, headed=headed, storage_state=storage_state) as page:
        return await render_observation(page)


async def capture_storage_state(url: str, output: Path) -> Path:
    """Let a human log in once, then persist Playwright cookies/local storage."""
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("browser world only accepts http(s) URLs")
    output.parent.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded")
            await asyncio.to_thread(
                input, "Log in in Chromium, then press Enter here: "
            )
            await context.storage_state(path=str(output))
        finally:
            await browser.close()
    return output
