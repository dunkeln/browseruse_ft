"""Launch and render the browser state seen by the policy."""

from urllib.parse import urlparse

from playwright.async_api import Page, async_playwright


async def render_observation(page: Page) -> str:
    """Return the current URL and compact ARIA representation."""
    snapshot = await page.locator("body").aria_snapshot()
    return f"URL: {page.url}\n\n{snapshot}"


async def observe_url(url: str, *, headed: bool = False) -> str:
    """Open one isolated browser world and return its initial observation."""
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("browser world only accepts http(s) URLs")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not headed)
        try:
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded")
            return await render_observation(page)
        finally:
            await browser.close()
