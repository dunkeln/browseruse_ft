"""Render the browser state seen by the policy."""

from playwright.async_api import Page


async def render_observation(page: Page) -> str:
    """Return the current URL and compact ARIA representation."""
    snapshot = await page.locator("body").aria_snapshot()
    return f"URL: {page.url}\n\n{snapshot}"
