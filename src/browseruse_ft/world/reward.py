"""Deterministic WebArena verification and bounded rollout rewards."""

import html
from urllib.parse import parse_qs, urlparse

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page

REWARD_POLICY = "terminal_with_safety"


def _clean(value: str) -> str:
    value = value.strip()
    if len(value) > 1 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return value.lower()


def verify_answer(reference_answers: dict, answer: str) -> float:
    """Mirror WebArena's deterministic exact and must-include string checks."""
    score = 1.0
    if "exact_match" in reference_answers:
        score *= float(_clean(answer) == _clean(reference_answers["exact_match"]))
    for required in reference_answers.get("must_include", []):
        score *= float(_clean(required) in _clean(answer))
    if "fuzzy_match" in reference_answers:
        raise ValueError("fuzzy_match requires a separately reviewed judge")
    return score


def verify_url(reference: str, actual: str) -> float:
    """Mirror WebArena's GOLD-in-PRED URL rule."""
    actual_url = urlparse(actual.rstrip("/"))
    for candidate in reference.split(" |OR| "):
        expected = urlparse(candidate.rstrip("/"))
        if expected.netloc + expected.path not in actual_url.netloc + actual_url.path:
            continue
        expected_query = parse_qs(expected.query)
        actual_query = parse_qs(actual_url.query)
        if all(
            any(value in actual_query.get(key, []) for value in values)
            for key, values in expected_query.items()
        ):
            return 1.0
    return 0.0


async def verify_task(task: dict, page: Page, answer: str | None) -> float:
    """Run every configured deterministic evaluator and multiply its result."""
    evaluation = task["eval"]
    score = 1.0
    for kind in evaluation["eval_types"]:
        if kind == "string_match":
            score *= verify_answer(evaluation["reference_answers"], answer or "")
        elif kind == "url_match":
            score *= verify_url(evaluation["reference_url"], page.url)
        elif kind == "program_html":
            for target in evaluation["program_html"]:
                if target["url"].startswith("func:"):
                    raise ValueError(
                        "functional WebArena target URLs are not supported"
                    )
                if target["url"] != "last":
                    await page.goto(target["url"], wait_until="domcontentloaded")
                locator = target["locator"].strip()
                selected = await page.content() if not locator else ""
                if locator.startswith(("document.", "[...document.")):
                    try:
                        selected = str(await page.evaluate(f"() => {locator}"))
                    except PlaywrightError:
                        selected = ""
                elif locator:
                    raise ValueError(f"unsupported program_html locator: {locator}")
                selected = html.unescape(selected)
                required = target["required_contents"]
                if "exact_match" in required:
                    score *= float(_clean(selected) == _clean(required["exact_match"]))
                else:
                    for value in required.get("must_include", []):
                        score *= float(
                            any(
                                _clean(option) in _clean(selected)
                                for option in value.split(" |OR| ")
                            )
                        )
        else:
            raise ValueError(f"unsupported evaluator: {kind}")
    return score


def rollout_rewards(terminal_score: float, steps: list[dict], max_steps: int) -> dict:
    """Return verifier-gated reward candidates.

    ``terminal_only`` is the unshaped task-verifier score.
    ``terminal_with_safety`` slightly penalizes invalid actions after success.
    ``terminal_with_efficiency`` slightly penalizes extra steps after success.
    Every candidate remains zero when terminal verification fails.
    """
    invalid = sum(not step["result"]["success"] for step in steps)
    used = len(steps)
    return {
        "terminal_only": terminal_score,
        "terminal_with_safety": terminal_score
        * (0.9 + 0.1 * (1 - invalid / max(used, 1))),
        "terminal_with_efficiency": terminal_score
        * (0.9 + 0.1 * (1 - max(used - 1, 0) / max(max_steps - 1, 1))),
    }
