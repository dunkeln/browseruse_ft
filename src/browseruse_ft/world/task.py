"""Resolve WebArena task URLs and Playwright authentication state."""

import os
from pathlib import Path
from urllib.parse import urlparse

SITE_ENV = {
    "__SHOPPING__": "SHOPPING",
    "__SHOPPING_ADMIN__": "SHOPPING_ADMIN",
    "__REDDIT__": "REDDIT",
    "__GITLAB__": "GITLAB",
    "__MAP__": "MAP",
    "__WIKIPEDIA__": "WIKIPEDIA",
    "__HOMEPAGE__": "HOMEPAGE",
}


def resolve_task(task: dict, *, root: Path = Path(".")) -> tuple[dict, Path | None]:
    """Resolve placeholders from environment variables without mutating the source row."""
    resolved = dict(task)

    def resolve(value: str) -> str:
        for placeholder, variable in SITE_ENV.items():
            if placeholder in value:
                base = os.environ.get(variable)
                if not base:
                    raise ValueError(f"{variable} is required for {placeholder}")
                value = value.replace(placeholder, base.rstrip("/"))
        return value

    resolved["start_url"] = resolve(resolved["start_url"])
    if urlparse(resolved["start_url"]).scheme not in {"http", "https"}:
        raise ValueError(f"invalid task start URL: {resolved['start_url']}")

    evaluation = dict(resolved["eval"])
    if evaluation.get("reference_url"):
        evaluation["reference_url"] = resolve(evaluation["reference_url"])
    evaluation["program_html"] = [
        {**target, "url": resolve(target["url"])}
        if target["url"] != "last" and not target["url"].startswith("func:")
        else dict(target)
        for target in evaluation.get("program_html", [])
    ]
    resolved["eval"] = evaluation

    state = resolved.get("storage_state")
    storage_state = (root / state).resolve() if state else None
    if resolved.get("require_login") and (
        storage_state is None or not storage_state.is_file()
    ):
        raise ValueError(f"authentication state is required: {storage_state}")
    return resolved, storage_state
