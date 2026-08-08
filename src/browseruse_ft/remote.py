"""Fireworks remote-rollout endpoint for the browser world."""

import json
import logging
import os
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="browseruse-ft")

def _task_from_request(request) -> dict:
    if not request.messages:
        raise ValueError("messages must contain a browser task")
    content = request.messages[-1].content
    if not isinstance(content, str):
        raise TypeError("the final message content must be browser-task JSON")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise TypeError("browser-task JSON must be an object")
    task = payload.get("task", payload)
    if not isinstance(task, dict):
        raise TypeError("task must be an object")
    missing = {"task_id", "intent", "start_url", "eval"} - task.keys()
    if missing:
        raise ValueError(f"browser task is missing fields: {sorted(missing)}")
    return task


def _validate_remote_urls(request, task: dict) -> None:
    tracing = urlparse(request.model_base_url or "")
    if tracing.scheme != "https" or tracing.hostname != "tracing.fireworks.ai":
        raise ValueError("model_base_url must use https://tracing.fireworks.ai")

    allowed = {
        host.strip().lower()
        for host in os.environ.get("BROWSERUSE_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    }
    if not allowed:
        raise RuntimeError("BROWSERUSE_ALLOWED_HOSTS is required")

    evaluation = task["eval"]
    urls = [task["start_url"]]
    urls.extend(evaluation.get("reference_url", "").split(" |OR| "))
    urls.extend(
        target["url"]
        for target in evaluation.get("program_html", [])
        if target["url"] != "last" and not target["url"].startswith("func:")
    )
    rejected = set()
    for url in filter(None, urls):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            rejected.add(url)
        elif parsed.hostname.lower() not in allowed:
            rejected.add(parsed.hostname)
    if rejected:
        raise ValueError(f"task URL hosts are not allowed: {sorted(rejected)}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/init")
async def init(payload: dict):
    from eval_protocol import (  # Lazy to keep Playwright container startup bounded.
        FireworksTracingHttpHandler,
        InitRequest,
        RolloutIdFilter,
        Status,
    )
    from fireworks import AsyncFireworks

    from browseruse_ft.policy import fireworks_policy
    from browseruse_ft.world.rollout import run_rollout
    from browseruse_ft.world.task import resolve_task

    request = InitRequest.model_validate(payload)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(
        isinstance(handler, FireworksTracingHttpHandler)
        for handler in root_logger.handlers
    ):
        root_logger.addHandler(FireworksTracingHttpHandler())
    logger = logging.getLogger(f"browseruse_ft.remote.{request.metadata.rollout_id}")
    logger.addFilter(RolloutIdFilter(request.metadata.rollout_id))
    try:
        task, storage_state = resolve_task(_task_from_request(request))
        _validate_remote_urls(request, task)

        model = str(request.completion_params.get("model", "")).removeprefix(
            "fireworks_ai/"
        )
        if not model:
            raise ValueError("completion_params.model is required")
        api_key = request.api_key or os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("a Fireworks API key is required")
        max_steps = int(os.environ.get("BROWSERUSE_MAX_STEPS", "12"))
        if not 1 <= max_steps <= 50:
            raise ValueError("BROWSERUSE_MAX_STEPS must be between 1 and 50")
        temperature = float(request.completion_params.get("temperature", 0.0))
        max_tokens = int(request.completion_params.get("max_tokens", 128))
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 1 <= max_tokens <= 2048:
            raise ValueError("max_tokens must be between 1 and 2048")

        async with AsyncFireworks(
            api_key=api_key,
            base_url=request.model_base_url,
        ) as client:
            rollout = await run_rollout(
                task,
                fireworks_policy(
                    client,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                storage_state=storage_state,
                max_steps=max_steps,
            )

        extras = {
            "task_id": rollout["task_id"],
            "answer": rollout["answer"],
            "terminal_score": rollout["terminal_score"],
            "reward_policy": rollout["reward_policy"],
            "reward": rollout["reward"],
            "rewards": rollout["rewards"],
            "steps": [
                {"action": step["action"], "result": step["result"]}
                for step in rollout["steps"]
            ],
        }
        logger.info(
            "browser rollout completed",
            extra={"status": Status.rollout_finished(), "extras": extras},
        )
        return {
            "status": "success",
            "rollout_id": request.metadata.rollout_id,
            "reward": rollout["reward"],
        }
    except Exception as error:  # noqa: BLE001 - Fireworks needs a terminal status.
        logger.error(
            "browser rollout failed: %s",
            error,
            extra={"status": Status.rollout_error(str(error))},
        )
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(error)},
        )
