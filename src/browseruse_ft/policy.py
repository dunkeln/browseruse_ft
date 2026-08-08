"""Model adapter shared by local and remote browser rollouts."""

import time

from fireworks import AsyncFireworks

from browseruse_ft.preprocess.policy import render_policy_messages
from browseruse_ft.world.rollout import Policy


def fireworks_policy(
    client: AsyncFireworks,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Policy:
    """Return the production-aligned next-action policy used during rollouts."""

    async def policy(intent: str, history: list[dict], observation: str) -> dict:
        messages = render_policy_messages(
            {
                "task": intent,
                "history": history,
                "current_observation": observation,
            }
        )
        started = time.perf_counter()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("model returned an empty action")
        usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
        return {
            "action": content.strip(),
            "generation": {
                "response_id": response.id,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "usage": usage,
            },
        }

    return policy
