"""Render the browser state shared by policy-training methods."""

SYSTEM_PROMPT = (
    "You are a browser-use agent. Complete the browser task by returning "
    "the next valid browser action."
)


def render_policy_messages(example: dict) -> list[dict[str, str]]:
    """Build the prompt shared by SFT and later RFT examples."""
    history = [
        f"{index}. {step['action']}"
        for index, step in enumerate(example["history"], start=1)
    ]
    prompt = (
        f"Goal:\n{example['task']}\n\n"
        f"Previous actions:\n{'\n'.join(history) if history else 'None'}\n\n"
        f"Current page:\n{example['current_observation']}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
