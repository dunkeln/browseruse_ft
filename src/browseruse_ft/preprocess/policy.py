"""Render the browser state shared by policy-training methods."""


SYSTEM_PROMPT = (
    "You are a browser-use agent. Complete the browser task by returning "
    "the next valid browser action."
)


def render_policy_messages(example: dict) -> list[dict[str, str]]:
    """Build the prompt shared by SFT and later RFT examples."""
    history = ["Previous observations omitted."]
    for index, step in enumerate(example["history"], start=1):
        history.append(f"{index}. {step['action']}")
        if step["remarks"]:
            history.append(f"   Remark: {step['remarks']}")
    prompt = (
        f"Goal:\n{example['task']}\n\n"
        f"Previous actions:\n{'\n'.join(history)}\n\n"
        f"Current page:\n{example['current_observation']}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
