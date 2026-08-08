import marimo

__generated_with = "0.23.16"
app = marimo.App(width="columns")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    return Path, alt, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reward calibration

    Inspect the execution-backed probes produced by `uv run browseruse-rft-smoke`.
    A successful task must stay above every failure; shaping only distinguishes
    successful trajectories by invalid actions or excess steps.

    This calibration uses WebArena task `0` and its official exact-answer verifier
    against a controlled local page. It does not claim canonical Magento/WebArena
    environment coverage.
    """)
    return


@app.cell
def _(Path, pl):
    artifact = Path(__file__).parents[1] / "data/eval/reward-distribution.jsonl"
    if not artifact.is_file():
        raise FileNotFoundError("Run `uv run browseruse-rft-smoke` first.")
    rewards = pl.read_ndjson(artifact)
    return artifact, rewards


@app.cell(hide_code=True)
def _(artifact, mo, rewards):
    mo.vstack(
        [
            mo.md(f"**Artifact:** `{artifact}`"),
            mo.ui.table(rewards),
        ]
    )
    return


@app.cell
def _(rewards):
    reward_columns = [
        "terminal_only",
        "terminal_with_safety",
        "terminal_with_efficiency",
    ]
    distribution = rewards.unpivot(
        on=reward_columns,
        index="label",
        variable_name="reward_policy",
        value_name="reward",
    )
    return (distribution,)


@app.cell(hide_code=True)
def _(alt, distribution, mo):
    chart = (
        alt.Chart(distribution)
        .mark_bar()
        .encode(
            x=alt.X("reward:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("label:N", sort=None, title=None),
            color=alt.Color("reward_policy:N", title="Reward policy"),
            column=alt.Column("reward_policy:N", title=None),
            tooltip=["label", "reward_policy", alt.Tooltip("reward:Q", format=".3f")],
        )
        .properties(width=220, height=180)
    )
    mo.ui.altair_chart(chart)
    return


@app.cell(hide_code=True)
def _(mo, rewards):
    successful = rewards.filter(rewards["terminal_score"] == 1)
    failures = rewards.filter(rewards["terminal_score"] == 0)
    reward_policy = rewards["reward_policy"][0]
    separation = successful["reward"].min() - failures["reward"].max()
    mo.md(
        f"""
        **Selected reward:** `{reward_policy}`. Minimum successful reward is
        `{successful["reward"].min():.3f}`; maximum failed reward is
        `{failures["reward"].max():.3f}`; separation is `{separation:.3f}`.

        `invalid_then_correct` exposes the intended distinction: terminal correctness
        remains `1.0`, while the safety-shaped reward drops below the clean success.
        The efficiency-only candidate ranks that invalid trajectory above the longer
        clean trajectory, so it is not safe to promote by itself.
        """
    )
    return


if __name__ == "__main__":
    app.run()
