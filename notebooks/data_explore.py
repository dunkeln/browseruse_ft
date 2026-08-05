import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluating WebArena SFT
    """)
    return


@app.cell
def _():
    import marimo as mo
    from browseruse_ft.data import audit, load_hub_dataset

    return load_hub_dataset, mo


@app.cell
def _(load_hub_dataset):
    sample_rows = 1_000
    dataset = load_hub_dataset("webarena_lite_sft", split="train", rows=sample_rows).to_polars()
    return (dataset,)


@app.cell(hide_code=True)
def _(dataset, mo):
    _df = mo.sql(
        f"""
        SELECT
            problem,
            step.round as round,
            step.action as action,
            step.remarks as remark,
            observation,
            solution,
            remarks as solution_remarks
        FROM dataset
        CROSS JOIN UNNEST(history) AS exploded(step);
        """
    )
    return


@app.cell(hide_code=True)
def _(dataset, mo):
    _df = mo.sql(
        f"""
        select * from dataset;
        """
    )
    return


@app.cell
def _(dataset):
    dataset.select("problem", "history")
    return


if __name__ == "__main__":
    app.run()
