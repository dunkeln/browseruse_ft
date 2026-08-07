<div align="center">
  <h1>Fine-tuning for long-horizon browser tasks</h1>
</div>

*Model fine-tuning often optimizes isolated responses. Here, we are exploring how to fine-tune for long-horizon tasks. The environment is a browser, where a subagent learns a policy from observations, actions, and their outcomes.*

---

The first supervised run is complete. It gives a trained adapter and an encouraging learning signal, not yet a successful browser agent.

| First run | Outcome |
|---|---|
| Model | Qwen3.5-9B with a rank-8 LoRA |
| Dataset | WebArena-Lite-SFT |
| Examples | 4,957 train · 2,523 validation |
| Training | 1 epoch · 1,240 steps · 19.83M tokens |
| Held-out result | 0.298 cross-entropy loss · 1.35 perplexity |
| Fireworks cost | $9.91 for the training job |

## Learning one decision at a time

Each example captures a decision within a longer browser trajectory:

```latex
goal + previous actions + current HTML  \rightarrow next browser action
```

The model is not asked to solve the entire task in one response. At inference time, a browser harness executes its action, observes the new page, extends the history, and asks again. Repeating that loop is what turns next-action prediction into long-horizon behavior.

The prompt mirrors that loop. A short system message defines the browser-policy role; the user message supplies the goal, prior actions, and current page. The assistant returns only the next executable action. We deliberately leave expert remarks and hidden reasoning out of the target so the model learns the interface it will use in production.

## Reproducing the run

Prepare fold 0:

```bash
uv run browseruse-sft --split train --fold 0 \
  --output data/processed/webarena-lite-fold-0-train.jsonl

uv run browseruse-sft --split validation --fold 0 \
  --output data/processed/webarena-lite-fold-0-validation.jsonl
```

Review the TOML, then start the paid Fireworks job:

```bash
uv run browseruse-train configs/sft/001-webarena-fold-0.toml --confirm
```

The command validates both artifacts, reuses matching remote resources, shows one live terminal progress display, sends training metrics to Weights & Biases, and records the completed run locally.

## The next proof

RFT!!!!!

## References

- [WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning](https://arxiv.org/html/2505.16421v2)
- [DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning](https://arxiv.org/abs/2406.11896)
- [WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning](https://openreview.net/pdf?id=oVKEAFjEqv)
