# AGENTS.md

We are making the work of finetuning repeatable and as easy and cost effective as possible while maintaining structured notes in marimo notebooks and fast evals. This is a finetuning flywheel and not a one off project.
Our job is to gather useful insights about the finetuning process and minimizing greenfield with proper ablations and rigorous approach.

We aim at being interpretable and fast.

## Browser automation objective

We are testing whether targeted fine-tuning can make a small language model a reliable, lower-cost, lower-latency browser automation policy. Matching a frontier model is a hypothesis to test, not a presumed outcome.

+ Train from execution-backed browser trajectories that preserve the observation, action, result, and final task outcome.
+ Prefer executable browser actions or code over coordinate imitation when the environment supports them, but treat the action representation as an ablation rather than a fixed conclusion.
+ Measure the untuned student, each training stage, and strong frontier baselines on identical held-out tasks and budgets. Report task success, invalid actions, steps, tokens, latency, cost, regressions, and failure categories.
+ Optimize verified task completion first. Reward fewer steps, tokens, or lower latency only among successful trajectories; a page-state change alone is not evidence of success.
+ Begin with the smallest useful comparison: baseline versus direct supervised fine-tuning. Add curriculum learning, distillation, or reinforcement learning only when the preceding result identifies a concrete failure they can address.
+ Keep generated browser code isolated from secrets, the host filesystem, and unrestricted network access.
