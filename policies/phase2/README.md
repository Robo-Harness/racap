# RACaP Phase 2

This directory is the frozen policy produced by autonomous self-evolution from
the Phase 1 starting point. It is the main RACaP artifact reported in the paper.

The runtime entry point remains `solution.controller.run_episode`, and the six
typed Policy APIs keep the same contracts as Phase 1. Phase 2 adds a thin set of
promoted, visually conditional changes instead of replacing the mature physical
substrate:

- shallow tray-like destinations convert unsafe default rim drops into centered
  interior releases while preserving explicit ReAct arguments;
- `memory/strategy_priors.json` supplies advisory grasp and placement strategies
  for physical classes whose geometry creates repeatable failure modes;
- complete instructions can retrieve routing priors for Full ReAct;
- independent anytime objectives can replan after one failed subgoal rather
  than ending the complete episode.

Every memory entry states a visible condition and an override rule. The ReAct
agent must inspect current evidence and may accept, reject, or revise the
suggestion. This keeps instance-level choice at runtime while preserving
general mechanisms in code.

Phase 2 achieves 49/90 (54.4%) on LIBERO-90. On the stronger transfer tests it obtains 45.0% on
zero-shot LIBERO-PRO and 46.0% on LIBERO-Long.

Run this snapshot through the repository-level wrapper:

```bash
bash scripts/evaluate.sh phase2 phase2_full90 --seeds 0
```

See `PHASE2_MANIFEST.md` for the policy structure and `PHASE1_COMPARISON_zh-CN.md` for a
code-level Chinese comparison.

