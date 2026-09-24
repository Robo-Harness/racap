# RACaP Phase 1

This directory is the frozen policy at the end of capability curriculum
learning. It is the starting point for Phase 2 autonomous self-evolution.

`solution/controller.py` binds Full ReAct and Transport ReAct to the six
candidate-owned wrappers in `solution/policy_api.py`. Those wrappers call the
general mechanisms under `racap/policy_api/`. `memory/object_appearances.json`
provides advisory visual descriptions for ambiguous object categories. Current
RGB-D evidence and tool reports always take precedence over memory.

Phase 1 provides:

- hierarchical task planning and causal subgoal ordering;
- visual transport recovery with typed failure reports;
- `pickplace`, `insert`, `stack`, `push`, `articulate`, and
  `actuate_control`;
- measured RGB-D grounding and negative-feedback re-grounding;
- collision-aware transit, footprint-aware placement, and go-home hooks;
- an 8000-step long-horizon budget with bounded per-tool retry budgets.

The retained Phase 1 LIBERO-90 run obtained 48/90 native successes (53.3%) at
seed 0. A later instrumented replay obtained 42/90 (46.7%), so both values are
kept in the release rather than hiding rollout variation.

Run this snapshot through the repository-level wrapper:

```bash
bash scripts/evaluate.sh phase1 phase1_full90 --seeds 0
```

Native predicates and simulator object state are not available through the
controller or Policy API surface.

