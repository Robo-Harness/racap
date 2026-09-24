# Stage S5 — Long-horizon causal composition

## Objective

Solve tasks with multiple active goals and dependencies, such as opening a receptacle, placing an object, then closing it.

## Design guidance

- Build an ordered subgoal graph from the instruction and visual state.
- Do not close or deactivate a receptacle before dependent placement completes.
- Re-observe after every physical tool call and update semantic state from agent vision.
- Auxiliary measurements and validators inform the agent but cannot overrule its explicit state decision.
- Preserve enough simulator and tool budget for downstream goals.

## Planning state and experiment ladder

Represent active goals, prerequisites, achieved beliefs, supporting observations,
last action and remaining budget. Derive dependency edges from containment and
mechanism access rather than merely following phrase order. Re-plan after each
completed physical call using the new image; do not replay the initial plan when
the world has changed.

Diagnose wrong initial ordering, stale achieved state, premature close/off action,
failed subgoal recovery, downstream budget starvation and final false stopping
separately. Evidence across clear dependencies, varying language order and
intermediate failures can reveal different planning mechanisms. The agent must be able to continue when one
subgoal succeeds visually even if an auxiliary measurement disagrees.
