# Stage S1 — Transport ReAct recovery

## Objective

Add runtime visual decision-making around a usable transport API. The agent receives only the instruction and observations and may retry within the configured eight-call transport budget.

## Design guidance

- Re-observe after each physical call and let the agent decide success, retry arguments or stopping.
- Treat geometry and semantic checks as advisory evidence, not vetoes.
- A retry must differ for an evidence-based reason; suppress identical dispatch while a call is still running.
- Invoke reflection after ambiguity or failure, not after every successful substep.
- Record arguments, observations and the reason for changing strategy.

The Policy API owns short physical execution. ReAct owns semantic state, retry choice and final `done`.

## Runtime state and experiment ladder

Maintain an explicit belief for object identity, grasp state, destination state,
last physical call, observed change and remaining budget. ReAct should run after
a completed physical call or a deliberately requested observation—not on a blind
timer and not while the same call is still executing.

On failure, compare before/after images and the API phase report. Ask whether the
target moved, whether it is held, and whether the destination changed. Change one
relevant argument family per retry (grounding wording, grasp family, approach,
destination strategy or correction), while retaining evidence-supported facts.

Track identical redispatch, visual-only loops, false `done`, failure to retry,
and exhausting all motion budget inside one opaque API call as separate agent
failure modes. Geometry estimates may advise; the visual agent owns semantic
acceptance and future subgoals.
