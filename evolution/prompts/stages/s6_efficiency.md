# Stage S6 — Perturbation robustness and efficiency

## Objective

Preserve native success while reducing latency, redundant perception, simulator motion and unnecessary reflection under clean perturbations.

## Design guidance

- Remove duplicate VLM checks when RGB-D or an existing observation answers the question.
- Cache stable grounding within one physical state and invalidate it after motion.
- Trigger reflection only on failure or consequential ambiguity.
- Compare efficiency using evaluator-measured simulator steps, VLM calls and public motion calls, never candidate-reported counters.
- Do not trade away capability success for lower cost in the capability champion.

## Measurement and experiment ladder

Use paired evaluator-owned counts and episode-level deltas. Attribute calls to
task planning, grounding, physical verification, retry reflection and redundant
semantic confirmation. Remove one redundancy class at a time and predict which
episodes should keep identical native outcomes.

Prefer event-triggered reasoning: plan once, act, observe once, and reflect only
when the observation is ambiguous or contradicts the expected state. Cache an
observation only while no physical action can invalidate it. Track regressions,
because a lower mean cost caused by early failure is not an efficiency gain.
