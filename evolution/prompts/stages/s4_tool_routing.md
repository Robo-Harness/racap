# Stage S4 — Single-goal tool routing

## Objective

Choose among evolved APIs from language, geometry and current visual state. API names remain advisory and may be reorganized.

## Design guidance

- Route by physical constraint: free transport, tight insertion, support stacking, articulation, pushing or compact control.
- Request another observation only when it resolves a concrete routing ambiguity.
- Pass adjustable geometry and strategy arguments instead of hiding every decision in fixed ladders.
- Preserve regression performance of every atomic capability.

## Routing evidence and experiment ladder

Build a capability table from public API contracts and empirical outcomes:
preconditions, adjustable controls, reports, recurring failure modes and current
reliability. It is advisory experience, not a hard-coded mapping from object names
to tools. Route from the physical relation and observed geometry, then pass the
qualifiers in the instruction through to grounding and API arguments.

For each wrong route, compare the selected API’s capability boundary with the
visible constraint: free-space transport, tight opening, support stability,
articulated joint or bounded contact path. Record uncertainty and one observation
that would discriminate between plausible APIs. Do not use repeated `look` calls
without such a question.
