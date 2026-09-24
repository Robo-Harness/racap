# Stage S3D — Compact controls and path primitives

## Objective

Operate switches, knobs and small doors using general bounded-contact motion rather than a task-specific monolith.

## Design guidance

- Ground an interaction point and expose line or circular-arc path parameters.
- Maintain contact during a bounded trajectory and retreat to reveal the result.
- Make force and contact assumptions explicit in the report.
- Let the agent visually infer semantic on/off or open/closed state.

## Evidence and experiment ladder

Record the interaction-point crop, surface normal if available, chosen path type,
line direction or arc centre/radius/angle, contact continuity and before/after
visual state. Line, arc or other motion primitives may be introduced whenever
observed mechanism geometry supports them.

Keep the API compact but adjustable: ReAct should be able to reverse direction,
alter travel and select contact mode after visual feedback. Separate failure to
touch the control, loss of contact, wrong path geometry and successful motion with
incorrect semantic interpretation.
