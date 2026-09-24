# Stage S0 — Direct exposed transport

## Objective

Improve object transport on a cohort where the evaluator supplies source and
destination phrases parsed from the instruction. This makes physical manipulation
the primary evidence focus without restricting what abstractions the agent may add.

## Development tasks

Simple object-to-basket, object-to-tray and bowl-to-support relations: 9, 10, 30, 36, 46–49 and 55–59.

## Design guidance

- Keep one transparent transport API: localize, grasp, clear the scene, transit, place and go home.
- Expose grasp family, destination strategy and offsets to future agents.
- Return distinct evidence for source grounding, grasp, transit and placement.
- Prefer a high collision-clearing transit over a table-level straight line.

## Evidence and experiment ladder

Instrument source candidate, chosen grasp point/orientation, gripper closure,
post-lift evidence, clearance height, destination candidate, release pose and
retreat. First establish whether failure is perception, contact, transit or
release; do not treat all four as `pickplace_failed`.

Useful hypotheses may concern source identity and reachable grasp, stable lift,
collision-clearing transit, destination grounding, release or go-home. The agent
may introduce any additional abstraction, API or runtime reasoning mechanism
supported by rollout evidence. A transport API is usable when runtime reasoning
can change its meaningful controls without editing the API implementation.

Generalize Policy API placement controls from source footprint/height/orientation,
destination free-space and containment clearance. ReAct experience memory may
record named grocery-item priors, provided the visual agent checks whether the
associated physical condition applies and dispatches an exposed placement mode;
do not bury those names in the Policy API as an unconditional strategy whitelist.

Typical anti-patterns are a centroid that is not a graspable surface, one fixed
EEF orientation for every visible geometry, table-level transit across fixtures,
and declaring success from command completion rather than post-action vision.
