# Stage S3B — Support-aware stacking

## Objective

Place one object stably on another instead of treating the support as a generic destination box.

## Design guidance

- Estimate support top height and both object footprints.
- Plan overlap and centre-of-mass margin under pose uncertainty.
- Release with minimal drop, retreat without sweeping the stack, and visually check stability.
- Distinguish grounding, grasp, overlap, collision and stability failures.

## Evidence and experiment ladder

Record source footprint, support top polygon/height, predicted overlap margin,
release height, object motion after opening the gripper and state after retreat.
First make a roomy support stable, then vary footprint ratio, source orientation
and uncertainty. Preserve a correction interface for ReAct rather than embedding
support-specific offsets.

Typical anti-patterns are using the support visual centre without checking usable
top surface, releasing from a large drop height, sweeping the stack during
retreat, and claiming success before settling. A stable stack requires sustained
support after the robot clears the view.
