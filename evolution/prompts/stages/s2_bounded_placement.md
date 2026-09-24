# Stage S2 — Identity grounding and bounded placement

## Objective

Handle visually ambiguous instances and destinations whose usable interior is smaller than the visible bounding box.

## Design guidance

- Generate VLM candidates, inspect independent visual evidence, and re-ground with explicit negative feedback when identity is doubtful.
- Preserve relational phrases such as front/back, left/right and compartment qualifiers.
- Estimate object footprint, yaw, carry offset and uncertainty.
- Erode the destination opening by the transformed object footprint to obtain a feasible object-centre region.
- If the region is empty, expose that fact instead of repeatedly aiming at the visible centre.

Avoid object-name coordinate tables and benchmark-specific aliases without visual evidence.

## Evidence and experiment ladder

For ambiguous sources, retain multiple candidates with crop/point evidence and
record why one was accepted or rejected. A failed lift is not proof of wrong
identity; separate candidate identity from grasp execution. Negative feedback to
re-grounding should describe the visible mismatch, not merely say “wrong”.

For destinations, distinguish visible region, opening polygon, support surface
and feasible centre set. Record footprint dimensions, yaw assumption, uncertainty
margin, selected centre and whether the feasible set is empty. Test geometry on
families of different object/opening sizes so apparent gains cannot come from a
single offset.

Typical anti-patterns are grounding an entire multi-compartment fixture, aiming
at its box centre, treating a point-inside test as full containment, and retrying
the same infeasible centre after a visually correct but mechanically invalid drop.
