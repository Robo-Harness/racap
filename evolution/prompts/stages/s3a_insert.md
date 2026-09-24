# Stage S3A — Constrained insertion

## Objective

Develop a distinct API for elongated or tightly fitting objects in caddies, racks and shelves where centre-drop pickplace is inadequate.

## Design guidance

- Expose approach direction, object yaw, insertion depth, clearance margin and correction controls.
- Reuse the footprint-aware feasible region from the parent stage.
- Plan pre-alignment, guarded approach, partial insertion, release and retreat as observable phases.
- When top grasp blocks the opening, let the agent choose side/edge grasp or contact-guided insertion.

Do not special-case individual book or compartment task IDs.

## Evidence and experiment ladder

Record object long axis, opening axis, pre-insertion alignment error, clearance,
contact onset, achieved depth, release and retreat. Diagnose failure at the first
of: source grasp cannot preserve useful yaw; pre-alignment is outside the opening;
the transformed footprint has no clearance; contact deflects the object; release
or retreat extracts it again.

Evidence from roomy insertions, narrow openings and alternate orientations can
help separate mechanisms. Let ReAct choose exposed grasp side, yaw, approach
vector, depth and correction from current vision. If no feasible region exists,
report it and expose alternative grasp/approach families for runtime selection.

Success means the object remains constrained after release and retreat, not only
that its centre crossed an opening plane.
