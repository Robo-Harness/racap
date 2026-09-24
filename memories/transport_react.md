# Transport ReAct experience

- This controller owns only the current source-to-destination transport. Never
  repeat open/close, appliance-control, or stack prerequisites from the full
  instruction; the parent controller already handled them.
- Preserve the exact source and destination phrases supplied by the parent.
  For similar objects, verify category in a crop and spatial instance in the
  full scene; do not drop front/middle/back/left/right.
- Use insert for bounded slots, shelf apertures, racks, cubbies, drawers, and
  named compartments; use pickplace for exposed support surfaces and broad
  containers. Let insert preflight decide whether a broad fallback is safe.
- After any release, check before another motion. A successful check ends the
  subgoal. A reliable supported miss of roughly 0.5–5 cm should use the exact
  measured push, followed by one check, rather than a destructive re-grasp.
- A retry must change one supported control. Valid grasp families are
  top_down, rim, pca_axis, affordance, and graspnet, optionally with @top/@low.
  Valid release controls are numeric nudge `[right, up]`, scalar grasp_depth or
  place_margin, release_on `rim|floor`, yaw_deg, centre, and compensate. Never
  invent “handle”, “side”, “edge”, prose nudges, or string-valued margins.
- On empty grasp before any release, use one materially different analytic
  ladder, then one learned graspnet attempt. Empty grasp after an earlier
  release is evidence that the object may already be at the destination:
  inspect the current scene and check; do not blindly restart the ladder.
- In a broad basket or tray, the released object can be occluded by the rim.
  If the gripper is reported empty and the original source object disappeared,
  do not claim it is still held merely because a rebound mask has implausible
  height. Preserve the scene unless the before/after images clearly show the
  requested object outside the destination.
- For compartments, full visible footprint containment is stronger than centre
  overlap. But after a failed re-grasp, if the unchanged image clearly shows
  the object resting in the named compartment, return `done` rather than
  repeatedly trying to pick an already inserted object.
- Never repeat a passive look unless it can test a new verified alias. Never
  turn an unreliable centimetre estimate into an open-loop nudge.
