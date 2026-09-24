# Public runtime SDK

The runtime is a non-privileged robot development substrate, not a mature
Policy API. Candidate code can inspect `runtime.capability_manifest()` for the
machine-readable contract.

- Observation: `observe`, the RGB-D compatibility decoder `_camera`, `views`,
  `move_camera`, `reset_cameras`, `crop_frame`, `inspect`.
- Grounding: `localize`, `localize_many`, `localize_across_views`,
  `localize_with_feedback`, `object_points`, pixel/bbox probes and image axes.
- State: `ee_pose`, `verify_grasp`, `verify_place`, `measure`.
- Motion: `move_to`, `delta_move`, `execute_waypoints`, `linear_contact`,
  `arc_contact`, `grasp`, `contact_grasp`, `place`, gripper commands,
  `go_home` and `recover`.

All Cartesian positions use world-frame XYZ metres and all arc angles use
radians. Semantic/motion validators are advisory. Native predicates and hidden
simulator state are evaluator-only.
