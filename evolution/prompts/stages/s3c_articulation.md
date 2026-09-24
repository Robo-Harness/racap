# Stage S3C — Articulated mechanisms

## Objective

Open and close drawers and related mechanisms using contact appropriate to the joint and desired direction.

## Design guidance

- Infer prismatic versus revolute motion from geometry and visual change.
- Opening may need a front-facing handle grasp; closing may be more reliable by pushing the exterior face without grasping.
- Expose contact point, end-effector orientation, motion direction/path and desired semantic state.
- Report observed motion, but let the visual runtime agent decide open/closed state.
- Avoid repeating a long motion when the agent already visually accepts the state.

## Evidence and experiment ladder

Record grounded handle/face, contact mode, EEF orientation, approach direction,
path samples, contact loss, before/after fixture geometry and the agent’s semantic
decision. Test opening and closing separately: they need not share a grasp mode.
For example, an opening hypothesis may require retaining handle contact while a
closing hypothesis may use an exterior-face push—but the rollout, not this hint,
must choose.

Distinguish wrong part grounding, unreachable approach, empty grasp, contact with
the wrong face, incorrect joint/path model, insufficient travel and semantic
verification. Do not let an auxiliary motion estimate repeat a motion after the
visual agent has explicitly accepted the desired state.
