# Robosuite cross-embodiment development

You are adapting a mature RGB-D manipulation controller to a new embodiment and
task family.  The starting controller is the frozen LIBERO-90 RACaP Phase 2,
not a toy seed.  Preserve every useful abstraction unless paired rollout
evidence shows that it is incompatible with this embodiment.

The development cohort contains five mechanism families: lifting a rigid
object, stacking, fitting a rigid part onto a fixture, wiping with a mounted
tool, and coordinated bimanual lifting.  Two other mechanism families and all
official evaluation seeds are sealed from the coding agent, critic, memory,
and promotion logic.

Use first-hand RGB-D/video evidence and public tool reports to find the earliest
causal failure.  Prefer changes that explain several seeds or mechanism
families.  Good directions may include embodiment-neutral grasp orientation,
contact-aware transit, support geometry, compliant insertion, path coverage,
and explicit arm assignment.  These are examples rather than a required API
list; add, remove, or reshape Policy APIs when rollout evidence justifies it.

Maintain the RACaP boundary:

- Policy APIs own reusable closed-loop physical mechanisms and expose strategy,
  geometry, tolerances, arm choice, and recovery parameters.
- The runtime agent owns semantic interpretation, causal subgoal ordering,
  parameter choice, retries, and stopping from current visual evidence.
- Memory records conditional experience, including where an API is reliable or
  outside its capability envelope.  It must not become a seed/task lookup table.

Never query reward, native predicates, simulator object poses, task ids, seeds,
or evaluator files.  Candidate observations are limited to the documented
public runtime.  A candidate is promoted only by paired native success on the
fixed development cohort; offline native diagnostics may be read by the critic
but cannot be exposed to runtime code.
