# Evaluation provenance and claim boundaries

This source package contains curated aggregate results, protocols, public task
manifests, and analysis code. Raw trajectories and videos are not duplicated in
the source release. New evaluations reproduce their artifact schema under
`outputs/`, which is ignored by version control.

## Frozen RACaP artifacts

- **Phase 1** is the end of capability curriculum learning. The earlier retained
  LIBERO-90 evaluation obtained 48/90 native successes. A later fully
  instrumented replay obtained 42/90. Both are reported because the difference
  measures residual stochastic rollout variation rather than a source change.
- **Phase 2** applies autonomous self-evolution to Phase 1 and obtains
  49/90 native successes in the reported LIBERO-90 evaluation.

The public policy code has no access to native predicates, simulator object
identities, or privileged joint state. Native predicates are used only by the
evaluator and by offline candidate selection.

## Main protocols

- **LIBERO-90:** 90 tasks, seed 0. This is the shared development distribution
  and the in-domain score.
- **LIBERO-PRO:** six suites covering object, goal, and spatial perturbations.
  Each suite has 10 tasks and uses seeds 0, 1, and 2. All methods therefore run
  the same 180 episodes. Zero-shot means that no target-domain rollout can
  update persistent code, memory, prompts, skills, or parameters. Temporary
  within-episode adaptation is allowed and discarded after the episode.
- **LIBERO-Long:** 10 tasks and five seeds, giving 50 episodes per method.
- **Robosuite:** seven task types and five sealed seeds, giving 35 episodes per
  method. Frozen transfer and post-development results are separate protocols.

Every ordinary episode uses one reset, an 8000 simulator-step horizon, and a
1000-second policy clock. Native simulator predicates are the sole final
success measure.

## Baseline scope

CaP-X writes task-specific Python over the common non-privileged perception and
low-level control interface. RATS-base uses the released executor with learned
skill reuse and failure memory disabled. RATS-90 uses the same executor with a
read-only skill library and failure memory from 15 LIBERO-90 development rounds.
These are controlled comparisons and not reproductions of the original RATS
paper setting.

## Development accounting

Phase 2 made 32 proposals, tested 23 candidates in simulation, promoted nine,
and consumed 596 development episodes. RATS-90 used 15 self-play rounds under
the same 596-reset ceiling. Its committed lineage consumed 206 resets and 2474
hosted calls. The interaction ceiling is shared, but proposals, tokens, cost,
and wall time are not exactly matched.

Exact aggregate rows are in `paper_results.csv`, and the six LIBERO-PRO suites
are separated in `libero_pro_breakdown.csv`. Executable comparison definitions are in
`controlled_comparison/protocol.yaml`.
