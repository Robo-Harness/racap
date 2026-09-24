# RACaP source release

This package contains the RACaP runtime, its Phase 1 and Phase 2 policies,
policy evolution, evaluation tools, tests, and third-party source dependencies.

Phase 1 provides capability curriculum learning. Phase 2 applies autonomous
self-evolution to the Phase 1 policy. Both use the same public Policy API contracts.

Included resources are source code, prompts, algorithm memory, task definitions,
and aggregate evaluation tables. Credentials, training datasets, model weights,
simulator initial states, generated skill libraries, raw trajectories, videos,
caches, local environments, and Git metadata are excluded.

External assets are declared in `configs/external_assets.json`. Provision them
from an authorized local asset directory with `scripts/prepare_assets.py`.
See `docs/REPRODUCIBILITY.md` for installation and evaluation.
