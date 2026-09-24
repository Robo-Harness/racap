# RACaP Phase 2 policy

Phase 2 applies autonomous self-evolution to the Phase 1 policy.
The entry point is `solution.controller.run_episode`; it uses the RACaP runtime
and the six Policy APIs in `racap/policy_api/`.

The policy consists of `solution/`, advisory resources in `memory/`, and
contract tests in `tests/`. The controller adds conditional placement defaults,
strategy retrieval, instruction-level routing memory, and continuation for
independent objectives. All runtime inputs follow the public observation contract.

Evaluation definitions and aggregate results are in `experiments/`.
