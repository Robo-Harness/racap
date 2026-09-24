# RACaP controlled comparison

This directory is the machine-readable source of truth for the controlled
RACaP, RATS, and CaP-X comparison.  It deliberately separates development
exposure from frozen evaluation:

- `LIBERO-90` is the only shared development pool.
- The archived RACaP Phase 2 is not evolved again.
- RATS receives 15 library-update rounds and at most 596 simulator resets
  (including runtime self-checks and retries).
- All artifacts are frozen before any LIBERO-PRO episode is opened.
- LIBERO-PRO results never select, repair, or update a policy or memory.

The experiment is a system-level comparison, not a claim that all methods
start from identical source code.  Therefore the report includes both final
systems and within-family deltas: RATS-base to RATS-90, and RACaP-Phase 1 to
RACaP-Phase 2.

## Fairness invariants

1. One method runs at a time, with ten workers and ten concurrent model
   requests. Reported CaP-X results include 141 episodes measured with two
   workers; latency summaries must identify concurrency for each cohort.
2. The requested runtime model is GPT-5.5 through the same VAPI relay.  Every
   response records the actual returned model when the provider exposes it.
3. Exact response caching is disabled for measured evaluations.
4. Every method receives the natural-language instruction and public visual /
   proprioceptive feedback, never evaluator predicates or object poses.
5. Native simulator predicates are the sole success authority and are read
   only after the controller stops.
6. A quota or unavailable-route error pauses the run and preserves completed
   episodes.  It is not counted as a policy failure.
7. Raw method-specific turns are reported, but comparisons use normalized
   model calls, physical/API calls, simulator steps, tokens, and wall time.
8. Videos, human-readable traces, request telemetry, manifests, and stop
   reasons are retained for every episode.
9. Every generated-policy evaluation episode runs in a fresh process.  Thus a
   success cache or failure observed on one initial state cannot adapt the
   method for the next initial state.
10. Because the GPT-5.5 chat route accepts images but not mp4 blocks, RATS
    verifier videos are converted to uniformly sampled first/middle/last PNG
    keyframes. The complete videos are still retained as raw artifacts.
11. Currency cost is reconstructed from per-request token usage with the
    dated `pricing_snapshot.yaml`. It is an OpenAI standard-list-price
    estimate; VAPI relay markup is unavailable and is never presented as an
    actual account charge.
12. Native predicates are a private evaluator channel. In strict benchmark
    mode they determine only the binary success label and are retained under
    private audit evidence; exact predicate strings, satisfied/unsatisfied
    lists, predicate-matched plan steps, and predicate-aware LLM analysis are
    excluded from policy repair, failure memory, planning, and memory curation.
    Runtime recovery receives public visual diagnostics plus the binary fact
    that independent verification did not accept completion.
13. Public seed 0 maps to the 1-based CaP-X/RATS simulator seed 1 and LIBERO
    init-state index 0; RACaP applies the same public-to-internal conversion.
    Reset telemetry records the simulator-applied init-state index rather than
    recomputing it from a raw trial seed. Separately registered public episode
    keys prevent the internal 1-based convention from changing paired labels.
14. An episode is marked complete only after an evidence gate finds non-empty
    native and resource telemetry, hosted-model telemetry, a machine summary,
    a human-readable trace, and an mp4 recording. Generated-method episodes
    index the exact files in their `EVIDENCE.json`; each RACaP group additionally
    proves that every registered `suite/task/seed` key has its own summary row,
    LLM/native telemetry, trace, trajectory, and non-empty video. A zero CLI
    return code without this evidence is an infrastructure failure, not a
    policy score. The gate also requires each applied init-state index to equal
    its registered public seed, so identically named but physically unpaired
    episodes cannot enter analysis. RACaP checks this invariant immediately
    after reset and before its first hosted-model call; a mismatch stops the
    method with an infrastructure checkpoint instead of consuming the rest of
    the registered grid. If the outer registered watchdog preempts RATS before
    its normal video epilogue, the episode still keeps its native terminal
    label. The harness encodes the already-retained, model-facing RGB frames in
    chronological order as `registered_timeout_public_trace.mp4` and records
    every source-frame hash in `registered_timeout_visual_trace.json`. No
    simulator-private observation is used, and a timeout with no retained
    public visual frames continues to fail the evidence gate.
15. Every RATS development launch or resume appends a credential-free source
    provenance record before the child process starts.  The record binds the
    loaded Git commit, dirty-diff hash, critical module hashes, protocol,
    command, and output root.  This prevents later metadata-only worktree edits
    from being mistaken for code that affected an already-running process.
16. Each finalized development round transactionally snapshots `skills.json`
    and failure memory. On resume, any partial live state is archived with
    hashes and the latest finalized snapshot is restored. Simulator resets,
    hosted calls, time, and videos consumed by interrupted rounds remain in the
    observed budget even though uncommitted strategy state is rolled back.
17. Every frozen evaluation key denotes one continuous simulator episode: one
    registered initial state, an 8000-step horizon, and no post-action reset.
    Development-only RATS attempts and runtime self-check rollouts are disabled
    at evaluation. CaP-X and RATS receive ten continuous visual code revisions
    in the registered grids; the custom generous-budget task uses 25. The
    evidence gate records and requires exactly one registered reset per sample.
18. After each generated-code method completes its 350-episode main grid,
    `run_all.py` generates a method-specific workbook and runs
    `audit_method_results.py`. RACaP-Phase 1 contributes an audited mixed-source
    350-row table whose LIBERO-90 slice is explicitly archived; RACaP-Phase 2
    contributes only its complete current 180-row PRO table. Missing evidence,
    duplicate keys, a changed response-model route, or source-generation
    telemetry inconsistent with the declared method stops the schedule. The
    audit does not enforce an expected ranking; an implausible result triggers
    implementation review before additional spending but is never modified,
    censored, or replaced after target-suite inspection.
19. Once RATS development reaches its registered budget, the supervisor writes
    a per-file SHA-256 seal manifest and removes write permission from the
    frozen skill-library and failure-memory tree. Every RATS-90 LIBERO and
    custom-long run records the skill and memory hashes before and after the
    phase and fails if they differ. Each episode works from a private copy, so
    evaluation-time failures cannot update the shared RATS-90 artifact.
20. Generated-method evaluation owns dedicated SAM3, Contact-GraspNet, and
    PyRoKi services on ports 8214--8216. The parent method process starts,
    readiness-checks, records, and stops that bundle; episode workers receive
    explicit client URLs. This prevents an occupied upstream default port from
    silently routing a measured episode through another user's process or a
    different model/server revision. Large SAM3 and Contact-GraspNet weights
    live under ignored `.cache/models/`; preflight and every service lifecycle
    record their paths, byte sizes, and SHA-256 digests.

Instruction authority is explicit. Ordinary and swap suites use the public
benchmark task-language metadata. LIBERO-PRO `*_task` perturbations instead use
their public BDDL `(:language ...)` field: the perturbator changes that field
and the native goal but preserves the original filename, while upstream
`Task.language` is reconstructed from the stale filename. The resolved source
is stored per episode as `instruction_source`, injected identically into every
method, and checked in telemetry. Native goal predicates remain private and
are never used to construct the instruction.

`protocol.yaml` freezes the exact cohorts and budgets.  `build_task_manifest.py`
materializes the benchmark task list without executing a policy.  Large raw
artifacts live under ignored `outputs/controlled_comparison/`.

## Registered execution

Load the private provider configuration without printing it, then run the
preflight and the registered RATS development phase:

```bash
set -a
source configs/env.sh
set +a
# Optional when the current Python does not contain the vendored RATS extras:
# export RACAP_EXPERIMENT_PYTHON=/path/to/rats-environment/bin/python
python experiments/controlled_comparison/audit_environment.py \
  --output outputs/controlled_comparison/provenance/preflight.json
RACAP_LLM_CACHE=0 python experiments/controlled_comparison/run_rats_development.py
```

The development command is transactionally resumable. It freezes both `skills.json` and the
LIBERO-90 failure memory under
`outputs/controlled_comparison/artifacts/rats90_frozen/`. A provider quota or
route failure exits with code 75 and writes a checkpoint instead of a failure
score. Interrupted partial-round state is archived under
`development/rats90_selfplay/interrupted_launches/`, then restored from the
latest `snapshots/iterNNN/`; its already-consumed calls, resets, time, and
recordings remain visible to the development audit.
The same transaction boundary applies when the 596-reset cap fires inside a
round: the calls and rollouts remain charged, while only the last completed
round snapshot can enter the frozen artifact. The development wrapper and the
sealing supervisor enforce this independently.
Before evaluation, the supervisor adds the append-only launch provenance to
the artifact, writes `seal_manifest.json`, and marks all artifact files and
directories read-only. Evaluation runners independently compare content
hashes before and after use; the frozen input is never used as an output path.

Development uses RATS's catalog-curiosity proposer over the 90 exact task IDs.
The proposer sees each task's public natural-language instruction, so its
curriculum decision is meaningful, but it never sees BDDL predicates or
simulator object state. The open-ended “play” proposer is intentionally not
used here because it invents new task specifications rather than selecting an
episode from the agreed LIBERO-90 development pool.

The reproduced RATS library has two audited write paths. Functions extracted
from a successful task program require native task success. Separately, the
method's failure-driven `SkillProposer` may add a helper after a failed round;
such a helper is retained only as `tier=experimental` with
`source_task=proposed_from_failures` and is not counted as a solved-task skill.
The development audit checks each round's snapshot to distinguish these paths
and rejects unlabeled failure writes. This preserves RATS's original learning
mechanism while preventing an experimental proposal from being reported as a
native-validated skill.

RATS retains its deterministic per-round trial convention during development:
round 1 uses LIBERO init-state index 0, round 2 uses index 1, and so on modulo
the available states. The archived RACaP lineage retains its historical init
states. The development pool and 596-reset ceiling are shared, but the
registered update counts are deliberately unequal (15 RATS rounds versus 32
archived RACaP proposals), and the per-rollout initial states are not
identical. Frozen evaluation, where public seeds are explicitly paired, is the
direct comparison.

At any point (and once more after completion), generate the append-only
development audit and curves with:

```bash
PYTHONPATH=. python experiments/controlled_comparison/analyze_development.py
```

This records each RATS play-task outcome, learned-skill count, resets, calls,
tokens, list-price estimate, and wall time. RACaP's 23 runtime-tested
candidates and 9 promotions are reconstructed from its archived lineage. The
two training-time success plots are intentionally separate because RATS
selects play tasks while RACaP evaluates changing curriculum cohorts; only the
later frozen evaluation is a direct method comparison.
Public-RGB human spot checks are retained in the append-only
`human_visual_spot_checks.jsonl` ledger and exported to the
`RATS human visual audits` worksheet.  Every row records its evidence path and
whether it affected control; registered spot checks are audit-only and must
have `control_effect=false`.
The per-launch source ledger is retained as
`development/rats90_selfplay/launch_source_provenance.jsonl`; it never stores
provider credentials.

After that artifact is frozen, run one method at a time. The examples below
show the in-domain and zero-shot cohorts. Omitting `--cohorts` safely runs only
the four main-report cohorts (LIBERO-90, LIBERO-PRO zero-shot, the base
diagnostic, and official LIBERO-Long); one-shot and the custom seven-object
anytime task require their dedicated drivers.

```bash
python experiments/controlled_comparison/run_libero.py \
  --method capx
python experiments/controlled_comparison/run_libero.py \
  --method rats_base
python experiments/controlled_comparison/run_libero.py \
  --method rats_90 \
  --rats-library outputs/controlled_comparison/artifacts/rats90_frozen/skills.json \
  --rats-memory outputs/controlled_comparison/artifacts/rats90_frozen/failure_memory
python experiments/controlled_comparison/run_libero.py \
  --method racap_phase1
python experiments/controlled_comparison/run_libero.py \
  --method racap_phase2
```

Each episode has its own completion marker, so relaunching the same registered
command skips completed episodes. The normalized CSV files, confidence
intervals, paired tests, figures, and Excel workbook are generated with:

```bash
python experiments/controlled_comparison/analyze_results.py
```

Once the registered RATS development run has produced the frozen artifact, the
entire registered comparison can be resumed safely with one command:

```bash
RACAP_EXPERIMENT_PYTHON=/path/to/python-with-libero-and-rats \
python experiments/controlled_comparison/run_all.py
```

`run_all.py` executes one method at a time. Main LIBERO evaluation retains the
registered ten-worker cap per method (or the number of registered episodes when
smaller); no methods overlap. It then runs
one-trial transfer, the custom seven-object anytime benchmark, and the
Robosuite diagnostic in order. Each child driver keeps episode-level completion
markers, and the orchestration state and separate phase logs are written under
`outputs/controlled_comparison/`, so an interrupted invocation can be launched
again without discarding completed episodes. The driver refuses to start until
the frozen RATS skills artifact exists and stops immediately on a provider or
quota error rather than silently changing models.
Per-method audits are stored under
`outputs/controlled_comparison/analysis/method_audits/<method>/`, including
`controlled_comparison.xlsx`, `completeness.csv`, and `method_audit.json`.

After every registered phase has completed, `run_all.py` invokes
`assemble_report.py`. The submission tables deliberately keep two coverage
contracts separate: a 1,400-row four-method complete grid (350 episodes per
method) and a 900-row five-method matched LIBERO-PRO grid (180 episodes per
method). RACaP-Phase 2 has no fabricated or imputed base/official-long rows,
and the primary RACaP-Phase 1 LIBERO-90 slice is the current instrumented
42/90 replay. The retained 48/90 Phase 1 result and archived 49/90 Phase 2
result remain separately labelled historical evidence; neither is
substituted into the controlled replay row.  The workbook includes dedicated
`RACaP History`, `Champion L90 Episodes`, and `RACaP Evidence Tiers` sheets so
their provenance and evidence strength cannot be conflated. The report also
fails closed unless all 300
paired one-trial rows, all 75 custom long-horizon checkpoints, the original 70
generated-executor Robosuite diagnostics, all 175 five-method transfer rows,
all 300 compute-matched Robosuite development episodes, and the clean development
audits are present. The final
submission-facing workbook is written to
`outputs/controlled_comparison/analysis/final/final_report.xlsx`; its manifest
records the workbook hash and the hashes of every source table.

## One-trial adaptation

The transfer experiment deliberately distinguishes zero-shot evaluation from
one-trial adaptation. For each method and each of the spatial, goal, and object
families, one calibration rollout supplies public video, public trace, and the
final native success bit to a common frozen visual critic. The critic writes a
bounded advisory experience card. It never receives predicate text, object
poses, or simulator state. Cards are frozen before testing all 60 LIBERO-PRO
tasks at seed 1, which pairs exactly with the seed-1 slice of the zero-shot
cohort. No updates occur between test episodes.
The reported adaptation budget includes both the three calibration rollouts
and the visual-critic calls that construct the cards. Each card is checkpointed
immediately; exit code 75 preserves completed cards on a quota interruption,
and a resumed run generates only the missing method/family cards.
Analysis verifies delivery at two levels. Each method manifest must bind the
registered card path and SHA-256. For the four request-instrumented methods,
every scored episode must also contain the complete family-specific card in at
least one decoded retained model request. The 60 RACaP-Phase 1 test episodes
predate request-body retention; they are explicitly audited at the weaker
launch-manifest/runtime-loader tier and are never labeled as request-level
proof. Raw JSON substring matching is not used because escaped newlines can
create a false negative. The final report requires 240 exact-request
attestations, 60 separately labeled legacy launch-chain attestations, and 300
total delivery-supported episodes.
The seed-1 comparison fixes task and simulator state but does not provide
bitwise model determinism. With one execution per key and condition, observed
changes include residual rollout/model stochasticity as well as the card
intervention; the paired statistics are therefore not a repeated-trial causal
estimate.

After the matched RATS artifact and zero-shot runs are complete:

```bash
python experiments/controlled_comparison/run_one_shot.py --phase all
python experiments/controlled_comparison/analyze_one_shot.py
```

## Long-horizon and Robosuite diagnostics

The custom long-horizon task reuses a fixed official LIBERO-10 scene and init
state but asks the agent to place seven tabletop objects in the basket. Native
containment is logged after every public API boundary without being shown to
the controller. We report completion at 5 and 10 minutes and at the final
agent-stop / 8000-step horizon.

```bash
python experiments/controlled_comparison/run_long_horizon.py --method capx
python experiments/controlled_comparison/run_long_horizon.py --method rats_base
python experiments/controlled_comparison/run_long_horizon.py --method rats_90 \
  --rats-library outputs/controlled_comparison/artifacts/rats90_frozen/skills.json \
  --rats-memory outputs/controlled_comparison/artifacts/rats90_frozen/failure_memory
python experiments/controlled_comparison/run_long_horizon.py --method racap_phase1
python experiments/controlled_comparison/run_long_horizon.py --method racap_phase2
python experiments/controlled_comparison/analyze_long_horizon.py
```

Robosuite has two deliberately separate RACaP experiments. First, the frozen
LIBERO-90 champion is connected through a non-privileged geometry-neutral
backend and evaluated zero-shot. The adapter translates RGB-D, Cartesian, IK,
joint, and gripper contracts, but contains no task-conditioned trajectory,
success feedback, object-pose truth, or Policy-API/memory update. This row
measures what transfers without Robosuite development.

Second, RACaP-RS and RATS-RS receive the same registered Robosuite curriculum:
cube lifting, cube stacking, nut assembly, spill wiping, and two-arm lifting at
seeds 100--102. The comparison is compute-matched rather than
trajectory-matched. A registered timing audit measured about 6 minutes per
RACaP 15-episode sweep and 37.6 minutes per RATS sweep. RACaP therefore receives
one baseline plus 15 candidate sweeps (240 trajectories), while RATS receives
one baseline plus three candidate sweeps (60 trajectories), targeting the same
effective development wall-time envelope. Both use three isolated development
workers and run sequentially. The workbook discloses actual wall time,
trajectories, simulator steps, hosted calls, tokens, and estimated API cost, so
the approximation is visible rather than assumed. Coding and reflection calls
remain algorithm-native. A candidate replaces the capability champion only
after a strict positive native-success delta on its complete 15-episode paired
sweep. No code-size, expected-ranking, or task-specific promotion gate is used.

After development, both artifacts are frozen and evaluated on the same seven
task templates at seeds 0--4 (35 episodes per method). Cube restacking and
two-arm handover are held-out task types; the other five tasks test unseen
seeds. The final evaluator uses five workers for both methods. All Robosuite
analyses fail closed on a missing episode, non-single reset, model-identity
drift, quota marker, service reset before readiness, source/config mismatch,
or perception-model asset mismatch. They additionally require one immutable
executor-source hash shared by CaP-X, RATS-90, and RATS-RS, plus a task-level
`EVIDENCE.json` containing paired model/native/reset telemetry, human-readable
traces, and videos. Episode telemetry reports public Policy
API boundaries separately from physical joint/gripper boundaries.

Transient `requests.Timeout` and `requests.ConnectionError` failures on the
registered VAPI endpoint use one method-neutral transport rule: retry the
exact request at most three times inside the same simulator episode, without
changing the prompt, policy, seed, or environment state. Every failed and
recovered attempt is appended to `transport_attempts.jsonl`; exhausting all
attempts makes the task block infrastructure-invalid rather than a strategy
failure. Local perception/IK endpoints are deliberately outside this hook.
Runs with incompatible transport contracts are excluded from scored tables.
Re-evaluation uses the complete frozen artifact and cohort rather than replacing
only failed rows.

```bash
python experiments/controlled_comparison/run_robosuite.py --method capx
python experiments/controlled_comparison/run_robosuite.py --method rats_90 \
  --rats-library outputs/controlled_comparison/artifacts/rats90_frozen/skills.json
python experiments/controlled_comparison/run_robosuite_racap.py \
  --solution-root policies/phase2

# Run these sequentially; both scripts checkpoint quota interruptions.
python experiments/controlled_comparison/run_robosuite_evolution.py --workers 3
python experiments/controlled_comparison/run_robosuite_rats_evolution.py --workers 3
python experiments/controlled_comparison/analyze_robosuite_development.py

# Evaluate the two frozen domain-evolved artifacts on the sealed 35-key grid,
# then build the five-method Robosuite tables, workbook, and figure.
python experiments/controlled_comparison/run_robosuite_racap.py \
  --solution-root outputs/evolution/robosuite_racap_rs_15/frozen_champion \
  --method-label racap_rs_evolved \
  --protocol-note "frozen after compute-matched Robosuite development"
python experiments/controlled_comparison/run_robosuite.py \
  --method rats_rs_evolved \
  --method-label rats_rs_evolved \
  --rats-library outputs/evolution/robosuite_rats_rs_time3/frozen_champion/skills.json
python experiments/controlled_comparison/analyze_robosuite_transfer.py
```

Every RACaP manifest records the `live_eef_link_robot0_base` contract and
backend source digest. Analyses reject results with a different Cartesian
contract or missing source evidence.
The matched RACaP-RS development run is resumable from
`outputs/evolution/robosuite_racap_rs_15`; RATS-RS starts only after RACaP-RS
finishes so provider routing and worker concurrency cannot differ because of
cross-method contention.

After credentials are available, the registered sequence can be resumed with
one fail-closed command:

```bash
python experiments/controlled_comparison/run_robosuite_campaign.py
```

The campaign never overlaps methods. It checkpoints the active phase, returns
exit code 75 on quota/provider exhaustion, requires explicit completion
evidence before advancing, and finishes by rebuilding the audited final Excel
workbook. Its state and per-phase logs are stored under
`outputs/controlled_comparison/robosuite_campaign/`.
