# Evolution protocol

## Why two phases

When evolution starts from only low-level primitives, most trajectories fail
before providing a useful diagnosis. A single failure can mix grounding,
grasping, transit, placement, routing, and stopping errors. The coding agent then
receives a weak learning signal and can enter a poor local minimum. Phase 1 uses
a fixed capability curriculum to create executable and diagnosable behavior.
Phase 2 then gives the harness autonomy over which remaining failure family to
improve.

## Phase 1 curriculum

The stage prompts in `evolution/prompts/stages/` define evidence distributions,
not hard-coded solutions.

| Stage | Capability focus |
|---|---|
| `s0_direct_transport` | direct source-to-destination execution |
| `s1_transport_react` | visual retry and changed-action recovery |
| `s2_bounded_placement` | margins, target geometry, and bounded correction |
| `s3a_insert` | footprint-aware constrained placement |
| `s3b_stack` | stable support geometry |
| `s3c_articulation` | drawers and hinged panels |
| `s3d_control` | buttons, switches, and rotary controls |
| `s4_tool_routing` | API selection and argument ownership |
| `s5_causal_composition` | dependent long-horizon subgoals |
| `s6_efficiency` | remove redundant model and physical calls |

## Phase 2 loop

The executable loop is implemented by `evolution/harness/orchestrator.py`:

1. The scheduler selects a capability cohort from unresolved failure evidence.
2. The current retained parent is evaluated on that cohort.
3. The critic reads tool traces and sampled before/after frames, then describes
   the earliest likely physical divergence.
4. The coding agent receives source, stage guidance, critic evidence, prior
   lineage, and memory. It proposes an edit under `solution/`, `memory/`, or
   candidate tests.
5. A fresh Git worktree isolates the candidate. Syntax, imports, unit tests, and
   a smoke rollout catch disconnected or mechanically invalid changes.
6. Parent and candidate run on exactly the same tasks, seeds, and budgets.
7. The candidate replaces the parent only when native successes strictly
   increase. Ties and regressions are archived. The next iteration always starts
   from the latest promoted source.

There is no hard-coded accuracy threshold and no source-line limit. The gate is
an observed positive paired native-success delta after mechanical validation.

## Evidence and non-privilege

The critic and coding agent may use public instructions, RGB-D frames, tool
reports, trajectories, aggregate native outcomes, and parent-candidate paired
differences. Generated controller code cannot call native predicates or access
simulator object identity. Candidate guards enforce the public runtime surface.

The source package contains the Phase 1 and Phase 2 policies. Generated
rollouts and candidate repositories are local experiment outputs.
