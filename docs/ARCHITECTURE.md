# Architecture

## Persistent policy

RACaP represents the deployed policy as
`Pi = (A_theta, pi_phi, M)`:

- `A_theta`: typed Policy APIs in `racap/policy_api/` plus the thin frozen
  wrapper selected by a paper phase;
- `pi_phi`: Full ReAct in `racap/agent/full_react.py`, Transport ReAct in
  `racap/agent/react.py`, and their typed tools in `racap/agent/tools.py`;
- `M`: visual and strategy memory stored with each frozen phase.

The simulator backend provides public RGB-D observations, camera geometry,
robot state, and execution reports. The native evaluator is isolated behind the
backend and is never part of a controller observation.

## Runtime control flow

1. Full ReAct receives the complete instruction, fresh observations, public
   robot state, and retrieved advisory memory.
2. It decomposes the instruction, orders dependent subgoals, and routes each
   subgoal to a state-changing API or to Transport ReAct.
3. Transport ReAct handles one source-to-destination goal. It chooses a typed
   call, sees the returned report and fresh image, and either finishes, checks,
   retries with different arguments, pushes, inserts, or requests another view.
4. The selected Policy API performs grounding, geometric planning, contact or
   transport, release, and its go-home hook.
5. Execution returns measurements and a new image. ReAct, not a hidden
   predicate, decides the next action. Native predicates are read only after the
   episode for evaluation and offline evolution selection.

Full ReAct and Transport ReAct are two scopes of the same runtime policy, not
two unrelated agents. The smaller transport loop keeps local retries from
forcing the task-level planner to reconstruct the complete causal plan.

## Policy API boundary

Code owns mechanisms that should transfer across scenes: RGB-D projection,
mask geometry, grasp generation, feasible placement regions, transit clearance,
contact paths, attachment checks, and recovery reports. ReAct owns choices that
depend on the current task and observation: which tool to call, which visible
candidate is intended, grasp family, target region, yaw, margin, amount, whether
to retry, and when to stop.

A useful API must therefore satisfy two competing needs. It must be substantial
enough to provide stable physical competence, but steerable enough that ReAct
can change strategy after new evidence. The typed arguments and structured
`SkillResult` reports are the interface between these responsibilities.

## Phase snapshots

`policies/phase1/solution/controller.py` is the curriculum-trained entry point.
`policies/phase2/solution/controller.py` is the self-evolved entry point. Both
export `run_episode(runtime, episode)` and bind the same six public API names.
This stable contract lets the evaluator compare phases without changing the
runtime substrate.

## Backend isolation

- `racap/backends/libero.py` adapts the shared RATs/LIBERO stack.
- `racap/backends/robosuite.py` exposes the same public policy boundary for
  cross-embodiment experiments.
- `racap/backends/vlm.py` and `llm.py` isolate model routing and telemetry.
- `racap/backends/graspgen.py` isolates optional grasp generation.

The in-tree `third_party/rats/` snapshot supplies the common simulator adapter,
reduced Franka primitive layer, and perception service launchers used by the
controlled baselines. RACaP-specific decisions remain outside that subtree.

