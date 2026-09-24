<h1 align="center">RACaP</h1>

<p align="center">
  <strong>Agentic Reasoning, Acting, and Coding as Policies for Evolvable Robot Learning</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#results">Results</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="docs/EVOLUTION.md">Evolution</a> ·
  <a href="README_zh-CN.md">中文</a> ·
  <a href="LICENSE">MIT License</a>
</p>

---

**RACaP** separates learning reusable robot code from making runtime decisions. Before deployment, a coding agent evolves typed **Policy APIs**, a **ReAct harness**, and **experience memory**. At execution time, a visual agent selects APIs and revises their arguments using fresh observations—without generating or repairing source code on the robot's execution path.

## Overview

<p align="center">
  <img src="assets/framework.png" width="1100" alt="RACaP framework: capability curriculum and paired self-evolution before deployment; visual ReAct control over frozen Policy APIs during execution">
</p>

**Phase 1 — Capability curriculum learning.** Build execution, recovery, physical API coverage, and task-level orchestration in stages. The frozen policy is in [policies/phase1](policies/phase1/).

**Phase 2 — Autonomous self-evolution.** A visual critic analyzes failures, a coding agent proposes reusable changes, and parent/candidate policies are evaluated on the same tasks and budgets. Only strictly improved candidates are promoted. The frozen policy is in [policies/phase2](policies/phase2/).

At runtime, **Full ReAct** handles subgoals and causal ordering; **Transport ReAct** handles local observation, execution, and recovery. Six steerable APIs—`pickplace`, `insert`, `stack`, `push`, `articulate`, and `actuate_control`—provide reusable physical mechanisms. Native success predicates remain evaluator-only. [Architecture →](docs/ARCHITECTURE.md)

## Results

Figures from the paper. Accuracy uses native simulator success, not the agent's declaration of completion. CaP-X and RATS rows are controlled comparisons under this study's protocol, not reproductions of their original paper settings.

### In-domain performance and long-horizon execution

<p align="center">
  <img src="assets/main-results.png" width="1100" alt="RACaP results on LIBERO-90, runtime computation, and LIBERO-Long; Phase 2 reaches 54.4% in-domain and 46.0% long-horizon success">
</p>

RACaP Phase 2 reaches **54.4%** on LIBERO-90 and **46.0%** on LIBERO-Long. The Phase 1 bar is the fully instrumented replay; the paper separately records rollout variation. [Protocols and provenance →](experiments/PROVENANCE.md)

### LIBERO-PRO transfer

<p align="center">
  <img src="assets/libero-pro.png" width="1100" alt="LIBERO-PRO frozen-artifact zero-shot accuracy and a separate calibration-trial adaptation experiment; RACaP Phase 2 zero-shot accuracy is 45.0%">
</p>

Frozen Phase 2 achieves **45.0%** zero-shot success. The right-hand panel is a separate calibration-based adaptation setting and is not part of the zero-shot result. [Detailed results →](experiments/paper_results.csv)

## Quick start

### 1. Get the source

```bash
git clone https://github.com/Robo-Harness/racap.git
cd racap
```

Use **Linux and Python 3.10–3.12**. Full simulation requires external benchmark assets, perception/motion services, and suitable rendering hardware. The repository contains source, frozen policy code and memory, prompts, task definitions, aggregate metrics, and paper figures. It does **not** bundle training datasets, raw trajectories, recordings, model weights, credentials, or downloaded simulator checkouts.

### 2. Install and test without simulation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" python -m pytest -q -p no:cacheprovider
```

Tests use mocked runtimes; they do not start hosted-model calls or download simulators. The external LIBERO integration test skips when that dependency is absent. Keep the source checkout: evaluation and evolution use its relative policy, prompt, memory, and third-party paths.

### 3. Set up the simulator stack

In a separate shell, bootstrap the pinned stack and configure a model route:

```bash
bash scripts/bootstrap.sh
cp configs/local.env.example configs/local.env
chmod 600 configs/local.env
# Edit configs/local.env locally; never commit credentials.
source configs/env.sh
bash scripts/start_services.sh
bash scripts/doctor.sh
```

Bootstrap downloads pinned external repositories and installs the simulation environment under `third_party/rats/.venv`. It does not provide restricted checkpoints or generated baseline skill libraries. Follow the [reproducibility guide](docs/REPRODUCIBILITY.md), including [external asset provisioning](docs/REPRODUCIBILITY.md#external-assets), before running experiments. The default service launcher binds only to loopback; custom remote service endpoints should be provisioned separately.

### 4. Evaluate a frozen policy

```bash
# Phase 2 smoke evaluation, with video and readable trace
bash scripts/evaluate.sh phase2 smoke --task-ids 71 --seeds 0 --workers 1

# Full in-domain evaluations
bash scripts/evaluate.sh phase1 phase1_full90 --seeds 0
bash scripts/evaluate.sh phase2 phase2_full90 --seeds 0
```

These commands execute simulations and may incur model API costs. Results are written to ignored `outputs/full_agent_eval/<tag>/` directories. For other suites and controlled baselines, see [reproduction instructions](docs/REPRODUCIBILITY.md) and [comparison drivers](experiments/controlled_comparison/README.md).

### 5. Run evolution

```bash
# Inspect the curriculum without model calls
racap-evolve plan

# Launch an isolated evolution experiment from Phase 1
racap-evolve run --experiment example_run \
  --stage auto --runtime-candidates 2 --workers 2
```

Evolution runs model-generated code and consumes simulation/API resources. Use a disposable environment with minimum necessary permissions. Candidate Git worktrees provide source isolation, **not a security sandbox**. See [evolution](docs/EVOLUTION.md) and [security](SECURITY.md).

## Code and documentation

```text
racap/          ReAct agents, typed Policy APIs, simulator/model backends
policies/       Frozen Phase 1 and Phase 2 controllers and experience memory
evolution/      Curriculum, critic, coder, paired evaluation, and selection
experiments/    Controlled comparisons, analysis, and ReAct training utilities
scripts/        Setup, services, evaluation, asset provisioning, release checks
tests/          Unit and contract tests
third_party/    Licensed RATs/CaP-X compatibility sources
```

- [Architecture](docs/ARCHITECTURE.md) and [evolution](docs/EVOLUTION.md)
- [Reproducibility and external assets](docs/REPRODUCIBILITY.md)
- [Evaluation protocols and claim boundaries](experiments/PROVENANCE.md)
- [Contributing](CONTRIBUTING.md), [release validation](docs/RELEASE.md), and [security](SECURITY.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [中文说明](README_zh-CN.md)

The method is **RACaP**; the Python package and repository are `racap`; the evolution command is `racap-evolve`.

## License

RACaP-authored code is released under [MIT](LICENSE). Vendored RATs, CaP-X, and PyRoKi components retain their original licenses and notices. External simulators, datasets, model weights, and services remain subject to their own terms.
