# Reproducibility guide

## What is and is not bundled

The source release contains all RACaP-authored runtime and experiment source, the two
frozen paper policies, public task manifests, prompts, tests, and pinned upstream
source metadata. It intentionally excludes credentials, raw rollouts, caches,
model weights, and large simulator checkouts.

`scripts/bootstrap.sh` downloads these pinned repositories:

| Dependency | Commit |
|---|---|
| LIBERO-PRO | `47aaa8038930bcdc84ab9ea2867e2ffc8039ab4a` |
| Robosuite fork | `a498b087d4bc5a3981e3d27030d09bc537a537f3` |
| Contact-GraspNet PyTorch | `8cd98632047e418dc938fc80add258a1ca15f9a9` |
| SAM3 fork | `6fe87d64a5beb9084923d7a9e002741178635b09` |

The vendored RATs source is snapshot
`1df65a180562e91911214fa1caba7c7ed9407b3d`.

## Environment

- Linux and Python 3.10 to 3.12
- CUDA GPU recommended
- MuJoCo headless rendering through EGL on GPU or OSMesa on CPU
- local services at the configurable SAM3, GraspNet, and PyRoKi endpoints
- an OpenAI-compatible VAPI route or OpenRouter credentials for hosted agents

Create the environment with `bash scripts/bootstrap.sh`, copy
`configs/local.env.example` to `configs/local.env`, configure one model route,
then run:

```bash
source configs/env.sh
bash scripts/start_services.sh
bash scripts/doctor.sh
```

The doctor checks Python imports, LIBERO data paths, strict evaluator flags,
service endpoints, and model credentials.

Bootstrap installs missing `uv` tooling in the project's `.runtime/` directory,
not into the global Python environment. Preserve the checkout for editable
installation; standalone wheels do not include the external simulator stack.

## Frozen-policy evaluation

```bash
bash scripts/evaluate.sh phase1 phase1_full90 --seeds 0
bash scripts/evaluate.sh phase2 phase2_full90 --seeds 0
```

To evaluate a subset, append `--task-ids` and use the same evaluator arguments.
For example:

```bash
bash scripts/evaluate.sh phase2 smoke --task-ids 0 1 2 --seeds 0 --workers 1
```

The standard configuration uses 45 Full ReAct turns, at most eight pick-place
calls, three pushes, four inserts, five state-changing calls, two stacks, and an
8000 simulator-step horizon. Each worker runs episodes in isolated processes.

Outputs follow this schema:

```text
outputs/full_agent_eval/<tag>/
├── summary.json
├── records.jsonl
├── index.html
└── episodes/
    └── t071_seed0/
        ├── episode.mp4
        ├── trace.md
        └── trajectory.json
```

## Controlled comparison

`experiments/controlled_comparison/protocol.yaml` records the task grid,
budgets, method order, and metric definitions. The drivers run one method at a
time with equal worker counts. `task_manifest.json` fixes public instructions,
suite IDs, task IDs, and seeds. Analysis scripts never infer a missing run as a
failure or silently combine protocols.

The released RATS comparison is controlled rather than an official-setting
reproduction. See `experiments/PROVENANCE.md` before comparing these rows with
numbers from another paper.

## Source-only verification

```bash
python -m pip install -e ".[dev]"
python -m compileall -q racap evolution policies experiments scripts tests
PYTHONPATH="$PWD" python -m pytest -q
```

The `dev` extra installs the analysis and geometry libraries used by the test
suite. Tests do not require simulator downloads or model credentials. A successful `scripts/doctor.sh` is the stronger readiness check for simulator
evaluation.

The LIBERO benchmark integration test is skipped when the external simulator
is unavailable. Run it after bootstrap to validate the installed benchmark.

## External assets

The source archive excludes initial-state binaries and generated third-party
skill libraries and benchmark records. Their expected locations and integrity
hashes are listed in `configs/external_assets.json`.

Obtain these assets from an authorized source and place them in a local directory
using the relative paths from that manifest. Then run:

```bash
python scripts/prepare_assets.py --source /path/to/authorized-assets
```

The command verifies each asset before copying it to its ignored runtime
location. It refuses changed files and never overwrites differing existing assets.
Run only the experiments whose external assets you have provided. The custom
long-horizon evaluator also accepts `RACAP_LONG_HORIZON_INIT` for an external
initial-state file. Missing assets are an explicit preflight error.

Credentials belong in `configs/local.env`; set `RACAP_VAPI_KEY` and
`RACAP_VAPI_BASE` for an HTTP(S) compatible endpoint, or configure OpenRouter.
Never put credentials in URLs, source files, or public logs. See
[security](../SECURITY.md) and [source release validation](RELEASE.md).
