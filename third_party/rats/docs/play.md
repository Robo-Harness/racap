# Play

RATS acquires reusable skills during free-form play, then reuses them as planner
context at evaluation time. See [setup.md](setup.md) for environment setup.

## LIBERO-PRO play

Run the RATS lifelong loop in explore mode: the curious-child proposer discovers
its own tasks and extracts reusable skills. Explore runs on standard
`libero_spatial` — the learned skills are then evaluated on the PRO suites.

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CAPX_ENV_STACK=libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python scripts/run_rats.py \
  --config env_configs/libero/rats_libero_play.yaml \
  --explore --iterations 50 \
  --output-dir outputs/play_libero
```

`--play-mode` (curious-child proposer) is on by default, and LIBERO explore runs
automatically score candidate tasks with the novelty×frontier formula (no flag
needed). Skill snapshots are written to
`outputs/<run>/snapshots/iterNNN/skills.json` for use as eval seeds.

## MolmoSpaces play

MolmoSpaces play is launched from the root RATS environment. The launcher
starts or reuses the MolmoSpaces bridge on port `9270` and Contact-GraspNet on
port `8115`, then runs the RATS lifelong loop. By default it uses
`rats-cache/molmospaces` for `MLSPACES_CACHE_DIR` and
`rats/third_party/molmospaces/assets` for `MLSPACES_ASSETS_DIR`, matching the
setup instructions. If you bootstrapped assets into a different cache, export
that same `MLSPACES_CACHE_DIR` before launching play.

```bash
scripts/run_play_molmospaces.sh outputs/play_molmospaces
```

The focused config uses `FrankaMolmoSpacesApiReducedSkillLibrary`, VLM playtime
verification, Contact-GraspNet, and the play-disjoint benchmark split.
