# Source release and validation

## Supported installation

Use a Git checkout with editable installation. Evaluation, evolution, and baseline drivers resolve policies, experience memory, experiment definitions, and vendored integrations relative to that checkout. The Python wheel is not a standalone simulator bundle. A source distribution includes the checkout resources; unpack it before an editable installation.

## Offline checks

```bash
python -m pip install -e '.[dev]' build
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" python -m pytest
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/policies/phase1" python -m pytest policies/phase1/tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/policies/phase2" python -m pytest policies/phase2/tests
python scripts/check_release.py --tracked
```

Tests exercise mocked runtime and public contracts. They do not establish native task success rates, pretrained-model quality, or compatibility with arbitrary simulator versions. Without external LIBERO assets, the corresponding integration test is skipped. Use the pinned bootstrap and `scripts/doctor.sh` before end-to-end evaluation.

Third-party source formatting is preserved. Lint checks for fatal Python errors
apply to project-authored code; documentation and packaging changes do not
reformat the vendored compatibility snapshot.

## Audit scope

The tracked-file scan checks the working-tree content of files selected for Git publication, including newly staged files. It does not inspect ignored secrets, model caches, or private generated outputs. For a clean source export without Git metadata, run `python scripts/check_release.py --root /path/to/export` instead. Audit a clean commit export as the final publication check.

The checker rejects known generated directories, external asset manifests' targets, credential-shaped strings, private machine paths, symlinks, and unreviewed binary images. Approved documentation image bytes are checksum-pinned. Review source and image content manually as well; pattern checks cannot recognize every possible secret.

## Release contents

The release contains algorithm source, both frozen policy snapshots and their method memory, evolution prompts, comparison and training utilities, public task definitions, aggregate paper metrics, selected figures, tests, and licensed compatibility source. These are not raw training trajectories. Initial-state binaries, learned model weights, generated skill libraries, recordings, credentials, local environments, and caches remain external.

`MANIFEST.in` explicitly describes source-distribution resources and exclusions. Build artifacts stay under ignored `dist/`; build and test in a clean export when preparing a release.
