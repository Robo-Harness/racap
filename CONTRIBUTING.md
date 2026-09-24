# Contributing

Use a source checkout and a dedicated Python environment. Install with `python -m pip install -e '.[dev]'` and run the checks in [release validation](docs/RELEASE.md).

Keep changes to the runtime, frozen paper policies, and evaluation protocol separate. Do not silently alter a frozen snapshot and describe its results as the paper's unchanged policy. If a change affects evaluation, state its observation inputs, tasks, initial states, budgets, native-success definition, and whether it updates code or memory on the target benchmark.

Add unit or contract tests for behavioral changes. Tests should use synthetic inputs or mocked services and must not require API credentials, start expensive model calls, or download simulator assets during collection.

Do not commit generated trajectories, datasets, model weights, local paths, credentials, caches, or service logs. Documentation images require an explicit review and checksum entry in `scripts/check_release.py`. Preserve third-party copyright and license notices. Describe modifications to vendored code in `THIRD_PARTY_NOTICES.md`.
