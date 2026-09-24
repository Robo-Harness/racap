# Skill Library Compilation

Analyze evaluation outputs and compile reusable skill libraries from successful trials.

## Prerequisites

- OpenRouter proxy running: `uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110`
- Evaluation outputs in `outputs/` (from `capx/envs/launch.py`)

## Usage

```bash
# 1. Parse outputs → generates analysis.txt, highlights.txt, functions.txt per experiment
uv run --no-sync --active scripts/skill_library_compilation/parse_outputs.py \
    --cfg.output-dir outputs/

# 2. Summarize across models and tasks
uv run --no-sync --active scripts/skill_library_compilation/summarize_analysis.py \
    --cfg.output-dir outputs/

# 3. Compile skill library from successful reduced-API experiments
uv run --no-sync --active scripts/skill_library_compilation/compile_skill_library.py \
    --cfg.output-dir outputs/
```

Output: `outputs/skill_library.txt` — curated reusable functions for future evaluations.

## Utilities

- **`trial_folder_rename.py`** — Standardize trial folder numbering (`--cfg.dry-run` to preview)
- **`eval_dir_to_code.py`** — Consolidate all `code.py` files into one file for review

Run any script with `--help` for full options.
