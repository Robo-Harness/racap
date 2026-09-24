"""LIBERO-specific utilities for the RATS lifelong loop.

Provides scene context extraction, task discovery, and environment type
detection for running RATS on LIBERO benchmarks (Franka Panda + robosuite/MuJoCo).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("rats.libero_utils")


def _parse_bddl_goal_predicates(bddl_path: str) -> list[str]:
    """Parse the ``(:goal ...)`` block of a LIBERO BDDL file into predicate strings.

    LIBERO goal bodies are always flat s-expressions of the form::

        (:goal (And (On akita_black_bowl_1 plate_1) (Turnon flat_stove_1)))

    Boolean combinators (``And``/``Or``/``Not``) are stripped; each atomic
    predicate becomes ``"Name(arg1, arg2)"``. Empty list on any parse failure
    so the caller degrades to the NL goal alone — mechanism A just stays
    dormant on that task instead of crashing.
    """
    try:
        text = Path(bddl_path).read_text()
    except OSError:
        return []

    # Balance parens starting from "(:goal" to isolate the goal block.
    idx = text.find("(:goal")
    if idx < 0:
        return []
    depth = 0
    start = None
    end = None
    for i in range(idx, len(text)):
        c = text[i]
        if c == "(":
            if start is None:
                start = i
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if start is None or end is None:
        return []
    block = text[start:end]

    # Atomic predicate ≈ a paren group with no nested parens inside it.
    # (LIBERO goals are flat — predicates don't nest.)
    tok_re = re.compile(r"\(([A-Za-z_][A-Za-z0-9_]*)([^()]*)\)")
    preds: list[str] = []
    for m in tok_re.finditer(block):
        name = m.group(1)
        if name in ("goal", "And", "Or", "Not"):
            continue
        args = m.group(2).split()
        preds.append(f"{name}({', '.join(args)})" if args else name)
    return preds

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Environment type detection
# ---------------------------------------------------------------------------

def detect_env_type(env: Any) -> str:
    """Detect whether *env* wraps a BEHAVIOR-1K, LIBERO, or MolmoSpaces env."""
    low_level = getattr(env, "low_level_env", env)
    cls_name = type(low_level).__name__
    if "MolmoSpaces" in cls_name or "molmospaces" in cls_name or hasattr(low_level, "list_task_descriptors"):
        return "molmospaces"
    if "Libero" in cls_name or "libero" in cls_name:
        return "libero"
    if hasattr(low_level, "handle"):
        handle = low_level.handle
        if hasattr(handle, "suite_name") and hasattr(handle, "task_language"):
            return "libero"
    return "behavior"


def detect_env_type_from_config(config_path: str) -> str:
    """Detect environment type from YAML config path string."""
    p = config_path.lower()
    if "molmospaces" in p or "mlsp2" in p:
        return "molmospaces"
    if "libero" in p:
        return "libero"
    if "r1pro" in p or "b1k" in p or "behavior" in p:
        return "behavior"
    return "behavior"


# ---------------------------------------------------------------------------
# Scene context extraction for LIBERO
# ---------------------------------------------------------------------------

def extract_libero_scene_context(env: Any) -> dict[str, Any]:
    """Build a RATS scene-context dict from a LIBERO CodeExecutionEnvBase.

    Works with ``FrankaLiberoCodeEnv`` wrapping ``FrankaLiberoEnv``.
    """
    low_level = getattr(env, "low_level_env", env)
    handle = getattr(low_level, "handle", None)

    # --- Task language / goal ---
    task_language = ""
    if handle is not None:
        task_language = getattr(handle, "task_language", "") or ""
    if not task_language:
        task_language = (
            getattr(env, "_task_prompt", "")
            or getattr(env, "prompt", "")
            or ""
        )

    # --- Suite / task identifiers ---
    suite_name = getattr(handle, "suite_name", "libero") if handle else "libero"
    task_id = getattr(handle, "task_id", 0) if handle else 0
    activity_name = f"{suite_name}_task{task_id}"

    # Intentionally empty. Previously populated from `low_level._current_obs`
    # keys (e.g. "porcelain_mug_1_pos" → "porcelain_mug") and MuJoCo
    # `sim.model.body_id2name` — both of those are internal simulator state
    # that a non-privileged vision-only baseline doesn't have. Letting it
    # flow into policy_writer / planner prompts was a leak of sim-internal
    # object identifiers. Keep the key in scene_context for schema
    # compatibility; downstream consumers (planner, policy_writer, failure
    # memory keyword lookup) all degrade gracefully to NL-goal-derived
    # objects when this is empty.
    object_scope: dict[str, str] = {}

    # --- API docs / available functions ---
    apis = getattr(env, "_apis", {})
    api_docs_parts: list[str] = []
    available_functions: list[str] = []
    for api in apis.values():
        if hasattr(api, "combined_doc") and callable(api.combined_doc):
            api_docs_parts.append(api.combined_doc())
        if hasattr(api, "functions") and callable(api.functions):
            available_functions.extend(list(api.functions().keys()))

    # Intentionally no BDDL goal-predicate plumbing here. Parsing the BDDL
    # `:goal` block would leak the verifier's symbolic checklist (e.g.
    # `On(akita_black_bowl_1, flat_stove_1_cook_region)`) into the
    # diagnoser — info a non-privileged baseline agent doesn't have.
    # The diagnoser uses plan steps (agent-generated) as its checkpoint
    # unit instead. `_parse_bddl_goal_predicates` is left in this module
    # only as a debugging utility; do not wire it into scene_context.

    return {
        "env_type": "libero",
        "scene_model": suite_name,
        "activity_name": activity_name,
        "object_scope": object_scope,
        "goal_conditions_nl": task_language,
        "task_prompt": task_language,
        "available_functions": sorted(set(available_functions)),
        "api_docs": "\n\n".join(api_docs_parts),
        # LIBERO-specific extras
        "suite_name": suite_name,
        "task_id": task_id,
        "task_language": task_language,
    }


# ---------------------------------------------------------------------------
# Task catalog discovery for LIBERO
# ---------------------------------------------------------------------------

# Mapping of suite_name → number of tasks
LIBERO_SUITE_SIZES: dict[str, int] = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
    # LIBERO-PRO evaluation suites (CaP-Agent0 baseline comparison)
    "libero_object_swap": 10,
    "libero_object_task": 10,
    "libero_goal_swap": 10,
    "libero_goal_task": 10,
    "libero_spatial_swap": 10,
    "libero_spatial_task": 10,
}


def discover_libero_tasks(
    suite_names: list[str] | None = None,
    *,
    from_yaml: bool = True,
) -> list[dict[str, Any]]:
    """Discover available LIBERO tasks.

    Parameters
    ----------
    suite_names:
        Which suites to include.  ``None`` → all standard suites.
    from_yaml:
        If True, also scan ``env_configs/libero/`` for YAML configs.

    Returns
    -------
    List of task dicts with keys: activity_name, scene_model, task_id,
    suite_name, env_config_path (if from YAML).
    """
    tasks: list[dict[str, Any]] = []

    # 1. Scan env_configs/libero/ for YAML files
    if from_yaml:
        config_dir = _PROJECT_ROOT / "env_configs" / "libero"
        if config_dir.exists():
            try:
                import yaml
            except ImportError:
                yaml = None  # type: ignore[assignment]
            if yaml is not None:
                for path in sorted(config_dir.glob("*.yaml")):
                    try:
                        with path.open() as f:
                            cfg = yaml.safe_load(f)
                        low_level = (
                            cfg.get("env", {}).get("cfg", {}).get("low_level", {})
                        )
                        sn = low_level.get("suite_name", "libero_10")
                        tid = low_level.get("task_id", 0)
                        tasks.append({
                            "activity_name": f"{sn}_task{tid}",
                            "scene_model": sn,
                            "suite_name": sn,
                            "task_id": tid,
                            "env_config_path": str(path),
                        })
                    except Exception:
                        continue

    # 2. Enumerate known suites
    suites = suite_names or list(LIBERO_SUITE_SIZES.keys())
    seen = {t["activity_name"] for t in tasks}
    for sn in suites:
        n_tasks = LIBERO_SUITE_SIZES.get(sn, 10)
        for tid in range(n_tasks):
            name = f"{sn}_task{tid}"
            if name not in seen:
                tasks.append({
                    "activity_name": name,
                    "scene_model": sn,
                    "suite_name": sn,
                    "task_id": tid,
                    "env_config_path": "",
                })
                seen.add(name)

    return tasks


def recreate_libero_env(
    old_env: Any,
    suite_name: str,
    task_id: int,
) -> Any:
    """Recreate a LIBERO CodeExecutionEnvBase for a different task.

    LIBERO environments can't be rebound at runtime — each task has its own
    BDDL file and MuJoCo scene.  This function tears down the old env and
    builds a fresh one with the same API configuration.

    Returns the new high-level env (``CodeExecutionEnvBase`` subclass).
    """
    import sys

    # Extract config from the old env so we mirror its API/privilege setup
    cfg_obj = getattr(old_env, "cfg", None)
    api_names = list(getattr(old_env, "_apis", {}).keys())
    if not api_names:
        api_names = ["FrankaLiberoApi"]
    privileged = getattr(cfg_obj, "privileged", False) if cfg_obj else False
    prompt = getattr(old_env, "_task_prompt", None)

    # Tear down old MuJoCo sim to free memory
    old_ll = getattr(old_env, "low_level_env", None)
    if old_ll is not None:
        old_viser = getattr(old_ll, "viser_server", None)
        if old_viser is not None and hasattr(old_viser, "stop"):
            try:
                old_viser.stop()
            except Exception:
                pass
        old_handle = getattr(old_ll, "handle", None)
        if old_handle is not None and hasattr(old_handle, "env"):
            try:
                old_handle.env.close()
            except Exception:
                pass

    from rats.envs.configs.instantiate import instantiate

    low_level_cfg = {
        "_target_": "rats.envs.simulators.libero.FrankaLiberoEnv",
        "suite_name": suite_name,
        "task_id": task_id,
    }
    # Copy extra low-level kwargs from old env (max_steps, control_freq, …)
    old_ll_cfg = getattr(cfg_obj, "low_level", None) if cfg_obj else None
    if old_ll_cfg is not None:
        for attr in ("max_steps", "control_freq", "seed", "enable_render", "viser_debug"):
            val = getattr(old_ll_cfg, attr, None)
            if val is not None:
                low_level_cfg[attr] = val

    env_cfg = {
        "_target_": "rats.envs.tasks.franka.franka_libero_env.FrankaLiberoCodeEnv",
        "cfg": {
            "_target_": "rats.envs.tasks.base.CodeExecEnvConfig",
            "low_level": low_level_cfg,
            "privileged": privileged,
            "apis": api_names,
        },
    }
    if prompt is not None:
        env_cfg["cfg"]["prompt"] = prompt

    original_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        new_env = instantiate(env_cfg)
    finally:
        sys.argv = original_argv

    new_env.reset()
    logger.info(f"Recreated LIBERO env: {suite_name} task {task_id}")
    return new_env


def parse_libero_activity_name(activity_name: str) -> tuple[str, int]:
    """Parse ``"libero_spatial_task3"`` → ``("libero_spatial", 3)``."""
    # Format: {suite_name}_task{id}
    parts = activity_name.rsplit("_task", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return activity_name, 0


def get_libero_task_language(suite_name: str, task_id: int) -> str:
    """Retrieve the natural-language description for a LIBERO task.

    Requires the LIBERO package to be importable.  Returns an empty string
    on failure.
    """
    try:
        import os, sys  # noqa: E401
        vendor_root = str(
            _PROJECT_ROOT / "rats" / "third_party" / "LIBERO-PRO"
        )
        if os.path.isdir(vendor_root) and vendor_root not in sys.path:
            sys.path.append(vendor_root)
        from libero import benchmark  # type: ignore[import-not-found]

        bd = benchmark.get_benchmark_dict(help=False)
        suite = bd[suite_name]()
        task = suite.get_task(task_id)
        return task.language
    except Exception as exc:
        logger.debug("Could not load LIBERO task language: %s", exc)
        return ""


# The 6 LIBERO-PRO evaluation suites — 60 tasks total. ``_swap`` and ``_task``
# pairs share their ``language`` field (same task, different perturbation),
# so the prompt rendering below dedups by string and tags which suite kinds
# each entry appears in.
LIBERO_PRO_EVAL_SUITES: tuple[str, ...] = (
    "libero_object_swap", "libero_object_task",
    "libero_goal_swap",   "libero_goal_task",
    "libero_spatial_swap", "libero_spatial_task",
)

_LIBERO_PRO_EVAL_TASK_CACHE: list[dict[str, Any]] | None = None


def _extract_bddl_block(bddl_text: str, tag: str) -> str:
    """Pull the first ``(:tag ...)`` block out of a BDDL file as a single line.

    Walks brace depth manually so nested parens inside ``:goal`` (e.g.
    ``(And (In a b))``) are captured intact. Returns the body between the
    outer parens (without the ``:tag`` prefix), whitespace-compressed.
    """
    import re

    marker = "(" + tag
    start = bddl_text.find(marker)
    if start < 0:
        return ""
    # First open paren is at `start`; scan forward until matched close.
    depth = 0
    end = -1
    for i in range(start, len(bddl_text)):
        ch = bddl_text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return ""
    # Drop the leading "(:tag" so the body starts after the tag name.
    body = bddl_text[start + len(marker):end]
    body = re.sub(r"\s+", " ", body).strip()
    return body


def _summarize_scene_from_bddl(bddl_path: "Path") -> dict[str, Any]:
    """Extract a compact scene summary from a LIBERO-PRO BDDL file.

    Returns ``{"objects": list[str], "fixtures": list[str], "goal": str}``
    where ``objects`` is a list of ``"name - type"`` declarations, fixtures
    the same, and ``goal`` is the literal goal predicate string. Empty
    fields on parse failure — the caller should handle.
    """
    try:
        txt = bddl_path.read_text()
    except Exception:
        return {"objects": [], "fixtures": [], "goal": ""}
    objs_body = _extract_bddl_block(txt, ":objects")
    fxts_body = _extract_bddl_block(txt, ":fixtures")
    goal_body = _extract_bddl_block(txt, ":goal")
    # Split declarations: BDDL ``:objects`` / ``:fixtures`` after whitespace
    # compression is a flat token stream like
    #     name1 name2 - type1 name3 - type2 name4 - type3
    # where each ``- type`` applies to every name accumulated since the
    # previous ``- type``. Walk tokens, accumulate names, flush at every
    # dash separator.
    def _split_decls(body: str) -> list[str]:
        if not body:
            return []
        out: list[str] = []
        tokens = body.split()
        current_names: list[str] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "-" and current_names and (i + 1) < len(tokens):
                type_name = tokens[i + 1]
                for n in current_names:
                    out.append(f"{n} - {type_name}")
                current_names = []
                i += 2
                continue
            # Skip stray dashes (no accumulated names): malformed input.
            if tok == "-":
                i += 1
                continue
            current_names.append(tok)
            i += 1
        return out

    goal_str = ""
    if goal_body:
        # ``goal_body`` is what came between the outer ``(:goal ... )`` parens.
        # It typically looks like "(And (P1 a b) (P2 c))" or just "(P a b)".
        # Strip the outer ``(And ...)`` wrapper when present so the predicate
        # list reads cleanly; otherwise hand back the body verbatim.
        goal_inner = goal_body.strip()
        import re as _re
        m = _re.match(r"^\(\s*and\s+(.*)\)\s*$", goal_inner, _re.IGNORECASE)
        if m:
            goal_str = m.group(1).strip()
        else:
            goal_str = goal_inner

    return {
        "objects": _split_decls(objs_body),
        "fixtures": _split_decls(fxts_body),
        "goal": goal_str,
    }


_BDDL_PATH_CACHE: dict[str, "Path"] | None = None


def _find_bddl_file(bddl_filename: str) -> "Path | None":
    """Locate a LIBERO-PRO BDDL file by basename. Lazy-builds a name→path map."""
    global _BDDL_PATH_CACHE
    if _BDDL_PATH_CACHE is None:
        _BDDL_PATH_CACHE = {}
        root = _PROJECT_ROOT / "rats" / "third_party" / "LIBERO-PRO"
        if root.is_dir():
            for p in root.rglob("*.bddl"):
                # Last write wins — we don't dedupe across suites because
                # each filename is task-unique within a suite folder. The
                # caller only needs ANY valid BDDL with that name to
                # extract the static scene summary.
                _BDDL_PATH_CACHE[p.name] = p
    return _BDDL_PATH_CACHE.get(bddl_filename)


def get_libero_pro_eval_tasks() -> list[dict[str, Any]]:
    """Return all 60 LIBERO-PRO eval-suite tasks with language + scene info.

    Each entry: ``{"suite_name", "task_id", "language", "objects", "fixtures",
    "goal"}``. Cached after the first call (LIBERO benchmark import is heavy
    and the BDDL filesystem scan is non-trivial). Returns an empty list when
    the LIBERO package isn't importable — caller should handle.
    """
    global _LIBERO_PRO_EVAL_TASK_CACHE
    if _LIBERO_PRO_EVAL_TASK_CACHE is not None:
        return _LIBERO_PRO_EVAL_TASK_CACHE
    tasks: list[dict[str, Any]] = []
    # Inline imports keep the LIBERO benchmark dependency optional for
    # consumers that never call this function (e.g. MolmoSpaces-only runs).
    try:
        import os, sys  # noqa: E401
        vendor_root = str(
            _PROJECT_ROOT / "rats" / "third_party" / "LIBERO-PRO"
        )
        if os.path.isdir(vendor_root) and vendor_root not in sys.path:
            sys.path.append(vendor_root)
        from libero import benchmark  # type: ignore[import-not-found]

        bd = benchmark.get_benchmark_dict(help=False)
    except Exception as exc:
        logger.debug("LIBERO-PRO eval task load failed: %s", exc)
        _LIBERO_PRO_EVAL_TASK_CACHE = []
        return []
    for sn in LIBERO_PRO_EVAL_SUITES:
        n = LIBERO_SUITE_SIZES.get(sn, 10)
        try:
            suite = bd[sn]()
        except Exception:
            continue
        for tid in range(n):
            try:
                task = suite.get_task(tid)
            except Exception:
                continue
            lang = getattr(task, "language", "") or ""
            if not lang:
                continue
            bddl_filename = getattr(task, "bddl_file", "") or ""
            scene: dict[str, Any] = {"objects": [], "fixtures": [], "goal": ""}
            if bddl_filename:
                p = _find_bddl_file(bddl_filename)
                if p is not None:
                    scene = _summarize_scene_from_bddl(p)
            tasks.append({
                "suite_name": sn,
                "task_id": tid,
                "language": lang,
                "bddl_file": bddl_filename,
                "objects": scene["objects"],
                "fixtures": scene["fixtures"],
                "goal": scene["goal"],
            })
    _LIBERO_PRO_EVAL_TASK_CACHE = tasks
    return tasks


def format_libero_pro_eval_tasks_block() -> str:
    """Render the eval-task list as a prompt-ready block.

    Each unique task gets a multi-line entry:
      * NL task description
      * Scene objects (BDDL ``:objects`` declarations)
      * Scene fixtures (BDDL ``:fixtures`` declarations, excluding floor/table)
      * Formal goal predicate (BDDL ``:goal``)

    Dedups by language (the ``_swap`` and ``_task`` variants of a suite share
    their language strings; only the perturbation type differs — and they
    use the same BDDL file, so the scene summary is identical too). Groups
    by the suite-family prefix (``libero_object`` / ``libero_goal`` /
    ``libero_spatial``).

    Returns an empty string when no eval tasks are available.
    """
    tasks = get_libero_pro_eval_tasks()
    if not tasks:
        return ""
    families: dict[str, list[dict[str, Any]]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for t in tasks:
        sn = str(t.get("suite_name") or "")
        lang = str(t.get("language") or "").strip()
        if not lang:
            continue
        for suffix in ("_swap", "_task"):
            if sn.endswith(suffix):
                family = sn[: -len(suffix)]
                break
        else:
            family = sn
        key = (family, lang)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        families.setdefault(family, []).append(t)
    if not families:
        return ""

    # Filter out the workspace-surface fixtures that appear in every task —
    # mentioning them once in the intro reduces token waste vs repeating
    # "main_table - table" on every entry.
    GENERIC_FIXTURES = {"floor", "main_table"}

    lines: list[str] = [
        "EVALUATION TASKS — your play should be targeted on the kinds of "
        "downstream tasks the agent will be evaluated on. The LIBERO-PRO eval "
        "benchmark uses 30 unique task descriptions (each appears in two "
        "perturbation suites, _swap and _task, for 60 total) across three "
        "families. Use these as a STRONG HINT for what objects, fixtures, "
        "and goal predicates to exercise during play; you do NOT have to "
        "propose these tasks verbatim, but proposals that build skills these "
        "tasks would need are most useful. Per-task scene context (objects, "
        "fixtures, and the formal LIBERO goal predicate) is included so you "
        "can see which objects co-occur and what predicate shape the eval "
        "verifier checks. The base workspace (main_table, floor) is omitted "
        "from per-task fixture lists.",
        "",
    ]
    for family in sorted(families):
        entries = sorted(families[family], key=lambda t: t["language"])
        lines.append(f"# {family} family ({len(entries)} unique tasks):")
        for i, t in enumerate(entries, start=1):
            lang = t["language"]
            lines.append(f"  {i}. {lang}")
            objs = [s for s in (t.get("objects") or []) if s]
            fxts = [
                s for s in (t.get("fixtures") or [])
                if s and s.split(" - ")[0].strip() not in GENERIC_FIXTURES
            ]
            goal = (t.get("goal") or "").strip()
            if objs:
                lines.append(f"     objects:   {'; '.join(objs)}")
            if fxts:
                lines.append(f"     fixtures:  {'; '.join(fxts)}")
            if goal:
                lines.append(f"     goal:      {goal}")
        lines.append("")
    return "\n".join(lines).rstrip()
