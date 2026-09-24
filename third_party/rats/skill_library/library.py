"""JSON-based skill library for RATS.

Stores reusable skills as parameterized functions. Provides context to
Task Proposer (lightweight summaries) and Planner (full code + docs).
Only the Planner retrieves from the library for task execution.

Reliability tracking
--------------------
Each learned skill carries usage_count/success_count, a Wilson-lower-bound
score, and a tier ∈ {verified, experimental, deprecated}. Tier is derived
from empirical usage, not declared at creation; record_usage(name, success)
drives the transitions. The Planner receives skills pre-sorted by
(tier, Wilson score) so reliable skills bubble up without any prompt-level
"must-use" rule.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

from skill_library.initial_primitives import build_initial_skills

logger = logging.getLogger("rats.skill_library")

# Tier transition thresholds. Kept conservative so a single-success skill
# doesn't immediately get promoted to "verified" (Wilson bound handles that
# automatically for few-sample cases).
TIER_PROMOTE_MIN_USES = 3
TIER_PROMOTE_MIN_SR = 0.5
TIER_DEPRECATE_MIN_USES = 10
TIER_DEPRECATE_MAX_SR = 0.2
TIER_DEMOTE_MIN_USES = 20
TIER_DEMOTE_MAX_SR = 0.3


def wilson_lower_bound(success: int, total: int, z: float = 1.96) -> float:
    """Wilson score lower bound at the given z (default 95% CI).

    Conservative estimate of true success rate: 1/1=0.21, 10/10=0.72,
    100/100=0.96. Skills with few observations are ranked low even if
    their raw success_rate is 1.0.
    """
    if total <= 0:
        return 0.0
    p = success / total
    denom = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return max(0.0, (center - margin) / denom)


# ---------------------------------------------------------------------------
# Cross-run helpers: tier-gated load + summed-counts merge
# ---------------------------------------------------------------------------

# Higher = stricter. `deprecated` is intentionally below the experimental
# floor so any positive `min_tier` filters it out.
_TIER_RANK = {"deprecated": -1, "experimental": 0, "verified": 1}


def filter_skills_by_tier(
    skills: list[dict[str, Any]],
    min_tier: str = "experimental",
    *,
    keep_primitives: bool = True,
) -> list[dict[str, Any]]:
    """Drop learned skills below ``min_tier``.

    Primitives always pass when ``keep_primitives`` is true (the default)
    — they're foundational and don't have an empirical tier the same way
    learned skills do. Unknown tier strings sort as ``experimental``.
    """
    threshold = _TIER_RANK.get(str(min_tier or "experimental").lower(), 0)
    out: list[dict[str, Any]] = []
    for s in skills:
        if not isinstance(s, dict):
            continue
        if s.get("is_primitive"):
            if keep_primitives:
                out.append(s)
            continue
        tier = str(s.get("tier", "experimental")).lower()
        if _TIER_RANK.get(tier, 0) >= threshold:
            out.append(s)
    return out


def merge_skill_library_files(
    seed_path: str | Path,
    extra_paths: list[str | Path] | None = None,
    *,
    min_tier: str | None = None,
) -> list[dict[str, Any]]:
    """Load a seed skill library and merge in extras with summed counts.

    Behavior:
      * Skills with the same ``name`` across files have their
        ``usage_count``, ``success_count``, and ``rediscovery_count``
        summed; ``success_rate`` is recomputed from the summed totals.
      * The seed's code body is preferred for collisions — if you opted
        a particular seed file, its implementation is what you trust;
        extras only contribute reliability statistics.
      * Skills only present in extras are appended verbatim.
      * After the merge, ``min_tier`` (if given) prunes the result —
        applied last so an extra that pushes a skill across the
        verification threshold via summed counts is reflected.

    Returns a fresh list of skill dicts; the input files are not
    modified. Caller is expected to hand the result to
    ``SkillLibrary._save`` / write JSON.
    """
    seed_p = Path(seed_path)
    seed_skills: list[dict[str, Any]] = []
    if seed_p.exists():
        try:
            with seed_p.open() as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                seed_skills = [s for s in loaded if isinstance(s, dict)]
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not parse seed skill library %s (%s); starting empty",
                seed_p, exc,
            )

    by_name: dict[str, dict[str, Any]] = {}
    for s in seed_skills:
        name = s.get("name")
        if not name:
            continue
        by_name[name] = dict(s)

    for extra in extra_paths or []:
        ep = Path(extra)
        if not ep.exists():
            logger.warning("Skill-library merge: extra path missing: %s", ep)
            continue
        try:
            with ep.open() as f:
                extras = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not parse extra skill library %s (%s); skipping",
                ep, exc,
            )
            continue
        if not isinstance(extras, list):
            continue
        for s in extras:
            if not isinstance(s, dict):
                continue
            name = s.get("name")
            if not name:
                continue
            if name not in by_name:
                # Brand-new skill — adopt whole record.
                by_name[name] = dict(s)
                continue
            merged = by_name[name]
            for fld in ("usage_count", "success_count", "rediscovery_count"):
                merged[fld] = int(merged.get(fld, 0) or 0) + int(s.get(fld, 0) or 0)
            n_total = int(merged.get("usage_count", 0))
            sc_total = int(merged.get("success_count", 0))
            merged["success_rate"] = sc_total / n_total if n_total > 0 else 0.0
            # Bump tier upward if the merged counts now warrant it. We
            # don't demote here; that would be too eager when extras
            # haven't been validated. Real demotions happen at runtime
            # via SkillLibrary._update_tier.
            current_tier = str(merged.get("tier", "experimental")).lower()
            if (
                current_tier == "experimental"
                and n_total >= TIER_PROMOTE_MIN_USES
                and merged["success_rate"] >= TIER_PROMOTE_MIN_SR
            ):
                merged["tier"] = "verified"

    skills = list(by_name.values())
    if min_tier:
        skills = filter_skills_by_tier(skills, min_tier)
    return skills


class SkillLibrary:
    def __init__(self, storage_path: str = "skill_library/skills.json") -> None:
        self._storage_path = Path(storage_path)
        self._skills: list[dict[str, Any]] = []
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Load from disk or initialize with CaP-Gym primitives."""
        if self._storage_path.exists():
            with self._storage_path.open() as f:
                self._skills = json.load(f)
            self._migrate_reliability_fields()
            self._backfill_dependent_skills()
        else:
            self._skills = build_initial_skills()
            self._migrate_reliability_fields()
            self._save()

    def _backfill_dependent_skills(self) -> None:
        """Fill ``dependent_skills`` for legacy entries via AST walk.

        Older skills were extracted before the auto-dep recording in
        add_skill existed; their ``dependent_skills`` is ``[]`` even
        when the body clearly calls a sibling. The injection-time
        dependency closure walks that field, so without backfill the
        deprecation guard can't see legacy wrappers as "depended on"
        and the orphaned-call NameError keeps reproducing in resumed
        runs. Idempotent: only fills entries that are still empty AND
        whose AST yields a non-empty learned-skill call set.
        """
        mutated = False
        for s in self._skills:
            if s.get("is_primitive"):
                continue
            if s.get("dependent_skills"):
                continue
            code = s.get("code") or ""
            if not code:
                continue
            try:
                v = self.validate_skill_code(code)
            except Exception:
                continue
            deps = sorted(
                set(v.get("learned_skill_calls") or []) - {s.get("name")}
            )
            if deps:
                s["dependent_skills"] = deps
                mutated = True
                logger.debug(
                    f"  backfilled dependent_skills for "
                    f"'{s.get('name')}' → {deps}"
                )
        if mutated:
            self._save()

    def _migrate_reliability_fields(self) -> None:
        """Backfill usage_count / success_count / tier for pre-tier skills.

        Old skills only carried success_rate (starting at 1.0, bumped via
        EMA on *rediscovery* — which is not the same as execution success).
        Rather than inherit those fictional rates, learned skills reset to
        0/0 at migration time so tier is driven entirely by real empirical
        data going forward. Primitives pin to tier='verified'.
        """
        mutated = False
        for s in self._skills:
            if "usage_count" in s and "success_count" in s and "tier" in s:
                continue
            mutated = True
            is_primitive = s.get("is_primitive", False)
            if is_primitive:
                s.setdefault("usage_count", 0)
                s.setdefault("success_count", 0)
                s.setdefault("tier", "verified")
                s.setdefault("success_rate", 1.0)
                continue
            # Learned skill: real empirical counters start fresh. The old
            # success_rate was rediscovery-noise, not execution-truth.
            s.setdefault("usage_count", 0)
            s.setdefault("success_count", 0)
            s.setdefault("tier", "experimental")
            s["success_rate"] = 0.0
        if mutated:
            try:
                self._save()
            except OSError:
                pass  # migration is best-effort; will retry on next save

    @staticmethod
    def _wilson(skill: dict[str, Any]) -> float:
        return wilson_lower_bound(
            int(skill.get("success_count", 0)),
            int(skill.get("usage_count", 0)),
        )

    def _update_tier(self, skill: dict[str, Any]) -> None:
        """Recompute tier from usage_count/success_count. Primitives untouched."""
        if skill.get("is_primitive", False):
            skill["tier"] = "verified"
            return
        n = int(skill.get("usage_count", 0))
        sc = int(skill.get("success_count", 0))
        sr = sc / n if n > 0 else 0.0
        current = skill.get("tier", "experimental")
        # Promotion: experimental → verified
        if (
            current == "experimental"
            and n >= TIER_PROMOTE_MIN_USES
            and sr >= TIER_PROMOTE_MIN_SR
        ):
            skill["tier"] = "verified"
            logger.info(
                f"  Skill promoted verified: {skill['name']} "
                f"(SR={sr:.2f}, n={n})"
            )
            return
        # Deprecation: experimental with enough data showing it's bad.
        # Guard: if any non-deprecated learned skill lists this one in its
        # dependent_skills, demoting to 'deprecated' would orphan the
        # wrapper at exec time (its preamble injection skips deprecated
        # entries). Hold the dep at 'experimental' instead so it stays in
        # scope. The wrapper's own SR will eventually drop as well, and
        # then both can be deprecated together.
        if (
            current == "experimental"
            and n >= TIER_DEPRECATE_MIN_USES
            and sr <= TIER_DEPRECATE_MAX_SR
        ):
            sname = skill.get("name")
            depended_on = sname and any(
                sname in (other.get("dependent_skills") or [])
                and other.get("tier") != "deprecated"
                and not other.get("is_primitive")
                for other in self._skills
            )
            if depended_on:
                logger.info(
                    f"  Skill kept experimental (dependency guard): "
                    f"{skill['name']} (SR={sr:.2f}, n={n}) — needed by an "
                    f"active wrapper"
                )
                return
            skill["tier"] = "deprecated"
            logger.info(
                f"  Skill deprecated: {skill['name']} (SR={sr:.2f}, n={n})"
            )
            return
        # Demotion: verified that turned out to regress
        if (
            current == "verified"
            and n >= TIER_DEMOTE_MIN_USES
            and sr <= TIER_DEMOTE_MAX_SR
        ):
            skill["tier"] = "experimental"
            logger.info(
                f"  Skill demoted experimental: {skill['name']} "
                f"(SR={sr:.2f}, n={n})"
            )

    def record_usage(
        self,
        skill_names: list[str],
        success: bool,
        *,
        iteration: int | None = None,
        source: str = "usage_tracker",
    ) -> list[dict[str, Any]]:
        """Increment usage/success counters for each named skill.

        Called by the lifelong loop after each code attempt completes. The
        list should be the set of learned-skill function names that appear
        in the attempt's code (regex-extracted — duplicates within the same
        attempt are collapsed to a single usage).

        Primitives are ignored — their success_rate is not tracked; they're
        always tier='verified'.
        """
        if not skill_names:
            return []
        seen: set[str] = set()
        changed = False
        lifecycle_events: list[dict[str, Any]] = []
        for name in skill_names:
            if name in seen:
                continue
            seen.add(name)
            for s in self._skills:
                if s.get("name") == name and not s.get("is_primitive", False):
                    old_tier = s.get("tier", "experimental")
                    s["usage_count"] = int(s.get("usage_count", 0)) + 1
                    if success:
                        s["success_count"] = int(s.get("success_count", 0)) + 1
                    n = s["usage_count"]
                    s["success_rate"] = s["success_count"] / n if n else 0.0
                    self._update_tier(s)
                    if old_tier != "deprecated" and s.get("tier") == "deprecated":
                        if iteration is not None:
                            s["deprecated_at_iteration"] = int(iteration)
                        s["deprecated_by"] = source
                        s["deprecated_reason"] = (
                            f"usage_count={s['usage_count']}, "
                            f"success_count={s['success_count']}, "
                            f"success_rate={s['success_rate']:.3f}"
                        )
                        lifecycle_events.append({
                            "iteration": iteration,
                            "action": "deprecated",
                            "skill": name,
                            "source": source,
                            "reason": s["deprecated_reason"],
                        })
                    changed = True
                    break
        if changed:
            self._save()
        return lifecycle_events

    def _save(self) -> None:
        """Persist skills to disk."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._storage_path.open("w") as f:
            json.dump(self._skills, f, indent=2)
        self._save_notebook()

    @staticmethod
    def _notebook_source(text: str) -> list[str]:
        """Convert cell text to notebook source lines."""
        if not text:
            return []
        return text.splitlines(keepends=True)

    def _save_notebook(self) -> None:
        """Persist a notebook mirror for easier visual inspection."""
        notebook_path = self._storage_path.with_suffix(".ipynb")
        learned_count = sum(
            1 for skill in self._skills if not skill.get("is_primitive", False)
        )
        cells: list[dict[str, Any]] = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": self._notebook_source(
                    "# RATS Skill Library\n\n"
                    f"- Source JSON: `{self._storage_path.name}`\n"
                    f"- Total skills: {len(self._skills)}\n"
                    f"- Learned skills: {learned_count}\n\n"
                    "Each skill appears as a summary cell followed by its code cell."
                ),
            }
        ]

        for skill in self._skills:
            deps = skill.get("dependent_skills") or []
            deps_text = ", ".join(deps) if deps else "None"
            usage_count = int(skill.get("usage_count", 0))
            success_count = int(skill.get("success_count", 0))
            tier = skill.get(
                "tier",
                "verified" if skill.get("is_primitive", False) else "experimental",
            )
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": self._notebook_source(
                        f"## `{skill.get('name', 'unknown_skill')}`\n\n"
                        f"- Description: {skill.get('description', '')}\n"
                        f"- Tier: {tier}\n"
                        f"- Primitive: {bool(skill.get('is_primitive', False))}\n"
                        f"- Usage: {success_count}/{usage_count} successes\n"
                        f"- Dependencies: {deps_text}\n"
                    ),
                }
            )
            code = skill.get("code", "")
            if code and not code.endswith("\n"):
                code += "\n"
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {
                        "skill_name": skill.get("name", ""),
                        "tags": ["skill"],
                    },
                    "outputs": [],
                    "source": self._notebook_source(code),
                }
            )

        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        notebook_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False))

    def get_context_for_task_proposer(
        self, include_deprecated: bool = False,
    ) -> dict[str, Any]:
        """Lightweight summary for Task Proposer: names, descriptions, counts.

        Used for novelty reasoning - "what skills are missing?". Deprecated
        skills are hidden by default — matching `get_full_skills_for_planner`
        and `get_learned_skills_for_curator` — so the proposer doesn't see
        retired skills as available inventory.
        """
        learned_skills = [s for s in self._skills if not s.get("is_primitive", False)]
        if not include_deprecated:
            learned_skills = [s for s in learned_skills if s.get("tier") != "deprecated"]
        primitive_skills = [s for s in self._skills if s.get("is_primitive", False)]
        return {
            "total_skills": len(primitive_skills) + len(learned_skills),
            "total_primitives": len(primitive_skills),
            "total_learned": len(learned_skills),
            "primitive_names": [s["name"] for s in primitive_skills],
            "learned_skills": [
                {
                    "name": s["name"],
                    "description": s["description"],
                    "success_rate": s.get("success_rate", 0.0),
                    "usage_count": int(s.get("usage_count", 0)),
                    "tier": s.get("tier", "experimental"),
                    "source_task": s.get("source_task", "unknown"),
                }
                for s in learned_skills
            ],
        }

    def get_full_skills_for_planner(
        self, include_deprecated: bool = False,
    ) -> list[dict[str, Any]]:
        """Full skill details for Planner: code + docs + preconditions + effects.

        The Planner is the SOLE retrieval owner. It reads the full library,
        selects relevant skills, and passes only the selected subset to Policy Writer.

        Returns skills pre-sorted: primitives first (stable), then learned
        skills ranked by (tier, Wilson lower bound) so 'verified' skills
        appear before 'experimental' ones, and within a tier, skills with
        more empirical evidence of success outrank untested ones.
        Deprecated skills are hidden by default.
        """
        # Tier rank — verified is worth showing first so the model picks them
        # without any prompt-level rule; experimental still included because
        # new skills live there until they prove themselves.
        tier_rank = {"verified": 0, "experimental": 1, "deprecated": 2}

        def sort_key(s: dict) -> tuple[int, int, float, str]:
            # Primitives first (group 0), then tier-ordered learned skills
            group = 0 if s.get("is_primitive", False) else 1
            t = tier_rank.get(s.get("tier", "experimental"), 2)
            # Negative Wilson so higher scores come first
            return (group, t, -self._wilson(s), s.get("name", ""))

        skills = list(self._skills)
        if not include_deprecated:
            skills = [s for s in skills if s.get("tier") != "deprecated"]
        skills.sort(key=sort_key)

        return [
            {
                "skill_id": s["skill_id"],
                "name": s["name"],
                "description": s["description"],
                "code": s["code"],
                "api_primitives_used": s.get("api_primitives_used", []),
                "preconditions": s.get("preconditions", []),
                "effects": s.get("effects", []),
                "dependent_skills": s.get("dependent_skills", []),
                "success_rate": s.get("success_rate", 0.0),
                "usage_count": int(s.get("usage_count", 0)),
                "success_count": int(s.get("success_count", 0)),
                "tier": s.get("tier", "verified" if s.get("is_primitive") else "experimental"),
                "wilson_score": self._wilson(s),
                "is_primitive": s.get("is_primitive", False),
                "example_code": s.get("example_code", ""),
            }
            for s in skills
        ]

    def get_reliability_summary(self, top_k: int = 8) -> dict[str, Any]:
        """Compact reliability snapshot for success_context enrichment.

        Returns the top-K learned skills by Wilson score plus tier counts.
        Used by lifelong_loop/policy_writer to surface "what has worked" as
        evidence (not as a rule), letting the model gravitate toward
        reliable skills without any prompt-level must-use directive.
        """
        learned = [s for s in self._skills if not s.get("is_primitive", False)]
        counts: dict[str, int] = {}
        for s in learned:
            t = s.get("tier", "experimental")
            counts[t] = counts.get(t, 0) + 1
        ranked = sorted(learned, key=lambda s: -self._wilson(s))
        top = []
        for s in ranked[:top_k]:
            n = int(s.get("usage_count", 0))
            sc = int(s.get("success_count", 0))
            top.append({
                "name": s.get("name"),
                "tier": s.get("tier", "experimental"),
                "success_count": sc,
                "usage_count": n,
                "success_rate": s.get("success_rate", 0.0),
                "wilson_score": self._wilson(s),
                "description": s.get("description", "")[:140],
            })
        return {
            "tier_counts": counts,
            "top_by_wilson": top,
            "total_learned": len(learned),
        }

    @staticmethod
    def _extract_def_name(code: str) -> str | None:
        """Extract the function name from a 'def func_name(' statement in code."""
        import re
        match = re.search(r'def\s+(\w+)\s*\(', code)
        return match.group(1) if match else None

    def _next_revived_skill_name(self, base_name: str) -> str:
        """Return a unique revived name for a new skill replacing a deprecated name."""
        existing = {str(s.get("name", "")) for s in self._skills}
        i = 1
        while True:
            candidate = f"{base_name}__revived_{i}"
            if candidate not in existing:
                return candidate
            i += 1

    @staticmethod
    def _rename_skill_function(code: str, old_name: str, new_name: str) -> str:
        """Rename the stored skill function and direct self-calls in its code."""
        import re

        if not code or not old_name or not new_name or old_name == new_name:
            return code
        escaped = re.escape(old_name)
        renamed = re.sub(
            rf"(^\s*def\s+){escaped}(\s*\()",
            rf"\1{new_name}\2",
            code,
            count=1,
            flags=re.MULTILINE,
        )
        # If the function was recursive, keep those calls pointing at the
        # renamed function instead of the deprecated same-name library entry.
        return re.sub(rf"\b{escaped}(?=\s*\()", new_name, renamed)

    # ---- builtins / library helpers that any skill body may call without
    # being declared as a learned-skill dependency. Kept narrow on purpose:
    # the broader Quality Checker handles full API call validation; this
    # set is just the names a typical extracted skill needs to be allowed
    # to call free of charge.
    _SKILL_VALIDATOR_BUILTINS = frozenset({
        "print", "len", "range", "int", "float", "str", "list", "dict",
        "tuple", "set", "bool", "abs", "min", "max", "sum", "sorted",
        "enumerate", "zip", "map", "filter", "isinstance", "type",
        "hasattr", "getattr", "setattr", "any", "all", "round", "format",
        "repr", "next", "iter", "reversed", "slice", "bytes", "frozenset",
        "object", "complex", "divmod", "pow", "hash", "callable",
        "Exception", "RuntimeError", "ValueError", "TypeError",
        "KeyError", "IndexError", "AttributeError", "StopIteration",
        "AssertionError", "OSError", "IOError", "FileNotFoundError",
        "NotImplementedError",
        # numpy / np aliases the writer + extractor use heavily
        "np", "numpy", "math", "json", "time",
    })

    def validate_skill_code(
        self,
        code: str,
        *,
        primitive_names: set[str] | None = None,
        extra_primitive_names: set[str] | None = None,
    ) -> dict[str, Any]:
        """Static AST validation for a candidate skill body.

        Walks every ``Name``-style call inside ``code`` and classifies it as
        ``primitive`` | ``learned_skill`` | ``builtin`` | ``self`` (for the
        skill calling itself recursively) | ``unknown``. Returns the
        breakdown so extractors can:
          - REJECT skills that call functions not in the library
          - auto-populate ``dependent_skills`` from the learned-skill set
          - keep recursive calls (def name calling itself) without flagging

        ``primitive_names`` lets callers override the in-library primitive
        set (tests). ``extra_primitive_names`` is the union with the live
        runtime API's ``available_functions`` — the JSON seed file often
        omits primitives that DO exist in the running API class
        (``grasp_with_wrist_closeloop`` etc.), and rejecting skills for using
        them produces false positives. Callers
        who know the runtime scope should pass it here; callers who don't
        (e.g. offline replays) get conservative behavior by default.
        """
        import ast
        result: dict[str, Any] = {
            "ok": False,
            "syntax_error": None,
            "self_name": None,
            "primitive_calls": [],
            "learned_skill_calls": [],
            "builtin_calls": [],
            "unknown_calls": [],
        }
        if not code or not code.strip():
            result["syntax_error"] = "empty code"
            return result
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result["syntax_error"] = f"{e.msg} at line {e.lineno}"
            return result

        # The skill's own def name may call itself recursively; allow it.
        local_defs: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local_defs.add(node.name)
            elif isinstance(node, ast.ClassDef):
                local_defs.add(node.name)
        # Heuristic "self_name": when the body has exactly one top-level
        # def, that's the wrapper name and the one we'll record as
        # canonical. Stored so callers can flag wrappers calling
        # themselves under a misspelt sibling name.
        if len(local_defs) == 1:
            result["self_name"] = next(iter(local_defs))

        if primitive_names is None:
            primitive_names = {
                s.get("name") for s in self._skills
                if s.get("is_primitive")
            }
        if extra_primitive_names:
            primitive_names = set(primitive_names) | set(extra_primitive_names)
        learned_names = {
            s.get("name") for s in self._skills
            if not s.get("is_primitive") and s.get("name")
        }

        primitive_calls: set[str] = set()
        learned_calls: set[str] = set()
        builtin_calls: set[str] = set()
        unknown_calls: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name in local_defs:
                    continue  # recursive / nested helper
                if name in primitive_names:
                    primitive_calls.add(name)
                elif name in learned_names:
                    learned_calls.add(name)
                elif name in self._SKILL_VALIDATOR_BUILTINS:
                    builtin_calls.add(name)
                elif name.startswith("_"):
                    # Private helpers passed through closure or top-level
                    # imports. Don't flag — extractor can't infer scope.
                    builtin_calls.add(name)
                else:
                    unknown_calls.add(name)

        result["primitive_calls"] = sorted(primitive_calls)
        result["learned_skill_calls"] = sorted(learned_calls)
        result["builtin_calls"] = sorted(builtin_calls)
        result["unknown_calls"] = sorted(unknown_calls)
        result["ok"] = not unknown_calls
        return result

    def collect_dependency_closure(self, names: list[str]) -> list[str]:
        """Walk ``dependent_skills`` transitively for the given seed names.

        Returns the dedup'd list of every learned-skill name reachable
        from the seeds (seeds included). Tolerates cycles. Skips
        primitives — they're always in scope. Used by lifelong_loop to
        force-inject deps even when they're tier='deprecated' (a
        deprecated dep otherwise orphans the wrapper that needs it).
        """
        by_name = {s.get("name"): s for s in self._skills}
        visited: set[str] = set()
        queue: list[str] = list(names)
        out: list[str] = []
        while queue:
            n = queue.pop(0)
            if not n or n in visited:
                continue
            visited.add(n)
            s = by_name.get(n)
            if not s or s.get("is_primitive"):
                continue
            out.append(n)
            for dep in s.get("dependent_skills", []) or []:
                if dep not in visited:
                    queue.append(dep)
        return out

    def add_skill(
        self,
        skill: dict[str, Any],
        *,
        available_functions: list[str] | set[str] | None = None,
    ) -> bool:
        """Add a new skill to the library (called by Feedback Generator on success).

        Performs exact name dedup + semantic dedup (LLM-based) to avoid
        storing functionally identical skills under different names.

        ``available_functions`` should be the live API's function-name set
        (``scene_context["available_functions"]``). It is unioned into the
        validator's "known primitive" pool so skills calling real-but-
        unseeded API functions (``grasp_with_wrist_closeloop`` etc.) don't
        get false-rejected. Pass ``None`` only when the
        caller can't know — e.g. an offline migration script — and accept
        that the validator will then be stricter.

        Returns True if skill was added (not a duplicate).
        """
        # --- Canonicalize name: use the actual def name from code ---
        code = skill.get("code", "")
        if code:
            def_name = self._extract_def_name(code)
            if def_name and def_name != skill.get("name"):
                logger.debug(
                    f"Skill name mismatch: metadata='{skill['name']}' vs "
                    f"code def='{def_name}', using code def name"
                )
                skill["name"] = def_name

        # A deprecated same-name skill should not absorb or block a fresh
        # implementation. Keep the old record untouched, add this candidate
        # under a unique revived name, and start empirical stats from zero.
        same_name = [s for s in self._skills if s.get("name") == skill.get("name")]
        if same_name and all(s.get("tier") == "deprecated" for s in same_name):
            old_name = str(skill["name"])
            new_name = self._next_revived_skill_name(old_name)
            logger.info(
                f"Skill '{old_name}' collides only with deprecated entries; "
                f"adding as '{new_name}'"
            )
            skill["name"] = new_name
            if code:
                skill["code"] = self._rename_skill_function(code, old_name, new_name)
                code = skill["code"]
            skill["skill_id"] = f"learned_{new_name}"
            skill["revived_from"] = old_name
            skill["revived_reason"] = "same name as deprecated skill"
            skill.pop("duplicate_of", None)
            skill.pop("curator_reason", None)
            skill["usage_count"] = 0
            skill["success_count"] = 0
            skill["success_rate"] = 0.0
            skill["tier"] = "experimental"
            skill["rediscovery_count"] = 1

        # --- Static validation: reject skills that call functions not in
        # the library, and auto-fill dependent_skills so the planner can
        # walk the graph at injection time. Without this, we had wrappers
        # that referenced sibling helpers by guessed names (off by `_in_`
        # vs `_on_`) AND wrappers whose real deps were silently
        # discarded — both produced NameError in exec scope. We do this
        # AFTER name canonicalisation but BEFORE dedup so a candidate
        # that calls hallucinated helpers never enters the library at
        # all. example_code (the verbatim sub-agent script) gets the
        # same treatment as a smoke-check: if the wrapper invents calls
        # the original script never made, the wrapper is unsafe even if
        # the example ran.
        if code:
            v = self.validate_skill_code(
                code,
                extra_primitive_names=(
                    set(available_functions) if available_functions else None
                ),
            )
            if v["syntax_error"]:
                logger.warning(
                    f"Skill '{skill.get('name')}' rejected: syntax — "
                    f"{v['syntax_error']}"
                )
                return False
            if v["unknown_calls"]:
                logger.warning(
                    f"Skill '{skill.get('name')}' rejected: calls unknown "
                    f"functions {v['unknown_calls']} (not in library + not "
                    f"a builtin/np). Likely an LLM-hallucinated helper."
                )
                return False
            # Auto-fill dependent_skills from learned-skill calls. The
            # extractor sometimes leaves it empty even when the wrapper
            # calls a sibling — record it here so deprecation can't
            # silently orphan us.
            existing_deps = set(skill.get("dependent_skills") or [])
            new_deps = set(v["learned_skill_calls"]) - {skill.get("name")}
            if new_deps - existing_deps:
                skill["dependent_skills"] = sorted(existing_deps | new_deps)
            # Smoke-check: example_code (raw sub-agent script) should
            # exercise the wrapper's deps. If the wrapper introduces a
            # learned-skill call that the original script never made, the
            # wrapper is unsafe — the LLM extruded plumbing the runtime
            # never proved out. Skip when example_code is empty (skills
            # extracted from main-loop success rather than sub-agent).
            example = (skill.get("example_code") or "").strip()
            if example and v["learned_skill_calls"]:
                ev = self.validate_skill_code(
                    example,
                    extra_primitive_names=(
                        set(available_functions) if available_functions else None
                    ),
                )
                proven = set(ev.get("learned_skill_calls", []))
                hallucinated = (
                    set(v["learned_skill_calls"]) - proven - {skill.get("name")}
                )
                if hallucinated:
                    logger.warning(
                        f"Skill '{skill.get('name')}' rejected: wrapper "
                        f"calls {sorted(hallucinated)} but the verbatim "
                        f"sub-agent script never invoked them. The wrapper "
                        f"is plumbing the model invented post-hoc."
                    )
                    return False

        # --- Exact name match ---
        existing_names = {s["name"] for s in self._skills}
        if skill["name"] in existing_names:
            # Rediscovery ≠ execution success. We used to EMA-bump
            # success_rate here, but that conflated "the skill proposer
            # extracted this pattern again" with "the skill worked when
            # executed". Real empirical updates now come from record_usage.
            # We only count the rediscovery as a soft rediscovery signal.
            for s in self._skills:
                if s["name"] == skill["name"]:
                    s["rediscovery_count"] = int(s.get("rediscovery_count", 0)) + 1
                    break
            self._save()
            return False

        # --- Semantic dedup against learned skills ---
        learned = [
            s for s in self._skills
            if not s.get("is_primitive", False) and s.get("tier") != "deprecated"
        ]
        if learned:
            duplicate_of = self._find_semantic_duplicate(skill, learned)
            if duplicate_of:
                logger.info(
                    f"Skill '{skill['name']}' is semantically duplicate of "
                    f"'{duplicate_of}', merging instead of adding"
                )
                for s in self._skills:
                    if s["name"] == duplicate_of:
                        s["rediscovery_count"] = int(s.get("rediscovery_count", 0)) + 1
                        # Keep the longer/newer code if it's more complete —
                        # but ONLY when the new code defines a function whose
                        # name matches the existing skill's name (otherwise
                        # we introduce a name↔def mismatch that causes
                        # NameError at exec-time; observed repeatedly).
                        new_code = skill.get("code", "")
                        old_code = s.get("code", "")
                        new_def = self._extract_def_name(new_code)
                        if (
                            len(new_code) > len(old_code)
                            and new_def
                            and new_def == s["name"]
                        ):
                            s["code"] = new_code
                        break
                self._save()
                return False

        # Ensure required fields. New skills enter 'experimental' with zero
        # usage; promotion to 'verified' happens in record_usage once the
        # skill has proven itself.
        skill.setdefault("skill_id", f"learned_{skill['name']}")
        skill.setdefault("is_primitive", False)
        skill.setdefault("usage_count", 0)
        skill.setdefault("success_count", 0)
        skill.setdefault("success_rate", 0.0)
        skill.setdefault("tier", "experimental")
        skill.setdefault("rediscovery_count", 1)
        skill.setdefault("api_primitives_used", [])
        skill.setdefault("preconditions", [])
        skill.setdefault("effects", [])
        skill.setdefault("dependent_skills", [])
        # Raw working invocation from the sub-agent that learned this skill
        # (when learned via sub-agent). Surfaced to downstream policy_writer
        # calls so the caller has a concrete template for arg values.
        skill.setdefault("example_code", "")
        # Provenance fields — populated by feedback_generator._extract_skills
        # (and may be set by subagent extraction paths too). Surfaced to the
        # planner so it can match a skill to a NEW task by similarity rather
        # than name alone.
        #   source_task: the natural-language task whose success produced
        #     this skill (e.g. "put the milk on the yellow plate").
        #   extraction_rationale: the LLM's 1-sentence justification for
        #     why this sub-behavior is worth promoting to a reusable skill.
        #   usage_example: ONE concrete invocation from the successful run
        #     with the verbatim argument values that actually worked.
        skill.setdefault("source_task", "")
        skill.setdefault("extraction_rationale", "")
        skill.setdefault("usage_example", "")
        # Structured parameter / return spec extracted alongside `code`. Each
        # entry in `params` carries name, Python type hint, array shape (for
        # tensor-valued args), default, and a 1-sentence description. `returns`
        # carries type, shape, and description. These are populated by the
        # LLM-driven extraction in prompts/feedback_generator.txt — older
        # records have empty defaults so the JSON schema stays stable.
        skill.setdefault("params", [])
        skill.setdefault("returns", {"type": "", "shape": "", "description": ""})

        self._skills.append(skill)
        self._save()
        return True

    @staticmethod
    def _name_similarity(a: str, b: str) -> float:
        """Quick word-overlap similarity between two skill names."""
        words_a = set(a.lower().replace("_", " ").split())
        words_b = set(b.lower().replace("_", " ").split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / max(len(words_a), len(words_b))

    def _find_semantic_duplicate(
        self, candidate: dict[str, Any], existing: list[dict[str, Any]],
    ) -> str | None:
        """Check if candidate skill is semantically equivalent to any existing skill.

        Uses a two-stage approach:
          1. Pre-filter: rank all existing skills by name/description word overlap
             and keep the top candidates (up to 50). This ensures we never miss a
             near-duplicate while staying within token budget.
          2. LLM comparison: send the filtered set for semantic judgement.

        Returns the name of the duplicate skill, or None if no duplicate found.
        """
        # Stage 1: pre-filter by name + description word overlap
        cand_words = set(
            candidate.get("name", "").lower().replace("_", " ").split()
            + candidate.get("description", "").lower().split()[:20]
        )

        def _overlap_score(skill: dict) -> float:
            skill_words = set(
                skill.get("name", "").lower().replace("_", " ").split()
                + skill.get("description", "").lower().split()[:20]
            )
            if not cand_words or not skill_words:
                return 0.0
            return len(cand_words & skill_words) / max(len(cand_words), len(skill_words))

        # Sort by overlap; keep top-50 so we never blow the context budget
        # but still check all skills with any word overlap
        ranked = sorted(existing, key=_overlap_score, reverse=True)
        # Always include at least 50, but also include anything with overlap > 0.15
        top_n = 50
        comparison_set = ranked[:top_n]
        for skill in ranked[top_n:]:
            if _overlap_score(skill) >= 0.15:
                comparison_set.append(skill)
            else:
                break  # ranked list, so remaining will have lower scores

        # Stage 2: LLM semantic comparison.
        #
        # The previous prompt sent only `name + description[:100] + primitives`
        # for each side, which is too thin to catch behavioral duplication —
        # e.g. in the LIBERO smoke run the check let through three skills
        # whose CODE bodies were trivial supersets of each other (one
        # literally inlined the other two's code) because the descriptions
        # used different verbs.
        #
        # FIX (truncation audit round 2): the earlier intermediate version
        # of this block still applied arbitrary slices (cand_code[:2000],
        # existing code[:1200], description[:300], max_tokens=512). Same
        # critique applies here as everywhere else in this branch — emit
        # the full code on both sides; the top-50 pre-filter already
        # bounds N. A typical learned skill is 30-60 lines of Python
        # (~2-3k chars), so even 50 skills × 3k = 150k chars is well
        # inside the 1M-token Gemini input window, and the responding
        # JSON is tiny (one boolean + one optional name). Pin max_tokens
        # to the cross-provider ceiling so reasoning + answer always fit.
        def _format_existing(s: dict) -> str:
            code = (s.get("code") or "").strip()
            return (
                f"- name: {s['name']}\n"
                f"  description: {s.get('description', '')}\n"
                f"  code:\n```python\n{code}\n```"
            )

        existing_summaries = "\n".join(_format_existing(s) for s in comparison_set)
        cand_code = (candidate.get("code") or "").strip()
        candidate_desc = (
            f"Name: {candidate['name']}\n"
            f"Description: {candidate.get('description', '')}\n"
            f"Primitives used: {candidate.get('api_primitives_used', [])}\n"
            f"Code:\n```python\n{cand_code}\n```"
        )

        try:
            from rats.agents.base_agent import query_llm_json
            result = query_llm_json(
                "You are a skill deduplication checker for a robot learning system. "
                "Respond only in valid JSON.",
                f"Is this new skill functionally equivalent to any existing skill?\n\n"
                f"NEW SKILL:\n{candidate_desc}\n\n"
                f"EXISTING SKILLS:\n{existing_summaries}\n\n"
                f"Compare the actual code, not just names. Two skills are duplicates "
                f"when their code does the same physical thing (same primitive sequence, "
                f"same effect on the world), even if parameter names differ. A composite "
                f"skill that inlines two existing skills DOES count as a duplicate of "
                f"the union — return the most specific match. Different defaults / "
                f"different parameter shapes around the SAME primitive sequence ARE "
                f"duplicates.\n\n"
                f"Respond with: {{\"is_duplicate\": true/false, \"duplicate_of\": \"skill_name\" or null}}",
                max_tokens=65536,
            )
            if result.get("is_duplicate") and result.get("duplicate_of"):
                dup_name = result["duplicate_of"]
                # Validate the name actually exists in the full library
                if any(s["name"] == dup_name for s in existing):
                    return dup_name
        except Exception as e:
            logger.debug(f"Semantic dedup check failed: {e}")
        return None

    def get_skill_by_name(self, name: str) -> dict[str, Any] | None:
        """Retrieve a skill by name."""
        for s in self._skills:
            if s["name"] == name:
                return s
        return None

    def mark_duplicate(
        self,
        duplicate_name: str,
        canonical_name: str,
        reason: str = "",
        *,
        iteration: int | None = None,
        source: str = "skill_curator",
    ) -> bool:
        """Fold a learned skill into another as a curator-decided duplicate.

        Sets the duplicate to tier='deprecated', records ``duplicate_of``
        and ``curator_reason`` for audit, folds the duplicate's empirical
        counters into the canonical skill, and rewrites every other learned
        wrapper's ``dependent_skills`` list so references to the duplicate
        now point at the canonical. After rewrite, nothing depends on the
        duplicate, so deprecating it can't orphan any wrapper at exec time.

        Refuses no-ops: primitives, missing names, self-reference.
        """
        if not duplicate_name or not canonical_name:
            return False
        if duplicate_name == canonical_name:
            return False
        dup = self.get_skill_by_name(duplicate_name)
        canon = self.get_skill_by_name(canonical_name)
        if not dup or not canon:
            logger.info(
                f"  curator: mark_duplicate skipped — "
                f"dup={duplicate_name!r} canon={canonical_name!r} "
                f"(one or both missing)"
            )
            return False
        if dup.get("is_primitive") or canon.get("is_primitive"):
            logger.info(
                "  curator: mark_duplicate refused — primitives are immutable"
            )
            return False
        dup["tier"] = "deprecated"
        dup["duplicate_of"] = canonical_name
        if iteration is not None:
            dup["deprecated_at_iteration"] = int(iteration)
        dup["deprecated_by"] = source
        if reason:
            dup["curator_reason"] = reason[:400]
        canon["usage_count"] = int(canon.get("usage_count", 0)) + int(
            dup.get("usage_count", 0)
        )
        canon["success_count"] = int(canon.get("success_count", 0)) + int(
            dup.get("success_count", 0)
        )
        canon["rediscovery_count"] = int(canon.get("rediscovery_count", 0)) + int(
            dup.get("rediscovery_count", 0) or 1
        )
        n = int(canon.get("usage_count", 0))
        canon["success_rate"] = (
            int(canon.get("success_count", 0)) / n if n else 0.0
        )
        self._update_tier(canon)
        # Rewrite dependent_skills graph so no active wrapper points at the
        # deprecated duplicate. Skips the duplicate itself and the canonical
        # (canonical pointing at itself would be nonsense).
        for s in self._skills:
            if s.get("is_primitive"):
                continue
            if s is dup or s is canon:
                continue
            deps = s.get("dependent_skills") or []
            if duplicate_name not in deps:
                continue
            new_deps = [
                canonical_name if d == duplicate_name else d for d in deps
            ]
            # Dedup while preserving order.
            seen: set[str] = set()
            deduped: list[str] = []
            for d in new_deps:
                if d not in seen:
                    seen.add(d)
                    deduped.append(d)
            s["dependent_skills"] = deduped
        self._save()
        logger.info(
            f"  curator: merged {duplicate_name!r} → {canonical_name!r} "
            f"(usage={canon['success_count']}/{canon['usage_count']}, "
            f"rediscovery_count={canon['rediscovery_count']})"
        )
        return True

    def deprecate_by_curator(
        self,
        name: str,
        reason: str = "",
        *,
        iteration: int | None = None,
        source: str = "skill_curator",
    ) -> bool:
        """Curator-driven deprecation, independent of Wilson/tier math.

        Honours the same dependency guard as the usage-driven deprecator:
        if any non-deprecated learned skill still lists this one in its
        ``dependent_skills``, we DON'T deprecate — the wrapper would
        orphan. Caller should MERGE the dependent first, or DELETE it, or
        wait. Returns True only when the tier actually changed.
        """
        s = self.get_skill_by_name(name)
        if not s or s.get("is_primitive"):
            return False
        if s.get("tier") == "deprecated":
            return False
        depended_on = any(
            name in (other.get("dependent_skills") or [])
            and other.get("tier") != "deprecated"
            and not other.get("is_primitive")
            and other.get("name") != name
            for other in self._skills
        )
        if depended_on:
            logger.info(
                f"  curator: deprecate {name!r} refused — "
                f"active wrapper still depends on it"
            )
            return False
        s["tier"] = "deprecated"
        if iteration is not None:
            s["deprecated_at_iteration"] = int(iteration)
        s["deprecated_by"] = source
        if reason:
            s["curator_reason"] = reason[:400]
        self._save()
        logger.info(f"  curator: deprecated {name!r}")
        return True

    def rewrite_skill_code(
        self,
        name: str,
        new_code: str,
        new_description: str = "",
        rationale: str = "",
    ) -> bool:
        """Curator-driven REWRITE: replace a skill's body with a more general
        version. Used to de-hardcode literals (e.g. ``object_name="butter"``
        baked into the body becomes a parameter with a default).

        Preconditions:
          - skill exists, is not a primitive, is not deprecated
          - ``new_code`` parses as Python
          - the def line in ``new_code`` keeps the same function name
            (otherwise dependent skills would break their import-by-name
            assumption)

        On success: previous code is appended to ``rewrite_history`` for
        audit, ``rewrite_count`` bumps, ``description`` is updated if
        provided, and the rationale is recorded.
        """
        if not name or not new_code:
            return False
        s = self.get_skill_by_name(name)
        if not s or s.get("is_primitive"):
            logger.info(
                f"  curator: rewrite refused — {name!r} missing or primitive"
            )
            return False
        if s.get("tier") == "deprecated":
            logger.info(
                f"  curator: rewrite refused — {name!r} is deprecated"
            )
            return False
        # AST-validate the proposed code so we don't store something the
        # executor will choke on. Any SyntaxError or compile error means
        # the curator gave us junk; reject and let it try next tick.
        import ast as _ast
        try:
            tree = _ast.parse(new_code)
        except SyntaxError as e:
            logger.warning(
                f"  curator: rewrite of {name!r} failed AST parse: {e}"
            )
            return False
        # Find a top-level FunctionDef whose name matches `name`. If the
        # rewrite renames the function, downstream callers will break, so
        # we refuse rather than corrupt the library.
        has_matching_def = any(
            isinstance(node, _ast.FunctionDef) and node.name == name
            for node in tree.body
        )
        if not has_matching_def:
            logger.warning(
                f"  curator: rewrite of {name!r} rejected — new code "
                f"does not contain a top-level `def {name}(...):`"
            )
            return False

        # Audit: stash the previous body before overwriting.
        history = s.get("rewrite_history") or []
        history.append({
            "timestamp": time.time(),
            "previous_code": s.get("code", ""),
            "previous_description": s.get("description", ""),
            "rationale": (rationale or "")[:600],
        })
        s["rewrite_history"] = history[-5:]  # cap retention
        s["rewrite_count"] = int(s.get("rewrite_count", 0)) + 1
        s["code"] = new_code
        if new_description:
            s["description"] = new_description[:400]
        if rationale:
            s["curator_reason"] = rationale[:400]
        self._save()
        logger.info(
            f"  curator: rewrote {name!r} (rewrite_count="
            f"{s['rewrite_count']}) — {rationale[:120]}"
        )
        return True

    def get_learned_skills_for_curator(
        self, *, include_full_code: bool = False, include_deprecated: bool = False,
    ) -> list[dict[str, Any]]:
        """Payload for the MemoryCurator's skill-curation pass.

        Default: code preview only (def line + docstring + first few body
        lines) — enough to spot near-duplicates by structure without
        shipping every body. With ``include_full_code=True`` we also pass
        the full body, which the curator needs to author REWRITE actions
        that turn hardcoded literals into parameters. Deprecated skills are
        hidden by default so the curator cannot merge fresh skills into
        entries that have already been retired.
        """
        def _preview(code: str, lines: int = 12) -> str:
            if not code:
                return ""
            parts = code.splitlines()
            return "\n".join(parts[:lines])

        out: list[dict[str, Any]] = []
        for s in self._skills:
            if s.get("is_primitive"):
                continue
            if s.get("tier") == "deprecated" and not include_deprecated:
                continue
            entry = {
                "name": s.get("name"),
                "description": s.get("description", ""),
                "tier": s.get("tier", "experimental"),
                "usage_count": int(s.get("usage_count", 0)),
                "success_count": int(s.get("success_count", 0)),
                "rediscovery_count": int(s.get("rediscovery_count", 0)),
                "api_primitives_used": s.get("api_primitives_used", []),
                "dependent_skills": s.get("dependent_skills", []),
                "duplicate_of": s.get("duplicate_of"),
                "rewrite_count": int(s.get("rewrite_count", 0)),
                "code_preview": _preview(s.get("code", "")),
            }
            if include_full_code:
                entry["code"] = s.get("code", "")
            out.append(entry)
        return out

    def get_learned_skill_count(self) -> int:
        """Return count of non-primitive learned skills."""
        return sum(1 for s in self._skills if not s.get("is_primitive", False))

    def get_all_skill_names(self) -> list[str]:
        """Return all skill names."""
        return [s["name"] for s in self._skills]

    def reset(self) -> None:
        """Reset to initial primitives only."""
        self._skills = build_initial_skills()
        self._save()
