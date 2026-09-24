"""Curiosity scoring for candidate task proposals.

Two scoring modes:
  - ``llm``: novelty / frontier come directly from the proposer LLM (as
    ``llm_novelty_score`` / ``llm_frontier_score`` fields on the candidate
    dict). Used unchanged after clipping to [0, 1].
  - ``formula``: novelty is computed from how often the candidate's
    (object, skill) pairs have been attempted before; frontier is
    computed from mean Wilson-lower-bound reliability of the candidate's
    required skills via 4 c (1 - c) (the Goldilocks parabola).

A retry bonus and a recent-failure penalty are added on top regardless of
which mode produced the base novelty / frontier values. The final score
composition is either ``product`` (``N * F + ...``) or ``weighted_sum``
(``0.5 N + 0.5 F + ...``).

Object-skill attempt counts are kept in a plain ``dict[(obj, skill), int]``
maintained by the lifelong loop and persisted as JSON. The scoring functions
here are pure — they take that dict as input but do not write to it.
"""

from __future__ import annotations

import math
from typing import Any, Callable


def _clip01(x: float | None) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def compute_object_skill_novelty(
    candidate: dict[str, Any],
    counts: dict[tuple[str, str], int],
) -> float:
    """Mean of 1/sqrt(N(o,s)+1) over (object, skill) pairs in the candidate.

    Returns 0.5 when either the object list or required-skill list is empty
    (no signal — treat as neutral).
    """
    objects = [
        str(o).strip().lower()
        for o in (candidate.get("objects") or [])
        if str(o).strip()
    ]
    skills = [
        str(s).strip().lower()
        for s in (candidate.get("required_skills") or [])
        if str(s).strip()
    ]
    if not objects or not skills:
        return 0.5
    pairs = [(o, s) for o in objects for s in skills]
    if not pairs:
        return 0.5
    vals = [1.0 / math.sqrt(int(counts.get(pair, 0)) + 1.0) for pair in pairs]
    return sum(vals) / len(vals)


def compute_competence(
    candidate: dict[str, Any],
    skill_lookup: Callable[[str], float | None],
    *,
    missing_skill_reliability: float = 0.05,
) -> float:
    """Mean reliability of the candidate's required skills.

    ``skill_lookup(name)`` returns the Wilson lower bound for a learned
    skill, a fixed baseline (~0.9) for primitives, or ``None`` for skills
    not in the library. Unknown skills are treated as
    ``missing_skill_reliability`` (default 0.05 — "present but unproven").

    Returns 0.5 when no required skills are declared.
    """
    required = [
        str(s).strip()
        for s in (candidate.get("required_skills") or [])
        if str(s).strip()
    ]
    if not required:
        return 0.5
    rels: list[float] = []
    for name in required:
        r = skill_lookup(name)
        if r is None:
            r = missing_skill_reliability
        rels.append(_clip01(r))
    return sum(rels) / len(rels)


def compute_frontier(competence: float) -> float:
    """Goldilocks parabola: peaks at competence = 0.5 with value 1.0."""
    c = _clip01(competence)
    return 4.0 * c * (1.0 - c)


def _word_set(text: str) -> set[str]:
    text = text.lower().replace("_", " ")
    return {w for w in text.split() if len(w) > 2}


def _candidate_word_set(candidate: dict[str, Any]) -> set[str]:
    parts: list[str] = [
        str(candidate.get("language", "")),
        str(candidate.get("activity_name", "")),
    ]
    parts.extend(str(o) for o in (candidate.get("objects") or []))
    parts.extend(str(f) for f in (candidate.get("fixtures") or []))
    return _word_set(" ".join(parts))


def _entry_word_set(entry: dict[str, Any]) -> set[str]:
    parts: list[str] = [
        str(entry.get("language", "")),
        str(entry.get("activity_name", "")),
    ]
    parts.extend(str(o) for o in (entry.get("objects_used") or []))
    parts.extend(str(f) for f in (entry.get("fixtures_used") or []))
    return _word_set(" ".join(parts))


def compute_recent_failure_penalty(
    candidate: dict[str, Any],
    task_history: list[dict[str, Any]],
    *,
    window: int = 10,
    similarity_threshold: float = 0.6,
) -> float:
    """Penalty for candidates that look like recent failures.

    For each failure in the last ``window`` entries, compute Jaccard
    overlap of word sets. Failures whose overlap is >= ``similarity_threshold``
    contribute their overlap to the penalty. Capped at 1.0.

    Cheap proxy — better than nothing, but the real "same task again"
    test should eventually use object + goal predicate equality.
    """
    if not task_history:
        return 0.0
    cand_words = _candidate_word_set(candidate)
    if not cand_words:
        return 0.0
    pen = 0.0
    for entry in task_history[-window:]:
        if entry.get("success"):
            continue
        entry_words = _entry_word_set(entry)
        if not entry_words:
            continue
        union = max(len(cand_words), len(entry_words))
        if union == 0:
            continue
        overlap = len(cand_words & entry_words) / union
        if overlap >= similarity_threshold:
            pen += overlap
    return min(1.0, pen)


def compute_retry_bonus(retry_item: dict[str, Any], max_ttl: int) -> float:
    """retry_bonus = surprise * diagnosable * ttl_decay.

    ``ttl_decay = ttl / max_ttl`` so fresh items (high ttl) bonus more
    than nearly-expired ones. Non-diagnosable items contribute zero.
    """
    if not retry_item.get("diagnosable", False):
        return 0.0
    surprise = _clip01(retry_item.get("surprise_score", 0.0))
    ttl = max(0, int(retry_item.get("ttl", 0)))
    decay = ttl / max(1, int(max_ttl))
    return _clip01(surprise * decay)


def score_candidate(
    candidate: dict[str, Any],
    *,
    mode: str,
    skill_lookup: Callable[[str], float | None],
    history_counts: dict[tuple[str, str], int],
    task_history: list[dict[str, Any]],
    retry_bonus_weight: float,
    failure_penalty_weight: float,
    score_composition: str,
) -> dict[str, Any]:
    """Annotate candidate with novelty / frontier / final_score in place.

    Formula values are ALWAYS computed (cheap) and stored under
    ``formula_novelty_score`` / ``formula_frontier_score`` so logs can
    compare LLM vs formula side by side regardless of which mode drives
    the final selection. The active values are mirrored into
    ``novelty_score`` / ``frontier_score`` for downstream consumers.

    ``mode`` must be ``"llm"`` or ``"formula"``. ``score_composition``
    must be ``"product"`` or ``"weighted_sum"``.
    """
    f_nov = compute_object_skill_novelty(candidate, history_counts)
    comp = compute_competence(candidate, skill_lookup)
    f_front = compute_frontier(comp)
    candidate["formula_novelty_score"] = round(f_nov, 4)
    candidate["formula_frontier_score"] = round(f_front, 4)
    candidate["competence_estimate"] = round(comp, 4)

    raw_l_nov = candidate.get("llm_novelty_score")
    raw_l_front = candidate.get("llm_frontier_score")
    if raw_l_nov is not None:
        candidate["llm_novelty_score"] = round(_clip01(raw_l_nov), 4)
    if raw_l_front is not None:
        candidate["llm_frontier_score"] = round(_clip01(raw_l_front), 4)

    if mode == "llm":
        novelty = candidate.get("llm_novelty_score")
        frontier = candidate.get("llm_frontier_score")
        if novelty is None:
            novelty = f_nov
        if frontier is None:
            frontier = f_front
    else:  # formula
        novelty = f_nov
        frontier = f_front

    novelty = _clip01(novelty)
    frontier = _clip01(frontier)
    candidate["novelty_score"] = round(novelty, 4)
    candidate["frontier_score"] = round(frontier, 4)

    retry_bonus = float(candidate.get("retry_bonus_score", 0.0) or 0.0)
    failure_penalty = compute_recent_failure_penalty(candidate, task_history)
    candidate["retry_bonus_score"] = round(_clip01(retry_bonus), 4)
    candidate["failure_penalty"] = round(failure_penalty, 4)

    if score_composition == "weighted_sum":
        base = 0.5 * novelty + 0.5 * frontier
    else:
        base = novelty * frontier  # product (default)

    final = (
        base
        + float(retry_bonus_weight) * retry_bonus
        - float(failure_penalty_weight) * failure_penalty
    )
    candidate["final_score"] = round(max(0.0, final), 4)
    candidate["score_breakdown"] = {
        "mode": mode,
        "composition": score_composition,
        "novelty_used": round(novelty, 4),
        "frontier_used": round(frontier, 4),
        "base": round(base, 4),
        "retry_bonus_term": round(float(retry_bonus_weight) * retry_bonus, 4),
        "failure_penalty_term": round(
            float(failure_penalty_weight) * failure_penalty, 4
        ),
        "final": candidate["final_score"],
    }
    return candidate


def make_skill_lookup(skill_library: Any) -> Callable[[str], float | None]:
    """Build a name → reliability lookup closure over a SkillLibrary.

    Tries exact name match, then case-insensitive match, then token-level
    fuzzy match ("grasp" inside "grasp_with_wrist_closeloop"). Returns
    0.9 for primitives, Wilson lower bound for learned skills, ``None``
    for skills not present in the library.
    """
    skills = list(getattr(skill_library, "_skills", []) or [])
    by_name = {str(s.get("name") or ""): s for s in skills}
    by_lower = {str(s.get("name") or "").lower(): s for s in skills}

    from skill_library.library import wilson_lower_bound

    def _wilson(s: dict[str, Any]) -> float:
        return wilson_lower_bound(
            int(s.get("success_count", 0)),
            int(s.get("usage_count", 0)),
        )

    def lookup(name: str) -> float | None:
        if not name:
            return None
        s = by_name.get(name)
        if s is None:
            s = by_lower.get(name.lower())
        if s is None:
            target = name.lower()
            for sk in skills:
                tokens = set(str(sk.get("name") or "").lower().split("_"))
                if target in tokens:
                    s = sk
                    break
        if s is None:
            return None
        if s.get("is_primitive"):
            return 0.9
        return _wilson(s)

    return lookup


def load_object_skill_counts(path: str | None) -> dict[tuple[str, str], int]:
    """Load object-skill attempt counts from a JSON file.

    Returns an empty dict if the file is missing or unparseable — the
    counter is recoverable, not load-bearing for correctness.
    """
    if not path:
        return {}
    import json
    import os

    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {}
    out: dict[tuple[str, str], int] = {}
    for entry in data or []:
        if not isinstance(entry, dict):
            continue
        o = str(entry.get("object", "")).strip().lower()
        s = str(entry.get("skill", "")).strip().lower()
        n = int(entry.get("count", 0))
        if o and s and n > 0:
            out[(o, s)] = n
    return out


def save_object_skill_counts(
    counts: dict[tuple[str, str], int], path: str | None,
) -> None:
    if not path:
        return
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = [
        {"object": o, "skill": s, "count": n}
        for (o, s), n in sorted(counts.items())
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def update_object_skill_counts(
    counts: dict[tuple[str, str], int],
    objects: list[str] | None,
    skills: list[str] | None,
) -> None:
    """In-place +1 for every (object, skill) pair from the executed task."""
    objs = [str(o).strip().lower() for o in (objects or []) if str(o).strip()]
    sks = [str(s).strip().lower() for s in (skills or []) if str(s).strip()]
    for o in objs:
        for s in sks:
            counts[(o, s)] = counts.get((o, s), 0) + 1
