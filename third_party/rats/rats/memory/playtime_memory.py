"""Lightweight memory for Piaget-style MolmoSpaces playtime observations.

Two layers of storage:

1. **Raw log** (the original behavior) — one JSONL entry per play attempt
   with the verifier's free-form ``observed_effect`` and ``resulting_state``.
   Useful as evidence and for auditing.

2. **Structured affordance cards** — derived from the raw log on every
   ``add()`` and rebuilt on ``_load()``. Each card aggregates typed
   boolean/qualitative affordance facts about an object (and the
   object's category for cross-house generalization), with running
   support/contradiction weights so confidence is computable. Cards
   are what the proposer and policy writer actually need: a stable
   "what we know about this object" view rather than a flat list of
   prose summaries.

Negative results (object did not move, drawer resisted push) are
first-class — they update the same affordance keys with a False value
and contribute to the contradiction weight, so subsequent VLM prompts
see an aggregated belief instead of the most-recent-wins free text.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


# Affordance key vocabulary. Stable strings the proposer / policy writer
# can pattern-match without trying to parse free-form English. Adding
# a new key here is non-breaking; removing one is.
AFFORDANCE_KEYS = (
    "moves_under_light_contact",   # push / tap / touch made the object translate
    "rolls",                       # round-ish: rolled when pushed or after release
    "tippable",                    # observed to wobble / fall on its side
    "resists_motion",              # gripper made contact but object stayed put
    "contact_sensitive",           # high VLM contact_sensitivity rating
    "graspable_topdown",           # at least one verified top-down grasp
    "articulated_movable",         # open/close caused visible joint motion
    "slides_on_surface",           # medium/large translation under push/slide
)


def _derive_affordances(entry: dict[str, Any]) -> list[tuple[str, bool, float]]:
    """Extract typed (key, value, weight) facts from one raw memory entry.

    Each fact's weight is the verifier's reported confidence (0..1)
    times a per-rule prior reflecting how reliable the inference is.
    The card aggregator sums supporting vs contradicting weight to
    derive a final belief + confidence.

    Negative results are intentionally produced — e.g. a push that did
    not move the object yields ``("moves_under_light_contact", False, w)``.
    Contradicting evidence is information, not noise.
    """
    rs = entry.get("resulting_state") or {}
    if not isinstance(rs, dict):
        rs = {}
    interaction = str(entry.get("interaction_type") or "").lower()
    success = bool(entry.get("success", False))
    moved = rs.get("object_moved")
    moved_bool: bool | None
    if isinstance(moved, bool):
        moved_bool = moved
    elif isinstance(moved, str):
        moved_bool = moved.strip().lower() in {"true", "yes", "1"}
    else:
        moved_bool = None
    direction = str(rs.get("movement_direction") or "").lower()
    stability = str(rs.get("object_stability") or "").lower()
    sens = str(rs.get("contact_sensitivity") or "").lower()
    magnitude = str(rs.get("movement_magnitude") or "").lower()

    light_contact = {"push", "tap", "touch", "slide", "knock_over"}
    grasp_actions = {"lift", "pick", "place_in", "place_on", "stack"}

    facts: list[tuple[str, bool, float]] = []

    if interaction in light_contact and moved_bool is not None:
        facts.append(("moves_under_light_contact", moved_bool, 1.0))

    if interaction == "roll":
        facts.append(("rolls", bool(moved_bool) if moved_bool is not None else success, 0.9))
    elif "roll" in direction:
        facts.append(("rolls", True, 0.6))

    if stability == "unstable":
        facts.append(("tippable", True, 0.8))
    elif stability == "stable":
        facts.append(("tippable", False, 0.6))
    elif stability == "resistant":
        facts.append(("resists_motion", True, 0.9))

    if sens == "high":
        facts.append(("contact_sensitive", True, 1.0))
    elif sens == "low":
        facts.append(("contact_sensitive", False, 0.7))

    if interaction in grasp_actions and success:
        facts.append(("graspable_topdown", True, 1.0))
    elif interaction in {"lift", "pick"} and not success:
        # Failed pick is weak evidence of ungraspable: could be pose error,
        # not an object-property fact. Keep the weight low so a single
        # subsequent success flips it.
        facts.append(("graspable_topdown", False, 0.4))

    if interaction in {"open", "close"}:
        if success:
            facts.append(("articulated_movable", True, 1.0))
        elif stability == "resistant":
            facts.append(("articulated_movable", False, 0.5))

    if interaction in {"push", "slide"}:
        if magnitude in {"medium", "large"}:
            facts.append(("slides_on_surface", True, 0.9))
        elif magnitude == "none":
            facts.append(("slides_on_surface", False, 0.7))

    base_conf: float
    raw_conf = entry.get("confidence", entry.get("verifier_confidence"))
    try:
        base_conf = float(raw_conf) if raw_conf is not None else 1.0
    except (TypeError, ValueError):
        base_conf = 1.0
    base_conf = max(0.0, min(1.0, base_conf))

    return [(k, v, max(0.0, min(1.0, w * base_conf))) for k, v, w in facts]


def _category_from_internal_name(internal_name: str | None) -> str:
    """Best-effort category extraction from an internal_name like ``winebottle_<hash>_1_0_2``.

    The bridge inventory carries category as a separate field, but raw
    entries written before category was logged don't have it. Splitting on
    the first underscore is a stable fallback for ProcTHOR-style names.
    """
    if not internal_name:
        return ""
    head = str(internal_name).split("_", 1)[0]
    return head.lower()


class PlaytimeMemory:
    """JSONL-backed store of exploratory object-affordance observations.

    Adds a structured affordance card layer on top of the raw log so the
    proposer / policy writer can read typed facts about objects (and
    per-category rollups) instead of re-parsing free-form English on
    every iteration.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_entries: int = 1000,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.read_only = bool(read_only)
        if not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max(1, int(max_entries))
        self._entries: list[dict[str, Any]] = []
        # internal_name -> {object, category, n, last_iter, affordances{key->{pos,neg,iters}}}
        self._object_cards: dict[str, dict[str, Any]] = {}
        # category -> same shape but aggregated across all objects in the category
        self._category_cards: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        self._object_cards = {}
        self._category_cards = {}
        if not self.path.exists():
            self._entries = []
            return
        entries: list[dict[str, Any]] = []
        try:
            for raw in self.path.read_text().splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    entries.append(obj)
        except OSError:
            entries = []
        self._entries = entries[-self.max_entries:]
        for entry in self._entries:
            self._update_cards(entry)

    def _persist(self) -> None:
        # No-op for read-only seed memories so a stray add() (or a
        # downstream caller that doesn't know it's holding a seed) can't
        # corrupt the source JSONL on disk.
        if self.read_only:
            return
        data = self._entries[-self.max_entries:]
        payload = "\n".join(json.dumps(e, sort_keys=True) for e in data)
        self.path.write_text(payload + ("\n" if payload else ""))

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    @property
    def object_cards(self) -> dict[str, dict[str, Any]]:
        return {k: v for k, v in self._object_cards.items()}

    @property
    def category_cards(self) -> dict[str, dict[str, Any]]:
        return {k: v for k, v in self._category_cards.items()}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, entry: dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            return
        self._entries.append(entry)
        self._entries = self._entries[-self.max_entries:]
        self._update_cards(entry)
        self._persist()

    @staticmethod
    def _new_card(*, object_name: str, internal_name: str, category: str) -> dict[str, Any]:
        return {
            "object_name": object_name,
            "internal_name": internal_name,
            "category": category,
            "n_observations": 0,
            "last_seen_iteration": 0,
            "affordances": {},
        }

    def _update_cards(self, entry: dict[str, Any]) -> None:
        internal = str(entry.get("target_internal_name") or "")
        display = str(
            entry.get("target_display_name")
            or entry.get("target_object")
            or internal
            or "object"
        )
        category = str(
            entry.get("target_category")
            or entry.get("category")
            or _category_from_internal_name(internal)
            or ""
        ).lower()
        iteration = int(entry.get("iteration", 0) or 0)
        derived = _derive_affordances(entry)

        # Per-object card (keyed by internal_name; falls back to display).
        obj_key = internal or display
        obj_card = self._object_cards.get(obj_key)
        if obj_card is None:
            obj_card = self._new_card(
                object_name=display, internal_name=internal, category=category,
            )
            self._object_cards[obj_key] = obj_card
        obj_card["n_observations"] += 1
        obj_card["last_seen_iteration"] = max(
            obj_card.get("last_seen_iteration", 0), iteration,
        )
        if not obj_card.get("category") and category:
            obj_card["category"] = category
        self._merge_facts(obj_card["affordances"], derived, iteration)

        # Per-category rollup. Aggregates evidence across all objects of
        # the same category so a fresh house with a never-before-seen
        # spoon still benefits from "all spoons we've seen slide easily".
        if category:
            cat_card = self._category_cards.get(category)
            if cat_card is None:
                cat_card = self._new_card(
                    object_name=f"<category:{category}>",
                    internal_name=category,
                    category=category,
                )
                self._category_cards[category] = cat_card
            cat_card["n_observations"] += 1
            cat_card["last_seen_iteration"] = max(
                cat_card.get("last_seen_iteration", 0), iteration,
            )
            self._merge_facts(cat_card["affordances"], derived, iteration)

    @staticmethod
    def _merge_facts(
        bucket: dict[str, dict[str, Any]],
        derived: list[tuple[str, bool, float]],
        iteration: int,
    ) -> None:
        for key, value, weight in derived:
            aff = bucket.setdefault(
                key,
                {
                    "supporting_weight": 0.0,
                    "contradicting_weight": 0.0,
                    "n_supporting": 0,
                    "n_contradicting": 0,
                    "evidence_iterations": [],
                },
            )
            if value:
                aff["supporting_weight"] = round(aff["supporting_weight"] + weight, 4)
                aff["n_supporting"] = int(aff["n_supporting"]) + 1
            else:
                aff["contradicting_weight"] = round(aff["contradicting_weight"] + weight, 4)
                aff["n_contradicting"] = int(aff["n_contradicting"]) + 1
            iters: list[int] = list(aff.get("evidence_iterations") or [])
            if iteration and iteration not in iters:
                iters.append(iteration)
                aff["evidence_iterations"] = iters[-12:]

    # ------------------------------------------------------------------
    # Belief readout
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_aff(aff: dict[str, Any]) -> tuple[bool | None, float, int, int]:
        """Collapse one affordance bucket to (value, confidence, n_pos, n_neg).

        Confidence is capped at 0.95 so a single observation never reads
        as certainty; the proposer/writer always has room to override.
        """
        pos = float(aff.get("supporting_weight", 0.0) or 0.0)
        neg = float(aff.get("contradicting_weight", 0.0) or 0.0)
        n_pos = int(aff.get("n_supporting", 0) or 0)
        n_neg = int(aff.get("n_contradicting", 0) or 0)
        total = pos + neg
        if total <= 0:
            return None, 0.0, n_pos, n_neg
        value = pos > neg if pos != neg else None
        confidence = max(pos, neg) / total
        confidence = min(confidence, 0.95)
        return value, confidence, n_pos, n_neg

    def _format_card(
        self,
        card: dict[str, Any],
        *,
        action_filter: str | None = None,
    ) -> str:
        """Render an object/category card as one human-readable block."""
        affs = card.get("affordances") or {}
        relevant_keys = AFFORDANCE_KEYS
        if action_filter:
            relevant_keys = self._affordances_for_action(action_filter) or AFFORDANCE_KEYS
        lines: list[str] = []
        for key in relevant_keys:
            aff = affs.get(key)
            if not aff:
                continue
            value, conf, n_pos, n_neg = self._resolve_aff(aff)
            if value is None:
                continue
            n = n_pos + n_neg
            tag = "yes" if value else "no"
            extra = ""
            if n_pos > 0 and n_neg > 0:
                extra = f", contested {n_pos}+/{n_neg}-"
            lines.append(f"    {key}: {tag} (conf={conf:.2f}, n={n}{extra})")
        if not lines:
            return ""
        header_obj = card.get("object_name") or card.get("internal_name") or "object"
        cat = card.get("category") or ""
        n_obs = int(card.get("n_observations", 0))
        last = int(card.get("last_seen_iteration", 0))
        header_extra = f"category={cat}, " if cat and not header_obj.startswith("<category:") else ""
        header = (
            f"  - {header_obj} ({header_extra}n_obs={n_obs}, last_iter={last}):"
        )
        return "\n".join([header, *lines])

    @staticmethod
    def _affordances_for_action(action: str) -> tuple[str, ...]:
        """Subset of affordance keys most relevant to a given action verb."""
        action = action.strip().lower()
        if action in {"push", "tap", "touch", "slide", "knock_over"}:
            return (
                "moves_under_light_contact", "slides_on_surface", "tippable",
                "contact_sensitive", "rolls", "resists_motion",
            )
        if action in {"lift", "pick", "place_in", "place_on", "stack"}:
            return (
                "graspable_topdown", "tippable", "contact_sensitive",
                "rolls",
            )
        if action in {"open", "close", "pull"}:
            return (
                "articulated_movable", "resists_motion", "contact_sensitive",
            )
        if action in {"shake", "drop"}:
            return (
                "graspable_topdown", "tippable", "contact_sensitive",
            )
        return AFFORDANCE_KEYS

    def summarize_affordances_for_proposer(
        self,
        *,
        max_objects: int = 12,
        max_categories: int = 6,
    ) -> str:
        """Compact AFFORDANCE FACTS block for the playtime proposer prompt.

        Rendered as two sub-blocks so the proposer can tell apart
        per-instance evidence (same house) from category rollups (cross
        house). Returns the empty string when nothing is known yet so
        callers can decide whether to inject the section at all.
        """
        if not self._object_cards and not self._category_cards:
            return ""
        # Sort objects by recency * sample size; categories by sample size.
        obj_sorted = sorted(
            self._object_cards.values(),
            key=lambda c: (
                int(c.get("last_seen_iteration", 0)),
                int(c.get("n_observations", 0)),
            ),
            reverse=True,
        )[:max_objects]
        cat_sorted = sorted(
            self._category_cards.values(),
            key=lambda c: int(c.get("n_observations", 0)),
            reverse=True,
        )[:max_categories]
        chunks: list[str] = []
        obj_blocks = [b for b in (self._format_card(c) for c in obj_sorted) if b]
        if obj_blocks:
            chunks.append("Per-object affordance facts (this run):")
            chunks.extend(obj_blocks)
        cat_blocks = [b for b in (self._format_card(c) for c in cat_sorted) if b]
        if cat_blocks:
            chunks.append("Per-category rollups (transferable across houses):")
            chunks.extend(cat_blocks)
        return "\n".join(chunks)

    # Affordance keys whose semantics translate cleanly to the
    # benchmark task families (pick / pick_and_place / open / close).
    # `tippable` and `slides_on_surface` matter when the gripper has
    # to approach a fragile object without knocking it; `rolls`
    # warns the policy writer about post-release wandering. Anything
    # not in this set is exploration-only noise from a benchmark's
    # perspective and would just inflate the prompt.
    _BENCHMARK_RELEVANT_AFFORDANCES = (
        "graspable_topdown",
        "tippable",
        "contact_sensitive",
        "rolls",
        "slides_on_surface",
        "articulated_movable",
        "resists_motion",
    )

    def summarize_for_benchmark(
        self,
        *,
        objects: list[str] | None = None,
        action: str | None = None,
        max_categories: int = 8,
    ) -> str:
        """Cross-run summary for benchmark tasks: category cards only.

        Per-object cards are house-specific (their keys are
        ProcTHOR-style internal_names that won't match a different
        run's inventory), so this method intentionally drops them and
        returns only the category rollups. Callers should pass the
        benchmark task's target categories or internal_names — this
        method also derives categories from the latter via the same
        ``head_before_underscore`` rule used during ingest.

        Affordance keys are restricted to those that actually shape
        benchmark behavior (``_BENCHMARK_RELEVANT_AFFORDANCES``); the
        rest are pruned to keep the policy-writer prompt cheap.

        Returns the empty string when nothing transferable is known so
        callers can decide whether to inject the section at all.
        """
        if not self._category_cards:
            return ""
        wanted: set[str] = set()
        if objects:
            for raw in objects:
                text = str(raw or "").strip().lower()
                if not text:
                    continue
                wanted.add(text)
                head = _category_from_internal_name(text)
                if head:
                    wanted.add(head)
        # Pick which categories to render. If the caller passed objects
        # we filter to those; otherwise we surface the most-observed
        # categories (useful for a benchmark proposer that doesn't know
        # the next task's objects yet).
        if wanted:
            cards = [
                card for cat, card in self._category_cards.items()
                if cat in wanted
            ]
        else:
            cards = list(self._category_cards.values())
        cards.sort(
            key=lambda c: int(c.get("n_observations", 0)),
            reverse=True,
        )
        cards = cards[:max(1, int(max_categories))]
        # Subset of relevant affordance keys, optionally narrowed by action.
        keys: tuple[str, ...] = self._BENCHMARK_RELEVANT_AFFORDANCES
        if action:
            action_keys = self._affordances_for_action(action)
            keys = tuple(k for k in action_keys if k in self._BENCHMARK_RELEVANT_AFFORDANCES) or keys
        rendered: list[str] = []
        for card in cards:
            block = self._format_card_with_keys(card, keys=keys)
            if block:
                rendered.append(block)
        if not rendered:
            return ""
        return (
            "Cross-run playtime affordance facts (from prior playtime "
            "memory; treat as object-property hints, not simulator "
            "guarantees):\n" + "\n".join(rendered)
        )

    def _format_card_with_keys(
        self,
        card: dict[str, Any],
        *,
        keys: tuple[str, ...],
    ) -> str:
        affs = card.get("affordances") or {}
        lines: list[str] = []
        for key in keys:
            aff = affs.get(key)
            if not aff:
                continue
            value, conf, n_pos, n_neg = self._resolve_aff(aff)
            if value is None:
                continue
            n = n_pos + n_neg
            tag = "yes" if value else "no"
            extra = f", contested {n_pos}+/{n_neg}-" if n_pos > 0 and n_neg > 0 else ""
            lines.append(f"    {key}: {tag} (conf={conf:.2f}, n={n}{extra})")
        if not lines:
            return ""
        cat = card.get("category") or card.get("internal_name") or "unknown"
        n_obs = int(card.get("n_observations", 0))
        return "\n".join([f"  - category={cat} (n_obs={n_obs}):", *lines])

    def summarize_affordances_for_policy_writer(
        self,
        *,
        objects: list[str] | None = None,
        action: str | None = None,
    ) -> str:
        """Action-relevant affordance card for the policy writer.

        Filters per-affordance: only keys that matter for ``action`` are
        rendered (see ``_affordances_for_action``). The writer then sees
        ``- the spoon: moves_under_light_contact: yes (conf=0.85, n=2)``
        rather than the entire card. Returns empty string when nothing
        relevant is known.
        """
        if not objects:
            return ""
        wanted_internal: set[str] = set()
        wanted_categories: set[str] = set()
        for raw in objects:
            text = str(raw or "").strip().lower()
            if not text:
                continue
            wanted_internal.add(text)
            wanted_categories.add(_category_from_internal_name(text))
            # Also treat the raw token's first underscore-segment as cat.
        chunks: list[str] = []
        seen_keys: set[str] = set()
        # Per-object first (most specific).
        for key, card in self._object_cards.items():
            obj_lower = str(card.get("object_name", "")).lower()
            internal_lower = str(card.get("internal_name", "")).lower()
            cat = str(card.get("category", "")).lower()
            if not (
                key.lower() in wanted_internal
                or internal_lower in wanted_internal
                or any(w and w in obj_lower for w in wanted_internal if w)
                or cat in wanted_categories
            ):
                continue
            block = self._format_card(card, action_filter=action)
            if block:
                chunks.append(block)
                seen_keys.add(key)
        # Then category rollups, only for categories not already covered
        # by a per-object card.
        covered_cats = {
            str(self._object_cards[k].get("category", "")).lower() for k in seen_keys
        }
        for cat, card in self._category_cards.items():
            if cat in covered_cats or cat not in wanted_categories:
                continue
            block = self._format_card(card, action_filter=action)
            if block:
                chunks.append(block)
        if not chunks:
            return ""
        return "Relevant playtime affordance facts:\n" + "\n".join(chunks)

    # ------------------------------------------------------------------
    # Backward-compatible raw log API (unchanged behaviour)
    # ------------------------------------------------------------------

    @staticmethod
    def _tokens(values: list[Any]) -> set[str]:
        out: set[str] = set()
        for value in values:
            if value is None:
                continue
            text = str(value).lower()
            if text:
                out.add(text)
                out.update(
                    part
                    for part in text.replace("/", " ").replace("_", " ").split()
                    if len(part) >= 3
                )
        return out

    def retrieve_for_task(
        self,
        *,
        objects: list[str] | None = None,
        action: str | None = None,
        task_language: str = "",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return recent entries relevant to objects/action/language."""
        query_tokens = self._tokens(list(objects or []) + [action or "", task_language])
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for idx, entry in enumerate(self._entries):
            rs = entry.get("resulting_state") or {}
            if not isinstance(rs, dict):
                rs = {}
            entry_tokens = self._tokens([
                entry.get("target_object"),
                entry.get("target_internal_name"),
                entry.get("interaction_type"),
                entry.get("observed_effect"),
                rs.get("policy_implication"),
            ])
            score = len(query_tokens & entry_tokens)
            if action and str(entry.get("interaction_type", "")).lower() == str(action).lower():
                score += 2
            if score > 0 or not query_tokens:
                scored.append((score, idx, entry))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [e for _, _, e in scored[:max(0, int(top_k))]]

    @staticmethod
    def _format_entry(entry: dict[str, Any]) -> str:
        rs = entry.get("resulting_state") or {}
        if not isinstance(rs, dict):
            rs = {}
        obj = entry.get("target_object") or entry.get("target_internal_name") or "object"
        action = entry.get("interaction_type") or "interaction"
        effect = entry.get("observed_effect") or rs.get("observed_effect") or "effect unclear"
        implication = rs.get("policy_implication") or entry.get("policy_implication") or ""
        conf = entry.get("confidence") or entry.get("verifier_confidence")
        conf_txt = f" confidence={float(conf):.2f};" if isinstance(conf, (int, float)) else ""
        line = f"- {obj}: after {action}, {effect}.{conf_txt}"
        if implication:
            line += f" Policy implication: {implication}"
        return line

    def summarize_for_proposer(self, *, top_k: int = 20) -> str:
        """Recent prose log + structured affordance cards.

        Backwards-compatible: callers that only used to see the prose
        log will now also see the affordance block prepended. Empty
        affordance block degrades to the original behavior.
        """
        cards_block = self.summarize_affordances_for_proposer()
        entries = self._entries[-max(0, int(top_k)):]
        if not entries:
            log_block = "(no prior playtime observations yet)"
        else:
            log_block = "\n".join(self._format_entry(e) for e in entries)
        if cards_block:
            return (
                cards_block
                + "\n\nRaw observation log (most recent last):\n"
                + log_block
            )
        return log_block

    def summarize_for_policy_writer(
        self,
        *,
        objects: list[str] | None = None,
        task_language: str = "",
        action: str | None = None,
        top_k: int = 8,
    ) -> str:
        """Affordance card (typed) + recent matching prose entries."""
        cards_block = self.summarize_affordances_for_policy_writer(
            objects=objects, action=action,
        )
        entries = self.retrieve_for_task(
            objects=objects,
            action=action,
            task_language=task_language,
            top_k=top_k,
        )
        log_block = "\n".join(self._format_entry(e) for e in entries) if entries else ""
        if cards_block and log_block:
            return cards_block + "\n\nRecent matching observations:\n" + log_block
        if cards_block:
            return cards_block
        return log_block
