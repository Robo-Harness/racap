"""MolmoSpaces task/environment consistency verifier.

This pre-flight guard is intentionally narrower than the task verifier: it only
answers whether the object named by the current task proposal appears in the
current agentview frame. The lifelong loop calls it only after events that are
likely to desynchronise task metadata from the simulator (house switches,
bridge recovery, or environment recreation).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

logger = logging.getLogger("rats.molmospaces_environment_verifier")


class MolmoSpacesEnvironmentVerifier:
    """Use local Molmo pointing to check that the task target is visible."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        provider: str = "molmo",
        max_retries: int = 4,
        fail_open_on_error: bool = True,
        molmo_model: str = "allenai/Molmo2-8B",
        molmo_base_url: str = "http://127.0.0.1:8122/v1",
        molmo_api_key: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.provider = str(provider or "molmo").lower()
        self.max_retries = max(1, int(max_retries or 4))
        self.fail_open_on_error = bool(fail_open_on_error)
        self.molmo_model = str(molmo_model or "allenai/Molmo2-8B")
        self.molmo_base_url = str(molmo_base_url or "http://127.0.0.1:8122/v1")
        self.molmo_api_key = molmo_api_key
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._detector: Callable[[Image.Image, list[str] | None], dict[str, tuple[int | None, int | None]]] | None = None

    def verify(
        self,
        env: Any,
        task_proposal: dict[str, Any],
        scene_context: dict[str, Any] | None = None,
        *,
        iteration: int = 0,
        attempt: int = 0,
        reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-serializable visibility verdict for the current task."""
        reasons = [str(r) for r in (reasons or []) if str(r).strip()]
        base: dict[str, Any] = {
            "enabled": self.enabled,
            "provider": self.provider,
            "iteration": int(iteration),
            "attempt": int(attempt),
            "reasons": reasons,
            "task": {
                "activity_name": task_proposal.get("activity_name"),
                "canonical_task_id": task_proposal.get("canonical_task_id"),
                "language": task_proposal.get("language")
                or task_proposal.get("goal_conditions")
                or (scene_context or {}).get("goal_conditions_nl"),
                "objects": task_proposal.get("objects", []),
                "task_family": task_proposal.get("task_family"),
            },
            "queries": [],
            "points": {},
            "suitable": True,
            "reason": "disabled",
            "error": None,
            "artifacts": {},
        }
        if not self.enabled:
            return base
        if self.provider != "molmo":
            base.update({"suitable": self.fail_open_on_error, "reason": f"unsupported provider {self.provider!r}"})
            return self._persist(base, None)

        image: Image.Image | None = None
        try:
            inventory_check = self._check_inventory_eligibility(env, task_proposal)
            base["inventory_eligibility"] = inventory_check
            if inventory_check is not None and not inventory_check.get("suitable", True):
                base.update({
                    "suitable": False,
                    "reason": inventory_check.get("reason", "target_not_taskable_in_inventory"),
                })
                return self._persist(base, None)

            image = self._capture_agentview(env)
            queries = self._target_queries(task_proposal, scene_context or {})
            base["queries"] = queries
            if image is None:
                raise RuntimeError("could not capture agentview image")
            if not queries:
                base.update({"suitable": self.fail_open_on_error, "reason": "no target query could be inferred"})
                return self._persist(base, image)

            detector = self._get_detector()
            # Only probe the primary target; this guard is a task-object
            # existence check, not a full scene inventory pass, and keeping it
            # to one Molmo request avoids turning rare mismatch guards into
            # multi-minute service waits.
            probe_queries = queries[:1]
            points = detector(image, probe_queries)
            serial_points: dict[str, list[int | None] | None] = {}
            for key, value in (points or {}).items():
                if value is None:
                    serial_points[str(key)] = None
                else:
                    x, y = value
                    serial_points[str(key)] = [x, y]
            base["points"] = serial_points

            width, height = image.size
            matched: list[str] = []
            for query, point in serial_points.items():
                if not isinstance(point, list) or len(point) != 2:
                    continue
                x, y = point
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    if 0 <= int(x) < width and 0 <= int(y) < height:
                        matched.append(query)
            primary_query = queries[0] if queries else ""
            primary_matched = primary_query in matched
            if primary_matched:
                base.update({
                    "suitable": True,
                    "reason": "molmo_found_target_in_agentview",
                    "matched_queries": matched,
                })
            else:
                base.update({
                    "suitable": False,
                    "reason": "molmo_returned_no_valid_point_for_primary_target",
                    "matched_queries": [],
                })
            return self._persist(base, image)
        except Exception as exc:  # noqa: BLE001 - verifier must not crash the loop by default.
            logger.warning(
                "MolmoSpaces environment verifier error (%s: %s)",
                type(exc).__name__, exc,
            )
            base.update({
                "suitable": self.fail_open_on_error,
                "reason": "verifier_error_fail_open" if self.fail_open_on_error else "verifier_error_fail_closed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            return self._persist(base, image)

    def _get_detector(self):
        if self._detector is None:
            from rats.integrations.vision.molmo import init_molmo

            self._detector = init_molmo(
                model_name=self.molmo_model,
                base_url=self.molmo_base_url,
                api_key=self.molmo_api_key,
            )
        return self._detector

    @staticmethod
    def _capture_agentview(env: Any) -> Image.Image | None:
        low_level = getattr(env, "low_level_env", env)
        render = getattr(low_level, "render", None) or getattr(env, "render", None)
        if not callable(render):
            return None
        try:
            try:
                arr = render(mode="rgb_array")
            except TypeError:
                arr = render()
        except Exception:
            return None
        if arr is None:
            return None
        if isinstance(arr, Image.Image):
            return arr.convert("RGB")
        np_arr = np.asarray(arr)
        if np_arr.ndim == 2:
            np_arr = np.stack([np_arr] * 3, axis=-1)
        if np_arr.ndim == 3 and np_arr.shape[-1] == 4:
            np_arr = np_arr[..., :3]
        if np_arr.dtype != np.uint8:
            if np_arr.max(initial=0) <= 1.0:
                np_arr = np_arr * 255.0
            np_arr = np.clip(np_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(np_arr).convert("RGB")

    @classmethod
    def _target_queries(
        cls,
        task_proposal: dict[str, Any],
        scene_context: dict[str, Any],
    ) -> list[str]:
        queries: list[str] = []

        play = task_proposal.get("_playtime") or {}
        for key in ("target_display_name", "target_internal_name"):
            value = str(play.get(key) or "").strip()
            if value:
                queries.append(value)

        for obj in task_proposal.get("objects") or []:
            text = cls._humanize_object_name(str(obj))
            if text:
                queries.append(text)

        # Benchmark descriptors sometimes expose object scope even when the
        # proposal object list was not preserved.
        scope = scene_context.get("object_scope") or {}
        if isinstance(scope, dict):
            for obj in list(scope.keys())[:3]:
                text = cls._humanize_object_name(str(obj))
                if text:
                    queries.append(text)

        if not queries:
            phrase = cls._object_phrase_from_language(
                str(
                    task_proposal.get("language")
                    or task_proposal.get("goal_conditions")
                    or scene_context.get("goal_conditions_nl")
                    or ""
                )
            )
            if phrase:
                queries.append(phrase)

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            query = re.sub(r"\s+", " ", query).strip(" .")
            if not query:
                continue
            key = query.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(query)
        return deduped[:5]

    @classmethod
    def _check_inventory_eligibility(
        cls,
        env: Any,
        task_proposal: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Reject proposals whose named targets are not taskable inventory rows.

        Main playtime targets must be in ``pickables`` or ``articulations``.
        Placement/support secondaries may additionally be in ``placeables`` or
        MolmoSpaces' current equivalent, ``receptacles``. If no live inventory
        API is available, return ``None`` and let the visibility check run as
        before.
        """
        inventory = cls._scene_inventory(env)
        if not inventory:
            return None
        sections = cls._inventory_sections(inventory)
        if not any(sections.values()):
            return None

        play = task_proposal.get("_playtime") or {}
        primary = str(play.get("target_internal_name") or "").strip()
        secondary = str(play.get("secondary_internal_name") or "").strip()
        if not primary:
            objects = [
                str(x).strip()
                for x in task_proposal.get("objects") or []
                if str(x).strip()
            ]
            primary = objects[0] if objects else ""
            secondary = objects[1] if len(objects) > 1 else ""

        primary_allowed = sections["pickables"] | sections["articulations"]
        secondary_allowed = primary_allowed | sections["placeables"] | sections["receptacles"]
        checks: list[dict[str, Any]] = []
        if primary:
            ok = (
                primary in primary_allowed
                or cls._root_object_name(primary) in primary_allowed
            )
            checks.append({
                "role": "target",
                "internal_name": primary,
                "allowed_sections": ["pickables", "articulations"],
                "suitable": ok,
            })
            if not ok:
                return {
                    "suitable": False,
                    "reason": "primary_target_not_pickable_or_articulated",
                    "checks": checks,
                    "inventory_counts": {
                        key: len(value) for key, value in sections.items()
                    },
                }
        if secondary:
            ok = (
                secondary in secondary_allowed
                or cls._root_object_name(secondary) in secondary_allowed
            )
            checks.append({
                "role": "secondary",
                "internal_name": secondary,
                "allowed_sections": [
                    "pickables", "articulations", "placeables", "receptacles",
                ],
                "suitable": ok,
            })
            if not ok:
                return {
                    "suitable": False,
                    "reason": "secondary_target_not_placeable_or_taskable",
                    "checks": checks,
                    "inventory_counts": {
                        key: len(value) for key, value in sections.items()
                    },
                }
        return {
            "suitable": True,
            "reason": "inventory_targets_are_taskable",
            "checks": checks,
            "inventory_counts": {
                key: len(value) for key, value in sections.items()
            },
        }

    @staticmethod
    def _scene_inventory(env: Any) -> dict[str, Any] | None:
        low_level = getattr(env, "low_level_env", env)
        fn = getattr(low_level, "describe_scene_inventory", None) or getattr(
            env, "describe_scene_inventory", None
        )
        if not callable(fn):
            return None
        try:
            inv = fn() or {}
        except Exception:
            return None
        return dict(inv) if isinstance(inv, dict) else None

    @classmethod
    def _inventory_sections(cls, inventory: dict[str, Any]) -> dict[str, set[str]]:
        out = {
            "pickables": set(),
            "articulations": set(),
            "placeables": set(),
            "receptacles": set(),
        }
        for section in out:
            for entry in inventory.get(section, []) or []:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("internal_name") or "").strip()
                if not name:
                    continue
                out[section].add(name)
                out[section].add(cls._root_object_name(name))
        return out

    @staticmethod
    def _root_object_name(name: str) -> str:
        parts = str(name or "").split("_")
        if len(parts) >= 5 and all(part.isdigit() for part in parts[-4:]):
            return "_".join(parts[:-4])
        return str(name or "")

    @staticmethod
    def _humanize_object_name(name: str) -> str:
        text = name.split("/")[-1]
        text = re.sub(r"[0-9a-fA-F]{24,}", "", text)
        text = re.sub(r"_\d+(?:_\d+)*$", "", text)
        text = text.replace("_", " ").replace("-", " ")
        text = re.sub(r"\b[0-9a-fA-F]{6,}\b", "", text)
        text = re.sub(r"\b\d+\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    @staticmethod
    def _object_phrase_from_language(language: str) -> str:
        text = language.strip()
        if not text:
            return ""
        match = re.search(
            r"\b(?:open|close|pick up|pick|place|move|pull|push|lift)\s+(?:the\s+|a\s+|an\s+)?([^.,;]+)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            phrase = match.group(1)
            phrase = re.split(r"\b(?:and|onto|into|on|in|to)\b", phrase, maxsplit=1, flags=re.IGNORECASE)[0]
            return phrase.strip()
        return text[:80]

    def _persist(self, result: dict[str, Any], image: Image.Image | None) -> dict[str, Any]:
        if self.output_dir is None:
            return result
        root = self.output_dir / "environment_verifier"
        root.mkdir(parents=True, exist_ok=True)
        stamp = f"iter{int(result.get('iteration', 0)):03d}_attempt{int(result.get('attempt', 0)):02d}_{int(time.time() * 1000)}"
        if image is not None:
            image_path = root / f"{stamp}_agentview.png"
            try:
                image.save(image_path)
                result.setdefault("artifacts", {})["agentview_image"] = str(image_path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("saving environment verifier image failed: %s", exc)
        json_path = root / f"{stamp}.json"
        try:
            json_path.write_text(json.dumps(result, indent=2, default=str))
            result.setdefault("artifacts", {})["json"] = str(json_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("saving environment verifier json failed: %s", exc)
        return result
