"""Environment Creator: generates and instantiates novel task environments.

For LIBERO: generates BDDL files from high-level task specs, then creates
MuJoCo environments. For MolmoSpaces: emits a validated JSON task artifact over
the bridge-reported catalog, then recreates a runnable bridge-backed env.
For BEHAVIOR: validates task feasibility before rebinding.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from rats.agents.base_agent import query_llm_text
from rats.agents.libero_catalog import (
    FIXTURES,
    OBJECTS,
    PREDICATES,
    PROBLEM_CLASSES,
    build_catalog_text,
)

logger = logging.getLogger("rats.environment_creator")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BDDL_OUTPUT_DIR = _PROJECT_ROOT / "generated_bddl"
_WORKSPACE_TYPES = {
    "table",
    "kitchen_table",
    "living_room_table",
    "study_table",
    "coffee_table",
    "floor",
}
_LANGUAGE_ASSET_EQUIVALENTS = {
    # LIBERO commonly uses akita_black_bowl for the natural-language
    # phrase "black bowl"; do not reject those scenes on language alone.
    "black_bowl": {
        "black_bowl",
        "akita_black_bowl",
        "red_akita_black_bowl",
        "bigger_akita_black_bowl",
    },
    "porcelain_mug": {
        "porcelain_mug",
        "white_porcelain_mug",
    },
}


class EnvironmentCreator:
    """Creates task environments from high-level proposals.

    Two modes:
    - **novel** (default for LIBERO): LLM generates BDDL, env instantiated from it.
    - **catalog**: select from predefined tasks (fallback / BEHAVIOR mode).
    """

    def __init__(
        self,
        env_type: str = "libero",
        bddl_output_dir: str | Path | None = None,
    ) -> None:
        self.env_type = env_type
        self.bddl_dir = Path(bddl_output_dir) if bddl_output_dir else _BDDL_OUTPUT_DIR
        self.bddl_dir.mkdir(parents=True, exist_ok=True)
        self.molmospaces_spec_dir = self.bddl_dir.parent / "generated_molmospaces_specs"
        self.molmospaces_spec_dir.mkdir(parents=True, exist_ok=True)
        self._prompt_template = self._load_prompt()
        # Wrap libero scene-runtime checks in the unified verifier surface
        # (same shape as MolmoEnvironmentVerifier — see agents/environment_verifier.py).
        # Import lives here to avoid a top-level circular import.
        from rats.agents.environment_verifier import LiberoEnvironmentVerifier
        self._libero_env_verifier = LiberoEnvironmentVerifier(self)

    def _load_prompt(self) -> str:
        path = _PROJECT_ROOT / "rats" / "prompts" / "environment_creator.txt"
        if path.exists():
            return path.read_text()
        return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_from_proposal(
        self,
        task_proposal: dict[str, Any],
        old_env: Any | None = None,
    ) -> dict[str, Any]:
        """Create an environment from a task proposal.

        Dispatches to LIBERO (novel BDDL generation), MolmoSpaces (validated
        JSON artifact generation), or BEHAVIOR (template injection + rebind)
        based on self.env_type.

        Returns:
            Dict with:
                - env: the new environment (CodeExecutionEnvBase)
                - scene_context: extracted scene context dict
                - activity_name: activity name string
                - bddl_path (LIBERO only): path to generated BDDL file
                - spec_path (MolmoSpaces only): path to generated JSON artifact
        """
        if self.env_type == "molmospaces":
            return self._create_molmospaces_from_proposal(task_proposal, old_env)
        return self._create_libero_from_proposal(task_proposal, old_env)


    # ------------------------------------------------------------------
    # MolmoSpaces: generate bounded task artifact over existing catalog
    # ------------------------------------------------------------------

    def _create_molmospaces_from_proposal(
        self,
        task_proposal: dict[str, Any],
        old_env: Any | None = None,
    ) -> dict[str, Any]:
        """Generate a validated MolmoSpaces task artifact and runnable env."""
        if old_env is None:
            raise RuntimeError("MolmoSpaces task creation requires an existing env")

        from rats.loop.molmospaces_utils import (
            extract_molmospaces_scene_context,
            recreate_molmospaces_env,
        )

        low_level = getattr(old_env, "low_level_env", old_env)
        catalog = low_level.list_task_descriptors()
        current_descriptor = low_level.get_task_descriptor()
        selected_descriptor = self._select_molmospaces_descriptor(
            task_proposal,
            catalog,
            current_descriptor=current_descriptor,
        )
        spec = self._build_molmospaces_spec(
            task_proposal,
            selected_descriptor,
            current_descriptor=current_descriptor,
        )
        spec_path = self._write_molmospaces_spec(spec)
        env = recreate_molmospaces_env(old_env, spec["canonical_task_id"])
        scene_context = extract_molmospaces_scene_context(env)
        scene_verification = self._verify_molmospaces_scene_runtime(spec, env)
        if scene_verification["errors"]:
            raise RuntimeError(
                "MolmoSpaces scene verification failed: "
                + "; ".join(scene_verification["errors"][:6])
            )
        scene_context["task_prompt"] = spec["language_goal"]
        scene_context["goal_conditions_nl"] = spec["language_goal"]
        scene_context["generated_spec_path"] = str(spec_path)
        scene_context["scene_verification"] = scene_verification

        return {
            "env": env,
            "scene_context": scene_context,
            "activity_name": spec["canonical_task_id"],
            "spec_path": str(spec_path),
            "generated_spec": spec,
            "scene_verification": scene_verification,
        }

    def _select_molmospaces_descriptor(
        self,
        task_proposal: dict[str, Any],
        catalog: list[dict[str, Any]],
        *,
        current_descriptor: dict[str, Any],
    ) -> dict[str, Any]:
        """Pick a bounded MolmoSpaces descriptor from the authoritative catalog."""
        # 1. Try exact canonical_task_id match (from activity_name or explicit field).
        for key in ("canonical_task_id", "activity_name"):
            requested_task_id = task_proposal.get(key, "")
            if isinstance(requested_task_id, str) and requested_task_id.startswith("molmospaces:"):
                for descriptor in catalog:
                    if descriptor["canonical_id"] == requested_task_id:
                        return descriptor

        # 2. Structured field matching.
        benchmark = task_proposal.get("benchmark") or current_descriptor.get("benchmark")
        scene_family = (
            task_proposal.get("scene_family")
            or task_proposal.get("scene_model")
        )
        task_family = task_proposal.get("task_family")
        variant = task_proposal.get("variant")
        requested_objects = set(task_proposal.get("objects", []))

        candidates = []
        for descriptor in catalog:
            if benchmark and descriptor.get("benchmark") != benchmark:
                continue
            if scene_family and descriptor.get("scene_family") != scene_family:
                continue
            if task_family and descriptor.get("task_family") != task_family:
                continue
            if variant and descriptor.get("variant") != variant:
                continue
            candidates.append(descriptor)

        if not candidates and requested_objects:
            candidates = [
                descriptor
                for descriptor in catalog
                if requested_objects.issubset(set(descriptor.get("objects", [])))
            ]

        # 3. Language-based fallback: substring match on task description.
        if not candidates:
            proposal_language = (task_proposal.get("language") or "").lower()
            if proposal_language:
                for descriptor in catalog:
                    desc_language = (descriptor.get("language") or "").lower()
                    if proposal_language in desc_language or desc_language in proposal_language:
                        candidates.append(descriptor)

        if not candidates:
            return current_descriptor

        # Prefer descriptors that cover more requested objects.
        if requested_objects:
            candidates.sort(
                key=lambda descriptor: (
                    len(requested_objects.intersection(set(descriptor.get("objects", [])))),
                    descriptor["canonical_id"],
                ),
                reverse=True,
            )
        return candidates[0]

    def _build_molmospaces_spec(
        self,
        task_proposal: dict[str, Any],
        descriptor: dict[str, Any],
        *,
        current_descriptor: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a phase-1 MolmoSpaces JSON artifact."""
        language_goal = task_proposal.get("language") or descriptor.get("language") or current_descriptor.get("language", "")
        meta = descriptor.get("metadata", {})
        generator_metadata = {
            "source": "EnvironmentCreator",
            "mode": task_proposal.get("mode", "novel"),
            "reasoning": task_proposal.get("reasoning", ""),
            "proposal_activity_name": task_proposal.get("activity_name", ""),
            "current_task_id": current_descriptor.get("canonical_id", ""),
        }
        spec: dict[str, Any] = {
            "artifact_version": "1.0",
            "env_type": "molmospaces",
            "format": "json",
            "canonical_task_id": descriptor["canonical_id"],
            "source_catalog_version": meta.get("catalog_source", "builtin"),
            "benchmark_or_catalog": descriptor.get("benchmark"),
            "scene_family": descriptor.get("scene_family"),
            "task_family": descriptor.get("task_family"),
            "variant": descriptor.get("variant"),
            "language_goal": language_goal,
            "allowed_assets": descriptor.get("objects", []),
            "allowed_scene_components": meta.get("fixtures", []),
            "constraints": task_proposal.get("goal", []),
            "privileged_api_requirements": descriptor.get("privileged_requirements", []),
            "initial_state_or_seed_policy": "reuse_bridge_catalog_descriptor",
            "generator_metadata": generator_metadata,
            "validation_status": "validated",
        }
        # Include episode-level metadata when available (benchmark-backed catalog).
        if meta.get("catalog_source") == "benchmark":
            spec["house_index"] = meta.get("house_index")
            spec["task_cls"] = meta.get("task_cls")
            spec["robot_name"] = meta.get("robot_name")
            spec["episode_index"] = meta.get("episode_index")
        return spec

    def _write_molmospaces_spec(self, spec: dict[str, Any]) -> Path:
        """Persist MolmoSpaces JSON artifact under the generated-specs directory."""
        task_id = spec["canonical_task_id"]
        slug = re.sub(r"[^a-z0-9]+", "_", task_id.lower()).strip("_")
        digest = hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
        path = self.molmospaces_spec_dir / f"{slug}_{digest}.json"
        path.write_text(json.dumps(spec, indent=2))
        logger.info("Generated MolmoSpaces task spec: %s", path)
        return path

    def _verify_molmospaces_scene_runtime(
        self,
        spec: dict[str, Any],
        env: Any,
    ) -> dict[str, Any]:
        """Verify that the bridge recreated the exact MolmoSpaces task spec.

        MolmoSpaces task creation is descriptor/catalog-based, not free-form
        scene synthesis. The important failure mode is therefore a mismatch
        between the spec we asked the bridge to load and the descriptor that
        the recreated env actually reports (often after a reconnect or remote
        server restart). Treat those mismatches as environment creation errors,
        not policy failures.
        """
        result: dict[str, Any] = {
            "ok": False,
            "errors": [],
            "warnings": [],
            "details": {
                "spec": {
                    key: spec.get(key)
                    for key in (
                        "canonical_task_id",
                        "benchmark_or_catalog",
                        "scene_family",
                        "task_family",
                        "variant",
                        "allowed_assets",
                        "allowed_scene_components",
                        "house_index",
                        "task_cls",
                        "robot_name",
                        "episode_index",
                    )
                    if key in spec
                },
                "descriptor": {},
            },
        }

        low_level = getattr(env, "low_level_env", env)
        get_descriptor = getattr(low_level, "get_task_descriptor", None)
        if not callable(get_descriptor):
            result["errors"].append("MolmoSpaces env does not expose get_task_descriptor()")
            return result

        try:
            descriptor = dict(get_descriptor() or {})
        except Exception as exc:
            result["errors"].append(f"MolmoSpaces get_task_descriptor() failed: {exc}")
            return result

        result["details"]["descriptor"] = {
            key: descriptor.get(key)
            for key in (
                "canonical_id",
                "benchmark",
                "scene_family",
                "task_family",
                "variant",
                "objects",
                "metadata",
            )
            if key in descriptor
        }

        def _same(label: str, expected: Any, actual: Any) -> None:
            if expected in (None, "", []):
                return
            if str(expected) != str(actual):
                result["errors"].append(
                    f"MolmoSpaces descriptor mismatch for {label}: "
                    f"expected {expected!r}, got {actual!r}"
                )

        _same("canonical_task_id", spec.get("canonical_task_id"), descriptor.get("canonical_id"))
        _same("benchmark", spec.get("benchmark_or_catalog"), descriptor.get("benchmark"))
        _same("scene_family", spec.get("scene_family"), descriptor.get("scene_family"))
        _same("task_family", spec.get("task_family"), descriptor.get("task_family"))
        _same("variant", spec.get("variant"), descriptor.get("variant"))

        spec_assets = {str(item) for item in (spec.get("allowed_assets") or [])}
        desc_assets = {str(item) for item in (descriptor.get("objects") or [])}
        missing_assets = sorted(spec_assets - desc_assets)
        if missing_assets:
            result["errors"].append(
                "MolmoSpaces descriptor is missing requested asset(s): "
                f"{missing_assets}; descriptor objects={sorted(desc_assets)}"
            )

        metadata = descriptor.get("metadata") or {}
        for key in ("house_index", "task_cls", "robot_name", "episode_index"):
            if key in spec and spec.get(key) is not None:
                actual = metadata.get(key, descriptor.get(key))
                _same(key, spec.get(key), actual)

        get_info = getattr(low_level, "get_task_info", None)
        if callable(get_info):
            try:
                info = get_info() or {}
                result["details"]["task_info_keys"] = sorted(map(str, info.keys()))
            except Exception as exc:
                result["warnings"].append(f"MolmoSpaces get_task_info() failed: {exc}")
        else:
            result["warnings"].append("MolmoSpaces env does not expose get_task_info()")

        result["ok"] = not result["errors"]
        if result["errors"]:
            logger.warning("MolmoSpaces scene verification failed: %s", result["errors"])
        elif result["warnings"]:
            logger.info("MolmoSpaces scene verification warnings: %s", result["warnings"])
        return result

    # ------------------------------------------------------------------
    # LIBERO: generate novel BDDL and instantiate new env
    # ------------------------------------------------------------------

    def _create_libero_from_proposal(
        self,
        task_proposal: dict[str, Any],
        old_env: Any | None = None,
    ) -> dict[str, Any]:
        """Generate BDDL and create a new LIBERO env."""
        # 1. Generate BDDL
        bddl_text = self._generate_bddl(task_proposal)
        if not bddl_text:
            raise RuntimeError("Failed to generate BDDL from proposal")

        # 2. Validate BDDL. The proposal-consistency pass catches cases
        # where syntactically valid BDDL silently swaps the requested asset
        # (for example black_book -> orange_juice) or omits the workspace.
        errors = self._validate_bddl(bddl_text)
        errors.extend(self._validate_bddl_scene_semantics(bddl_text, task_proposal))
        if errors:
            logger.warning(f"BDDL validation issues: {errors}")
            bddl_text = self._fix_bddl(bddl_text, errors, task_proposal)
            errors = self._validate_bddl(bddl_text)
            errors.extend(self._validate_bddl_scene_semantics(bddl_text, task_proposal))
            if errors:
                logger.error(f"BDDL still invalid after fix: {errors}")
                raise RuntimeError(f"Invalid BDDL: {errors}")

        # 3. Write BDDL to disk
        language = task_proposal.get("language", "novel_task")
        slug = re.sub(r"[^a-z0-9]+", "_", language.lower()).strip("_")[:60]
        bddl_hash = hashlib.md5(bddl_text.encode()).hexdigest()[:8]
        bddl_filename = f"{slug}_{bddl_hash}.bddl"
        bddl_path = self.bddl_dir / bddl_filename
        bddl_path.write_text(bddl_text)
        logger.info(f"Generated BDDL: {bddl_path}")

        # 4. Instantiate environment
        env = self._instantiate_libero_env(bddl_path, old_env)

        # 5. Verify the realized scene before exposing it to the loop.
        # This is intentionally privileged: bad scene generation should be
        # rejected here rather than diagnosed as a policy failure later.
        # Routed through ``LiberoEnvironmentVerifier`` so the call surface
        # matches the molmospaces post-rebind verifier (both implement
        # ``EnvironmentVerifier.verify`` with a unified return shape). The
        # check logic still lives on this class because it depends on
        # ``_parse_bddl_scene_spec`` / MuJoCo helpers other paths also use.
        scene_verification = self._libero_env_verifier.verify(
            env, task_proposal, bddl_text=bddl_text,
        )
        if scene_verification["errors"]:
            raise RuntimeError(
                "Scene verification failed: "
                + "; ".join(scene_verification["errors"][:6])
            )

        # 6. Extract scene context
        from rats.loop.libero_utils import extract_libero_scene_context
        scene_context = extract_libero_scene_context(env)
        scene_context["goal_conditions_nl"] = language
        scene_context["task_prompt"] = language
        scene_context["activity_name"] = slug
        scene_context["scene_verification"] = scene_verification

        # 7. Generate custom verifier (relaxed predicate check)
        custom_verifier_code = ""
        try:
            custom_verifier_code = self._generate_custom_verifier(
                task_proposal, bddl_text,
            )
        except Exception as e:
            logger.warning(f"Custom verifier generation failed (non-fatal): {e}")

        return {
            "env": env,
            "bddl_path": str(bddl_path),
            "scene_context": scene_context,
            "activity_name": slug,
            "custom_verifier_code": custom_verifier_code,
            "scene_verification": scene_verification,
        }

    # ------------------------------------------------------------------
    # BDDL generation
    # ------------------------------------------------------------------

    def _generate_bddl(self, task_proposal: dict[str, Any]) -> str:
        """Use LLM to generate a BDDL file from a high-level task proposal."""
        catalog_text = build_catalog_text()

        # Format the proposal for the prompt
        proposal_parts = []
        proposal_parts.append(f"Language goal: {task_proposal.get('language', 'unknown')}")
        if task_proposal.get("scene_type"):
            proposal_parts.append(f"Scene type: {task_proposal['scene_type']}")
        if task_proposal.get("objects"):
            proposal_parts.append(f"Objects to use: {', '.join(task_proposal['objects'])}")
        if task_proposal.get("fixtures"):
            proposal_parts.append(f"Fixtures to use: {', '.join(task_proposal['fixtures'])}")
        if task_proposal.get("goal"):
            proposal_parts.append(f"Goal predicates: {task_proposal['goal']}")
        if task_proposal.get("reasoning"):
            proposal_parts.append(f"Reasoning: {task_proposal['reasoning']}")

        proposal_text = "\n".join(proposal_parts)

        user_prompt = self._prompt_template.replace(
            "{catalog}", catalog_text
        ).replace(
            "{task_proposal}", proposal_text
        )

        system_prompt = (
            "You are a BDDL specification generator. Output ONLY the BDDL inside "
            "a code fence. The BDDL must be syntactically valid and use only assets "
            "from the catalog."
        )

        response = query_llm_text(system_prompt, user_prompt)
        return self._extract_bddl_from_response(response)

    def _extract_bddl_from_response(self, response: str) -> str:
        """Extract BDDL text from LLM response (may be in code fence)."""
        # Try code fence first
        match = re.search(r"```(?:bddl|lisp|scheme)?\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try to find (define ...) block
        match = re.search(r"\(define\s.*\)\s*$", response, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(0).strip()
        # Last resort: return everything after stripping obvious non-BDDL
        lines = [l for l in response.splitlines() if not l.startswith("```")]
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Custom verifier generation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_goal_predicates(bddl_text: str) -> list[tuple[str, str, str]]:
        """Extract simple binary goal predicates from a BDDL :goal block."""
        goal_match = re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", bddl_text, re.DOTALL)
        goal_text = goal_match.group(1) if goal_match else bddl_text
        predicates: list[tuple[str, str, str]] = []
        for pred, obj, target in re.findall(
            r"\((On|In)\s+([A-Za-z_][A-Za-z0-9_]*_\d+)\s+([A-Za-z_][A-Za-z0-9_]*_\d+)\)",
            goal_text,
        ):
            predicates.append((pred, obj, target))
        return predicates

    @staticmethod
    def _generate_structured_custom_verifier(
        predicates: list[tuple[str, str, str]],
    ) -> str:
        """Generate a deterministic XY-containment verifier for simple On/In goals."""
        if not predicates:
            return ""
        checks = repr(predicates)
        return f'''def custom_verify(sim_env):
    try:
        import numpy as np

        goal_checks = {checks}
        details = []
        all_ok = True

        def _obj(name):
            return sim_env.get_object(name)

        def _body_id(name):
            return sim_env.obj_body_id[name]

        def _pos(name):
            return np.array(sim_env.sim.data.body_xpos[_body_id(name)], dtype=float)

        def _target_xy_bounds(name):
            center = _pos(name)[:2]
            fallback_half_extent = np.array([0.10, 0.10], dtype=float)
            mins = []
            maxs = []

            try:
                body_id = _body_id(name)
                model = sim_env.sim.model
                data = sim_env.sim.data
                geom_bodyid = getattr(model, "geom_bodyid", None)
                geom_size = getattr(model, "geom_size", None)
                geom_group = getattr(model, "geom_group", None)
                geom_contype = getattr(model, "geom_contype", None)
                geom_conaffinity = getattr(model, "geom_conaffinity", None)
                body_parentid = getattr(model, "body_parentid", None)
                geom_xpos = getattr(data, "geom_xpos", None)
                geom_xmat = getattr(data, "geom_xmat", None)
                if geom_bodyid is None or geom_size is None or geom_xpos is None:
                    raise AttributeError("geom data unavailable")
                target_body_ids = set()
                try:
                    root_body_id = int(body_id)
                    target_body_ids.add(root_body_id)
                    if body_parentid is not None:
                        for candidate in range(len(body_parentid)):
                            current = int(candidate)
                            seen = set()
                            while current not in seen:
                                seen.add(current)
                                if current == root_body_id:
                                    target_body_ids.add(int(candidate))
                                    break
                                parent = int(body_parentid[current])
                                if parent == current or parent < 0:
                                    break
                                current = parent
                except Exception:
                    target_body_ids = set()

                def _geom_matches(index):
                    try:
                        if int(geom_bodyid[index]) in target_body_ids:
                            return True
                    except Exception:
                        pass
                    try:
                        body_id2name = getattr(model, "body_id2name", None)
                        body_name = body_id2name(int(geom_bodyid[index])) if callable(body_id2name) else ""
                        if body_name and (body_name.startswith(name) or name in body_name):
                            return True
                    except Exception:
                        pass
                    try:
                        geom_id2name = getattr(model, "geom_id2name", None)
                        geom_name = geom_id2name(index) if callable(geom_id2name) else ""
                        return bool(geom_name and (geom_name.startswith(name) or name in geom_name))
                    except Exception:
                        return False

                for index in range(len(geom_bodyid)):
                    if not _geom_matches(index):
                        continue
                    try:
                        if geom_group is not None and int(geom_group[index]) == 1:
                            continue
                    except Exception:
                        pass
                    try:
                        if (
                            geom_contype is not None
                            and geom_conaffinity is not None
                            and int(geom_contype[index]) == 0
                            and int(geom_conaffinity[index]) == 0
                        ):
                            continue
                    except Exception:
                        pass
                    geom_center = np.asarray(geom_xpos[index], dtype=float)[:2]
                    raw_size = np.asarray(geom_size[index], dtype=float).reshape(-1)
                    if raw_size.size == 0:
                        continue
                    if raw_size.size == 1:
                        half_local = np.array([raw_size[0], raw_size[0], raw_size[0]], dtype=float)
                    elif raw_size.size == 2:
                        half_local = np.array([raw_size[0], raw_size[1], 0.0], dtype=float)
                    else:
                        half_local = np.abs(raw_size[:3])
                    if geom_xmat is not None:
                        try:
                            mat = np.asarray(geom_xmat[index], dtype=float).reshape(3, 3)
                            half_xy = np.abs(mat[:2, :]) @ half_local
                        except Exception:
                            half_xy = half_local[:2]
                    else:
                        half_xy = half_local[:2]
                    mins.append(geom_center - half_xy)
                    maxs.append(geom_center + half_xy)
            except Exception:
                pass

            if not mins:
                return center - fallback_half_extent, center + fallback_half_extent

            return np.min(np.stack(mins), axis=0), np.max(np.stack(maxs), axis=0)

        for pred, obj_name, target_name in goal_checks:
            obj_pos = _pos(obj_name)
            target_min_xy, target_max_xy = _target_xy_bounds(target_name)
            try:
                contact = bool(sim_env.check_contact(_obj(obj_name), _obj(target_name)))
            except Exception:
                contact = False

            xy_inside = bool(
                np.all(obj_pos[:2] >= target_min_xy)
                and np.all(obj_pos[:2] <= target_max_xy)
            )

            if pred in ("On", "In"):
                ok = xy_inside
            else:
                ok = False

            all_ok = all_ok and ok
            details.append(
                f"{{pred}} {{obj_name}} {{target_name}}: "
                f"obj_xy={{obj_pos[:2].tolist()}}, "
                f"target_xy_min={{target_min_xy.tolist()}}, target_xy_max={{target_max_xy.tolist()}}, "
                f"xy_inside={{xy_inside}}, contact={{contact}}, ok={{ok}}"
            )

        return {{"success": bool(all_ok), "details": "; ".join(details)}}
    except Exception as e:
        return {{"success": False, "details": str(e)}}
'''

    def _generate_custom_verifier(
        self, task_proposal: dict[str, Any], bddl_text: str,
    ) -> str:
        """Generate the deterministic structured verifier for novel BDDL tasks.

        LIBERO's built-in predicates can be too strict for dynamically
        generated tasks. This function only emits deterministic XY containment
        checks for simple On/In goals. Visual fallback verification happens at
        execution time in ``agents.verifier.Verifier`` using the final frame.
        """
        _ = task_proposal  # kept in the signature for caller compatibility
        structured = self._generate_structured_custom_verifier(
            self._extract_goal_predicates(bddl_text),
        )
        if structured:
            logger.info(f"Generated structured custom verifier ({len(structured)} chars)")
            return structured

        logger.info("No structured custom verifier generated for this BDDL")
        return ""

    # ------------------------------------------------------------------
    # Scene verification
    # ------------------------------------------------------------------

    def _validate_bddl_scene_semantics(
        self, bddl_text: str, task_proposal: dict[str, Any],
    ) -> list[str]:
        """Validate that generated BDDL actually describes the proposal."""
        errors: list[str] = []
        spec = self._parse_bddl_scene_spec(bddl_text)

        problem_class = spec.get("problem_class", "")
        problem_info = PROBLEM_CLASSES.get(problem_class)
        fixtures: dict[str, str] = spec["fixtures"]
        objects: dict[str, str] = spec["objects"]

        if problem_info:
            workspace = problem_info["workspace"]
            workspace_type = problem_info["workspace_type"]
            actual_workspace_type = fixtures.get(workspace)
            if actual_workspace_type != workspace_type:
                errors.append(
                    "BDDL fixtures must declare workspace "
                    f"'{workspace} - {workspace_type}' for {problem_class}; "
                    f"found '{workspace} - {actual_workspace_type}'"
                )

        if not fixtures:
            errors.append(
                "BDDL declares no fixtures/workspace, so no scene/table will be generated"
            )
        # Only require movable objects when the proposal asked for any.
        # Pure articulation tasks (open/close drawer, turn on stove,
        # open microwave) legitimately have proposal.objects=[] and the
        # BDDL correctly emits (:objects) empty. The OLD unconditional
        # check fired "no movable objects" on these tasks; auto-fix
        # then demoted an articulated fixture (yellow_cabinet,
        # short_fridge) from (:fixtures) into (:objects) to silence the
        # error, which broke the subsequent proposal-consistency check
        # ("required fixture not declared"), cascaded to env_creation
        # failure → fallback → fake ✅ inherited from previous iter's
        # sim state. Observed in libero_main_30iter iter 26-28.
        proposal_objects = task_proposal.get("objects") or []
        if not objects and proposal_objects:
            errors.append(
                f"BDDL declares no movable objects but proposal expected "
                f"{sorted({str(o) for o in proposal_objects})}"
            )

        declared_object_types = set(objects.values())
        declared_fixture_types = set(fixtures.values()) - _WORKSPACE_TYPES
        required_objects, language_objects = self._required_bddl_assets(
            task_proposal, OBJECTS, field_name="objects",
        )
        required_fixtures, language_fixtures = self._required_bddl_assets(
            task_proposal, FIXTURES, field_name="fixtures",
        )

        missing_objects = sorted(required_objects - declared_object_types)
        missing_language_objects = sorted(
            obj for obj in language_objects
            if not self._language_asset_satisfied(obj, declared_object_types)
        )
        if missing_objects:
            errors.append(
                "BDDL object types do not include required proposal object(s): "
                f"{missing_objects}; declared objects: {sorted(declared_object_types)}"
            )
        if missing_language_objects:
            errors.append(
                "BDDL object types conflict with the language goal. Missing "
                f"{missing_language_objects}; declared objects: {sorted(declared_object_types)}"
            )

        missing_fixtures = sorted(required_fixtures - declared_fixture_types)
        missing_language_fixtures = sorted(language_fixtures - declared_fixture_types)
        if missing_fixtures:
            errors.append(
                "BDDL fixture types do not include required proposal fixture(s): "
                f"{missing_fixtures}; declared fixtures: {sorted(declared_fixture_types)}"
            )
        if missing_language_fixtures:
            errors.append(
                "BDDL fixture types conflict with the language goal. Missing "
                f"{missing_language_fixtures}; declared fixtures: {sorted(declared_fixture_types)}"
            )

        errors.extend(self._validate_bddl_references(spec))
        errors.extend(self._validate_bddl_goal_predicate_runtime_compatibility(spec))
        return errors

    def _validate_bddl_goal_predicate_runtime_compatibility(
        self, spec: dict[str, Any],
    ) -> list[str]:
        """Reject BDDL goals that LIBERO's predicate evaluator cannot execute.

        LIBERO's In/Stack predicates call check_contain() on the second
        argument.  A plain movable object state then calls object.in_box(...)
        (base_object_states.py:68), but regular objects such as WhiteBowl,
        WoodenTray, Plate, and Basket do not implement in_box.  In/Stack must
        therefore target a site/region such as wooden_cabinet_1_top_region or
        basket_1_contain_region, not a bare movable object instance.
        """
        errors: list[str] = []
        objects: dict[str, str] = spec["objects"]
        for predicate, args in self._iter_bddl_predicates(spec.get("goal_text", "")):
            pred = predicate.lower()
            if pred not in {"in", "stack"} or len(args) < 2:
                continue
            target = args[1]
            target_type = objects.get(target)
            if target_type is None:
                continue
            errors.append(
                f"BDDL goal predicate '{predicate}' targets movable object "
                f"'{target} - {target_type}'. LIBERO's {predicate} evaluator "
                "requires the second argument to be a site/region with in_box(); "
                "bare movable objects like bowls, trays, plates, and baskets "
                "raise AttributeError at runtime. Use a supported region "
                "(for example basket_1_contain_region or a fixture region), "
                "or use On for open bowl/tray/plate destinations."
            )
        return errors

    def _verify_libero_scene_runtime(
        self, env: Any, bddl_text: str, task_proposal: dict[str, Any],
    ) -> dict[str, Any]:
        """Reset/render the realized LIBERO scene and reject bad initial states."""
        spec = self._parse_bddl_scene_spec(bddl_text)
        result: dict[str, Any] = {
            "ok": False,
            "errors": [],
            "warnings": [],
            "details": {
                "problem_class": spec.get("problem_class", ""),
                "proposal_language": task_proposal.get("language", ""),
                "objects": spec["objects"],
                "fixtures": spec["fixtures"],
                "checked_seeds": [],
            },
        }

        for seed in (0, 1):
            result["details"]["checked_seeds"].append(seed)
            try:
                try:
                    env.reset(seed=seed)
                except TypeError:
                    env.reset()
            except Exception as exc:
                result["errors"].append(f"LIBERO scene reset failed for seed {seed}: {exc}")
                continue

            self._verify_libero_render(env, result, seed=seed)
            sim = self._get_libero_sim(env)
            if sim is None:
                result["errors"].append("LIBERO scene has no accessible MuJoCo sim")
                continue

            self._verify_libero_declared_instances(sim, env, spec, result, seed=seed)
            self._verify_libero_goal_predicates_executable(env, result, seed=seed)
            self._verify_libero_initial_contacts(sim, result, seed=seed)

        result["ok"] = not result["errors"]
        if result["errors"]:
            logger.warning("LIBERO scene verification failed: %s", result["errors"])
        elif result["warnings"]:
            logger.info("LIBERO scene verification warnings: %s", result["warnings"])
        return result

    def _verify_libero_render(
        self, env: Any, result: dict[str, Any], *, seed: int,
    ) -> None:
        try:
            try:
                frame = env.render(mode="rgb_array")
            except TypeError:
                frame = env.render()
        except Exception as exc:
            result["errors"].append(f"LIBERO scene render failed for seed {seed}: {exc}")
            return

        if frame is None:
            result["errors"].append(f"LIBERO scene render returned None for seed {seed}")
            return

        try:
            import numpy as np

            arr = np.asarray(frame)
            result["details"][f"render_shape_seed_{seed}"] = list(arr.shape)
            if arr.size == 0 or arr.ndim < 2:
                result["errors"].append(
                    f"LIBERO scene render is empty for seed {seed}: shape={arr.shape}"
                )
                return
            if arr.ndim >= 3 and arr.shape[-1] < 3:
                result["errors"].append(
                    f"LIBERO scene render has no RGB channels for seed {seed}: shape={arr.shape}"
                )
            if not np.isfinite(arr).all():
                result["errors"].append(
                    f"LIBERO scene render has non-finite pixels for seed {seed}"
                )
            if float(np.max(arr)) == float(np.min(arr)):
                result["errors"].append(
                    f"LIBERO scene render is constant-valued for seed {seed}; video is likely blank"
                )
        except Exception as exc:
            result["errors"].append(f"LIBERO scene render validation failed for seed {seed}: {exc}")

    def _verify_libero_declared_instances(
        self,
        sim: Any,
        env: Any,
        spec: dict[str, Any],
        result: dict[str, Any],
        *,
        seed: int,
    ) -> None:
        body_names = self._model_names(sim.model, "body", "nbody")
        obs_keys = self._libero_observation_keys(env)
        missing: list[str] = []
        invalid_pose: list[str] = []

        for instance in sorted(spec["objects"]):
            body_id = self._find_body_id(sim, instance, body_names=body_names)
            if body_id is None and not self._instance_in_observation(instance, obs_keys):
                missing.append(instance)
                continue
            if body_id is None:
                continue
            pos = sim.data.xpos[body_id]
            if not self._pose_looks_valid(pos):
                invalid_pose.append(f"{instance}@{self._format_vec3(pos)}")

        for instance, fixture_type in sorted(spec["fixtures"].items()):
            if fixture_type in _WORKSPACE_TYPES:
                continue
            if self._find_body_id(sim, instance, body_names=body_names) is None:
                missing.append(instance)

        if missing:
            result["errors"].append(
                f"LIBERO scene missing declared body/object(s) for seed {seed}: {missing}"
            )
        if invalid_pose:
            result["errors"].append(
                f"LIBERO scene has invalid object pose(s) for seed {seed}: {invalid_pose}"
            )

    def _verify_libero_initial_contacts(
        self, sim: Any, result: dict[str, Any], *, seed: int,
    ) -> None:
        data = getattr(sim, "data", None)
        model = getattr(sim, "model", None)
        contacts = getattr(data, "contact", None)
        ncon = int(getattr(data, "ncon", 0) or 0)
        if data is None or model is None or contacts is None or ncon <= 0:
            return

        robot_penetrations: list[str] = []
        scene_penetrations: list[str] = []
        for idx in range(ncon):
            contact = contacts[idx]
            try:
                dist = float(contact.dist)
            except Exception:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            names1 = self._contact_names(model, geom1)
            names2 = self._contact_names(model, geom2)
            name1 = "/".join(n for n in names1 if n) or f"geom:{geom1}"
            name2 = "/".join(n for n in names2 if n) or f"geom:{geom2}"

            side1_robot = self._is_robot_name(name1)
            side2_robot = self._is_robot_name(name2)
            if side1_robot or side2_robot:
                if side1_robot and side2_robot:
                    continue
                if dist < -0.003:
                    robot_penetrations.append(f"{name1} <-> {name2} dist={dist:.4f}")
                continue

            if dist < -0.03 and not self._is_workspace_contact(name1, name2):
                scene_penetrations.append(f"{name1} <-> {name2} dist={dist:.4f}")

        if robot_penetrations:
            result["errors"].append(
                "Robot starts in penetrating contact for seed "
                f"{seed}: {robot_penetrations[:5]}"
            )
        if scene_penetrations:
            result["errors"].append(
                "Scene objects start in severe penetrating contact for seed "
                f"{seed}: {scene_penetrations[:5]}"
            )

    def _verify_libero_goal_predicates_executable(
        self, env: Any, result: dict[str, Any], *, seed: int,
    ) -> None:
        """Probe LIBERO's symbolic goal evaluator for immediate runtime errors."""
        domain_env = self._get_libero_domain_env(env)
        if domain_env is None:
            return
        parsed = getattr(domain_env, "parsed_problem", None)
        eval_pred = getattr(domain_env, "_eval_predicate", None)
        if not isinstance(parsed, dict) or not callable(eval_pred):
            return

        for state in parsed.get("goal_state", []) or []:
            desc = "[" + " ".join(str(s) for s in state) + "]"
            try:
                eval_pred(state)
            except Exception as exc:
                result["errors"].append(
                    "LIBERO goal predicate cannot be evaluated for seed "
                    f"{seed}: {desc} raised {type(exc).__name__}: {exc}"
                )

    def _parse_bddl_scene_spec(self, bddl_text: str) -> dict[str, Any]:
        fixtures_section = self._extract_bddl_section(bddl_text, ":fixtures")
        objects_section = self._extract_bddl_section(bddl_text, ":objects")
        regions_section = self._extract_bddl_section(bddl_text, ":regions")
        init_section = self._extract_bddl_section(bddl_text, ":init")
        goal_section = self._extract_bddl_section(bddl_text, ":goal")

        problem_match = re.search(r"\(define\s+\(problem\s+([A-Za-z_]\w*)\)", bddl_text)
        language_match = re.search(r"\(:language\s+(.+?)\)", bddl_text, re.DOTALL)
        return {
            "problem_class": problem_match.group(1) if problem_match else "",
            "language": self._collapse_ws(language_match.group(1)) if language_match else "",
            "fixtures": self._parse_typed_declarations(fixtures_section),
            "objects": self._parse_typed_declarations(objects_section),
            "regions": self._parse_bddl_regions(regions_section),
            "init_text": init_section,
            "goal_text": goal_section,
        }

    def _extract_bddl_section(self, bddl_text: str, section_name: str) -> str:
        match = re.search(r"\(\s*" + re.escape(section_name) + r"(?=\s|\))", bddl_text)
        if not match:
            return ""
        depth = 0
        for idx in range(match.start(), len(bddl_text)):
            char = bddl_text[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return bddl_text[match.start(): idx + 1]
        return bddl_text[match.start():]

    def _section_body(self, section_text: str) -> str:
        text = section_text.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        if not text:
            return ""
        parts = text.split(None, 1)
        return parts[1] if len(parts) == 2 else ""

    def _parse_typed_declarations(self, section_text: str) -> dict[str, str]:
        declarations: dict[str, str] = {}
        for raw_line in self._section_body(section_text).splitlines():
            line = re.sub(r";.*$", "", raw_line).strip()
            if not line or "-" not in line:
                continue
            for decl_match in re.finditer(
                r"((?:\b[A-Za-z_]\w*\b\s*)+)-\s*([A-Za-z_]\w*)",
                line,
            ):
                names_part, type_name = decl_match.groups()
                for name in re.findall(r"\b[A-Za-z_]\w*\b", names_part):
                    declarations[name] = type_name
        return declarations

    def _parse_bddl_regions(self, section_text: str) -> dict[str, str]:
        regions: dict[str, str] = {}
        body = self._section_body(section_text)
        for region_match in re.finditer(r"\(\s*([A-Za-z_]\w*)\b", body):
            region_name = region_match.group(1)
            if region_name.startswith(":"):
                continue
            region_block = self._balanced_block_from(body, region_match.start())
            target_match = re.search(r"\(:target\s+([A-Za-z_]\w*)\)", region_block)
            if target_match:
                regions[region_name] = target_match.group(1)
        return regions

    def _balanced_block_from(self, text: str, start: int) -> str:
        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return text[start: idx + 1]
        return text[start:]

    def _required_bddl_assets(
        self,
        task_proposal: dict[str, Any],
        catalog: dict[str, Any],
        *,
        field_name: str,
    ) -> tuple[set[str], set[str]]:
        structured: set[str] = set()
        for raw in task_proposal.get(field_name) or []:
            base = self._catalog_base_name(str(raw))
            if base in catalog:
                structured.add(base)

        for pred in task_proposal.get("goal") or []:
            if not isinstance(pred, list):
                continue
            for arg in pred[1:]:
                base = self._catalog_base_name(str(arg))
                if base in catalog:
                    structured.add(base)

        language_text = " ".join(
            str(task_proposal.get(key) or "")
            for key in ("language", "goal_conditions")
        )
        language_mentions = self._catalog_mentions_in_text(language_text, catalog)
        return structured, language_mentions - structured

    def _catalog_mentions_in_text(self, text: str, catalog: dict[str, Any]) -> set[str]:
        normalized_text = f" {self._normalize_asset_text(text)} "
        no_and_text = normalized_text.replace(" and ", " ")
        mentions: set[str] = set()
        selected_spans: list[tuple[int, int]] = []
        for asset in sorted(catalog, key=len, reverse=True):
            phrase = self._normalize_asset_text(asset)
            if not phrase:
                continue
            span = self._phrase_span(normalized_text, phrase)
            if span is None:
                span = self._phrase_span(no_and_text, phrase)
            if span is None:
                continue
            if any(max(span[0], s0) < min(span[1], s1) for s0, s1 in selected_spans):
                continue
            selected_spans.append(span)
            mentions.add(asset)
        return mentions

    def _phrase_span(self, text: str, phrase: str) -> tuple[int, int] | None:
        match = re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text)
        if match:
            return match.span()
        return None

    def _validate_bddl_references(self, spec: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        fixtures: dict[str, str] = spec["fixtures"]
        objects: dict[str, str] = spec["objects"]
        declared_instances = set(fixtures) | set(objects)
        valid_locations = set(declared_instances)

        regions: dict[str, str] = spec["regions"]
        valid_locations.update(regions)
        for region_name, target in regions.items():
            valid_locations.add(f"{target}_{region_name}")
        for fixture_inst, fixture_type in fixtures.items():
            for region_name in FIXTURES.get(fixture_type, {}).get("regions", []):
                valid_locations.add(f"{fixture_inst}_{region_name}")

        for section_name in ("init_text", "goal_text"):
            body = spec.get(section_name, "")
            for predicate, args in self._iter_bddl_predicates(body):
                if predicate.lower() in {"and", "or", "not"}:
                    continue
                for arg in args:
                    if arg in valid_locations:
                        continue
                    base = self._declared_instance_prefix(arg, declared_instances)
                    if base is not None:
                        continue
                    if re.search(r"_\d+(?:_|$)", arg):
                        errors.append(
                            f"BDDL {section_name.replace('_text', '')} references undeclared "
                            f"instance/region '{arg}'"
                        )
        return errors

    def _iter_bddl_predicates(self, section_text: str) -> list[tuple[str, list[str]]]:
        predicates: list[tuple[str, list[str]]] = []
        for match in re.finditer(r"\(([A-Za-z_]\w*)\s+([^()]+?)\)", section_text):
            pred = match.group(1)
            args = re.findall(r"\b[A-Za-z_]\w*\b", match.group(2))
            predicates.append((pred, args))
        return predicates

    def _get_libero_sim(self, env: Any) -> Any | None:
        domain_env = self._get_libero_domain_env(env)
        if domain_env is not None and hasattr(domain_env, "sim"):
            return getattr(domain_env, "sim", None)
        low_level = getattr(env, "low_level_env", env)
        handle = getattr(low_level, "handle", None)
        raw_env = getattr(handle, "env", None)
        return getattr(raw_env, "sim", None)

    def _get_libero_domain_env(self, env: Any) -> Any | None:
        low_level = getattr(env, "low_level_env", env)
        handle = getattr(low_level, "handle", None)
        raw_env = getattr(handle, "env", None)
        first_with_sim = None
        for _ in range(5):
            if raw_env is None:
                return first_with_sim
            if first_with_sim is None and hasattr(raw_env, "sim"):
                first_with_sim = raw_env
            if hasattr(raw_env, "parsed_problem") and hasattr(raw_env, "_eval_predicate"):
                return raw_env
            if hasattr(raw_env, "obj_body_id"):
                return raw_env
            next_env = getattr(raw_env, "env", None)
            if next_env is raw_env:
                return first_with_sim or raw_env
            raw_env = next_env
        return first_with_sim or raw_env

    def _libero_observation_keys(self, env: Any) -> set[str]:
        low_level = getattr(env, "low_level_env", env)
        obs = getattr(low_level, "_current_obs", {}) or {}
        return set(obs.keys()) if isinstance(obs, dict) else set()

    def _instance_in_observation(self, instance: str, obs_keys: set[str]) -> bool:
        return (
            f"{instance}_pos" in obs_keys
            or f"{instance}_quat" in obs_keys
            or any(key.startswith(instance + "_") for key in obs_keys)
        )

    def _model_names(self, model: Any, kind: str, count_attr: str) -> list[str]:
        count = int(getattr(model, count_attr, 0) or 0)
        name_fn = getattr(model, f"{kind}_id2name", None)
        names: list[str] = []
        for idx in range(count):
            try:
                names.append(name_fn(idx) if name_fn else "")
            except Exception:
                names.append("")
        return names

    def _find_body_id(
        self, sim: Any, instance: str, *, body_names: list[str] | None = None,
    ) -> int | None:
        model = sim.model
        candidates = [
            instance,
            f"{instance}_main",
            f"{instance}_root",
        ]
        body_name2id = getattr(model, "body_name2id", None)
        if body_name2id is not None:
            for name in candidates:
                try:
                    return int(body_name2id(name))
                except Exception:
                    pass

        names = body_names if body_names is not None else self._model_names(model, "body", "nbody")
        for idx, body_name in enumerate(names):
            if not body_name:
                continue
            if body_name == instance or body_name.startswith(instance + "_"):
                return idx
        return None

    def _contact_names(self, model: Any, geom_id: int) -> tuple[str, str]:
        geom_name = ""
        body_name = ""
        try:
            geom_name = model.geom_id2name(geom_id) or ""
        except Exception:
            pass
        try:
            body_id = int(model.geom_bodyid[geom_id])
            body_name = model.body_id2name(body_id) or ""
        except Exception:
            pass
        return geom_name, body_name

    def _is_robot_name(self, name: str) -> bool:
        lowered = name.lower()
        return any(
            marker in lowered
            for marker in ("robot0", "panda", "gripper", "finger", "hand", "eef")
        )

    def _is_workspace_contact(self, name1: str, name2: str) -> bool:
        combined = f"{name1} {name2}".lower()
        return any(marker in combined for marker in ("table", "floor", "workspace"))

    def _pose_looks_valid(self, pos: Any) -> bool:
        try:
            import numpy as np

            arr = np.asarray(pos, dtype=float).reshape(-1)
            if arr.size < 3 or not np.isfinite(arr[:3]).all():
                return False
            x, y, z = arr[:3]
            return -2.0 <= x <= 2.0 and -2.0 <= y <= 2.0 and -0.05 <= z <= 2.0
        except Exception:
            return False

    def _format_vec3(self, pos: Any) -> str:
        try:
            values = [float(v) for v in list(pos)[:3]]
            return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"
        except Exception:
            return str(pos)

    def _catalog_base_name(self, name: str) -> str:
        normalized = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
        if normalized in OBJECTS or normalized in FIXTURES:
            return normalized
        match = re.match(r"^(.+)_\d+(?:_.*)?$", normalized)
        return match.group(1) if match else normalized

    def _language_asset_satisfied(self, asset: str, declared_types: set[str]) -> bool:
        equivalent_types = _LANGUAGE_ASSET_EQUIVALENTS.get(asset, {asset})
        return bool(equivalent_types & declared_types)

    def _declared_instance_prefix(
        self, name: str, declared_instances: set[str],
    ) -> str | None:
        for instance in declared_instances:
            if name == instance or name.startswith(instance + "_"):
                return instance
        return None

    def _normalize_asset_text(self, text: str) -> str:
        text = text.lower().replace("_", " ")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return self._collapse_ws(text)

    def _collapse_ws(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    # ------------------------------------------------------------------
    # BDDL validation
    # ------------------------------------------------------------------

    def _validate_bddl(self, bddl_text: str) -> list[str]:
        """Validate BDDL structure. Returns list of error strings (empty = valid)."""
        errors = []

        if not bddl_text.strip().startswith("(define"):
            errors.append("Must start with (define ...)")
            return errors

        # Check required sections
        for section in [":domain", ":language", ":regions", ":fixtures", ":objects", ":init", ":goal"]:
            if section not in bddl_text:
                errors.append(f"Missing section: {section}")

        # Check problem class name
        match = re.search(r"\(define\s+\(problem\s+(\w+)\)", bddl_text)
        if match:
            problem_name = match.group(1)
            if problem_name not in PROBLEM_CLASSES:
                errors.append(
                    f"Unknown problem class '{problem_name}'. "
                    f"Must be one of: {list(PROBLEM_CLASSES.keys())}"
                )
        else:
            errors.append("Could not parse problem class name")

        # Check that objects in :goal are declared in :objects or :fixtures
        fix_match = re.search(r":fixtures\s*(.*?)\)", bddl_text, re.DOTALL)
        goal_match = re.search(r":goal\s*\((.*)\)\s*\)\s*$", bddl_text, re.DOTALL)
        if goal_match:
            goal_text = goal_match.group(1)
            # Extract instance names from goal (e.g., akita_black_bowl_1, flat_stove_1)
            goal_instances = set(re.findall(r"\b(\w+_\d+)\b", goal_text))
            # Also match fixture_region patterns
            goal_instances.update(re.findall(r"\b(\w+_\d+_\w+)\b", goal_text))

            # Collect declared instances
            declared = set()
            obj_match = re.search(r":objects\s*(.*?)\)", bddl_text, re.DOTALL)
            if obj_match:
                declared.update(re.findall(r"\b(\w+_\d+)\b", obj_match.group(1)))
            if fix_match:
                declared.update(re.findall(r"\b(\w+_\d+)\b", fix_match.group(1)))
            # Fixture regions are implicitly declared
            for d in list(declared):
                for f_info in FIXTURES.values():
                    for region in f_info["regions"]:
                        declared.add(f"{d}_{region}")

            undeclared = {g for g in goal_instances if not any(g.startswith(d) for d in declared)} - declared
            if undeclared:
                errors.append(f"Goal references undeclared instances: {undeclared}")

        # Check fixture sub-region names match catalog
        # e.g., microwave_1_heating_region OK, microwave_1_contain_region BAD
        valid_fixture_regions: dict[str, list[str]] = {}
        if fix_match:
            for inst_name, ftype in re.findall(r"(\w+_\d+)\s*-\s*(\w+)", fix_match.group(1)):
                if ftype in FIXTURES:
                    valid_fixture_regions[inst_name] = FIXTURES[ftype]["regions"]

        # Scan all region references in :goal and :init for invalid fixture sub-regions
        for section_name, section_re in [
            ("goal", r":goal\s*\((.+)\)\s*\)\s*$"),
            ("init", r":init\s*\((.+?)\)\s*(?:\(:)", ),
        ]:
            sec_match = re.search(section_re, bddl_text, re.DOTALL)
            if not sec_match:
                continue
            sec_text = sec_match.group(1)
            for inst_name, valid_regions in valid_fixture_regions.items():
                # Find all references like microwave_1_XXXX_region
                for ref in re.findall(rf"\b{re.escape(inst_name)}_(\w+)\b", sec_text):
                    if ref not in valid_regions and ref not in ("init_region",):
                        errors.append(
                            f"Invalid fixture sub-region '{inst_name}_{ref}' in :{section_name}. "
                            f"Valid regions for {inst_name}: {valid_regions}"
                        )

        # Check workspace type in :fixtures (common LLM error: using problem class as type)
        if fix_match:
            for inst_name, ftype in re.findall(r"(\w+)\s*-\s*(\w+)", fix_match.group(1)):
                if ftype in PROBLEM_CLASSES:
                    errors.append(
                        f"Fixture '{inst_name}' has type '{ftype}' which is a problem class name, "
                        f"not a fixture type. Use the workspace type instead "
                        f"(e.g., 'table', 'kitchen_table')."
                    )

        # Check balanced parentheses
        if bddl_text.count("(") != bddl_text.count(")"):
            errors.append(
                f"Unbalanced parentheses: {bddl_text.count('(')} open, "
                f"{bddl_text.count(')')} close"
            )

        return errors

    def _fix_bddl(
        self, bddl_text: str, errors: list[str], task_proposal: dict[str, Any]
    ) -> str:
        """Ask LLM to fix BDDL validation errors."""
        system_prompt = "Fix the BDDL specification. Output ONLY the corrected BDDL in a code fence."
        user_prompt = (
            f"The following BDDL has errors:\n\n```\n{bddl_text}\n```\n\n"
            f"Errors:\n" + "\n".join(f"- {e}" for e in errors) +
            f"\n\nOriginal task: {task_proposal.get('language', '')}\n"
            f"Fix ALL errors and output the corrected BDDL."
        )
        response = query_llm_text(system_prompt, user_prompt)
        return self._extract_bddl_from_response(response)

    # ------------------------------------------------------------------
    # Environment instantiation
    # ------------------------------------------------------------------

    def _instantiate_libero_env(self, bddl_path: Path, old_env: Any | None) -> Any:
        """Create a LIBERO CodeExecutionEnvBase from a BDDL file."""
        # Mirror config from old env if available
        api_names = ["FrankaLiberoPrivilegedApi"]
        privileged = True
        viser_debug = False
        if old_env is not None:
            api_names = list(getattr(old_env, "_apis", {}).keys()) or api_names
            cfg_obj = getattr(old_env, "cfg", None)
            privileged = getattr(cfg_obj, "privileged", True) if cfg_obj else True
            low_level = getattr(old_env, "low_level_env", None)
            viser_debug = bool(
                getattr(cfg_obj, "viser_debug", False)
                or getattr(low_level, "viser_debug", False)
            )

        env = build_full_libero_env_from_bddl(
            bddl_path=str(bddl_path),
            api_names=api_names,
            privileged=privileged,
            viser_debug=viser_debug,
        )
        logger.info(f"Instantiated LIBERO env from BDDL: {bddl_path.name}")
        return env


def build_full_libero_env_from_bddl(
    bddl_path: str,
    api_names: list[str],
    privileged: bool = False,
    viser_debug: bool = False,
) -> Any:
    """Build a fully-configured FrankaLiberoCodeEnv (low-level + APIs) from BDDL.

    Standalone counterpart to ``EnvironmentCreator._instantiate_libero_env``,
    callable from outside the orchestrator process. The parallel-sub-agent
    worker subprocess uses this to spin up its own env from the same BDDL
    the main process used. Returns an env that's already ``reset()``-ed
    and has the named APIs attached as exec-scope functions.
    """
    from rats.envs.configs.instantiate import instantiate

    bddl_text = Path(bddl_path).read_text()
    lang_match = re.search(r":language\s+(.+?)(?:\)|\n)", bddl_text)
    task_language = lang_match.group(1).strip() if lang_match else "complete the task"

    env_cfg = {
        "_target_": "rats.envs.tasks.franka.franka_libero_env.FrankaLiberoCodeEnv",
        "cfg": {
            "_target_": "rats.envs.tasks.base.CodeExecEnvConfig",
            "low_level": {
                "_target_": "rats.agents.environment_creator._create_libero_env_from_bddl",
                "bddl_path": str(bddl_path),
                "viser_debug": viser_debug,
            },
            "privileged": privileged,
            "viser_debug": viser_debug,
            "apis": list(api_names),
            "prompt": (
                f"You are controlling a Franka Emika robot with API described below.\n"
                f"Goal: {task_language}\n"
                f"Write executable Python code (no code fences). "
                f"APIs are already imported. Import numpy explicitly if needed.\n"
            ),
        },
    }

    original_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        env = instantiate(env_cfg)
    finally:
        sys.argv = original_argv

    env.reset()
    return env


def _create_libero_env_from_bddl(bddl_path: str, viser_debug: bool = False):
    """Factory function to create a FrankaLiberoEnv from a custom BDDL file.

    This is called by the instantiate() system. It bypasses the standard
    suite_name/task_id lookup and directly creates the OffScreenRenderEnv.
    """
    import os

    # Add LIBERO to path
    vendor_root = str(_PROJECT_ROOT / "rats" / "third_party" / "LIBERO-PRO")
    if os.path.isdir(vendor_root) and vendor_root not in sys.path:
        sys.path.append(vendor_root)

    from rats.envs.simulators.libero import FrankaLiberoEnv
    from rats.integrations.libero import LiberoHandle

    from libero.envs import OffScreenRenderEnv  # type: ignore

    # Read language from BDDL
    bddl_content = Path(bddl_path).read_text()
    lang_match = re.search(r":language\s+(.+?)(?:\)|\n)", bddl_content)
    task_language = lang_match.group(1).strip() if lang_match else "complete the task"

    # Create the robosuite env directly from BDDL
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": 512,
        "camera_widths": 800,
        "controller": "JOINT_POSITION",
        "horizon": 4000,
        "control_freq": 20,
        "camera_depths": True,
    }
    raw_env = OffScreenRenderEnv(**env_args)
    raw_env.seed(0)

    # Use the BDDL filename slug as suite_name so the env carries a
    # task-specific identity rather than the generic "novel". libero_utils
    # builds activity_name = f"{suite_name}_task{task_id}" — without this
    # patch, every novel-BDDL env became "novel_task0", which then leaked
    # into the report whenever _build_proposal_from_env fell back to the
    # current env's identity.
    bddl_basename = Path(bddl_path).stem
    # BDDL filenames are written as f"{slug}_{md5[:8]}.bddl"; drop the
    # trailing 8-hex-char hash so the suite_name is the readable slug.
    suite_slug = re.sub(r"_[0-9a-f]{8}$", "", bddl_basename) or "novel"

    handle = LiberoHandle(
        env=raw_env,
        suite_name=suite_slug,
        task_id=0,
        task_language=task_language,
        init_states=None,
    )

    # Create FrankaLiberoEnv by injecting the handle
    # We use a minimal subclass to bypass the normal __init__ which calls load_libero_task
    libero_env = _FrankaLiberoEnvFromHandle(handle, viser_debug=viser_debug)
    return libero_env


class _FrankaLiberoEnvFromHandle:
    """Create a FrankaLiberoEnv with a pre-built LiberoHandle.

    Monkey-patches the module-level ``load_libero_task`` reference inside
    ``rats.envs.simulators.libero`` so that ``FrankaLiberoEnv.__init__``
    picks up our handle instead of querying the benchmark registry.
    """

    def __new__(cls, handle, viser_debug: bool = False):
        import rats.envs.simulators.libero as sim_mod

        original_load = sim_mod.load_libero_task

        def _patched_load(**kwargs):
            return handle

        sim_mod.load_libero_task = _patched_load
        try:
            env = sim_mod.FrankaLiberoEnv(
                suite_name="novel",
                task_id=0,
                privileged=True,
                max_steps=4000,
                viser_debug=viser_debug,
            )
        finally:
            sim_mod.load_libero_task = original_load

        return env
