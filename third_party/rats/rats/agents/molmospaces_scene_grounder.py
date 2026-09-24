"""VLM-grounded naming + difficulty cues for the MolmoSpaces inventory.

The novel-task proposer for MolmoSpaces (`_propose_novel_molmospaces_open`
in `agents/task_proposer.py`) needs two things from the active scene that
the symbolic inventory cannot provide:

1. **Human-readable display names** for each inventory item that the LLM
   can refer to unambiguously. Internal names look like
   `Cup_<32-char-hash>_0_0_0` — useless for picking which cup the agent
   should target. The VLM looks at the rendered scene and produces
   short phrases like "the white cup on the kitchen counter" or "the
   bottom drawer of the dresser".

2. **Per-target difficulty cues** — short flags that the proposer can
   use to weight its choice ("the sponge is wedged behind the faucet",
   "the drawer is fully obscured by the toaster"). Cheap because we
   reuse the same image we just sent for naming.

A single LLM call produces both outputs. Result is cached per
`(house_index, episode_seed)` so re-grounding the same scene across
proposer iterations is free.

Falls back to a deterministic templater (`f"the {category} in the {room}"`)
when the VLM call fails or vision is disabled — the proposer keeps
working, it just has less colourful names.
"""

from __future__ import annotations

import json
import logging
import hashlib
from typing import Any

import numpy as np

from rats.agents.base_agent import image_to_data_url, query_llm_json

logger = logging.getLogger("rats.molmospaces_scene_grounder")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def ground_inventory(
    env: Any,
    inventory: dict[str, Any],
    *,
    enabled: bool = True,
    use_geometric_visibility_gate: bool = False,
    cache: dict[tuple[Any, Any], dict[str, Any]] | None = None,
    cache_key: tuple[Any, Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Return ``{display_names, difficulty_cues}`` for the inventory.

    Args:
        env: the active rats env (for ``render`` / ``render_wrist``).
        inventory: payload from ``extract_molmospaces_scene_inventory``.
        enabled: when False (or no API key, or render fails) skip the
            VLM call and return the deterministic fallback. Lets the
            YAML knob (``molmospaces.vlm_grounding.enabled``) turn this
            off without conditional code at the call site.
        use_geometric_visibility_gate: when True, apply the conservative
            object-center camera-frustum veto after the VLM image pass.
            Defaults off because large articulated targets (dressers,
            cabinets) can be visibly present while their inventory center
            projects outside the current camera frame.
        cache: optional dict for memoization across iterations. Keyed
            by ``cache_key`` (typically ``(house_index, episode_seed)``).
        cache_key: explicit cache key; falls back to
            ``(inventory.house_index, inventory.scene_dataset)``.
        model: override the default vision-capable LLM.

    Returns:
        ``{
            "display_names": {internal_name: human_readable_phrase},
            "difficulty_cues": {internal_name: short_string_or_empty},
            "visible": {internal_name: bool},
        }``

    The keys cover every entry in ``inventory['pickables'] +
    inventory['receptacles'] + inventory['articulations']``. Any
    internal name missing from the VLM's response gets the
    deterministic fallback so the proposer can always look every name
    up. ``visible`` is a separate per-item boolean so the proposer can
    filter targets the robot cannot actually see from its current pose,
    which the previous merged ``difficulty_cue`` field handled
    unreliably (the VLM would assign positions to off-camera items
    instead of flagging them).
    """
    items = _flatten_inventory(inventory)
    if not items:
        return {"display_names": {}, "difficulty_cues": {}}

    fallback = _fallback_grounding(items)
    scope = _current_task_room_scope(inventory, items)
    scoped_items = scope["items"]
    scoped_fallback = _fallback_grounding(scoped_items)

    if not enabled:
        key: tuple[Any, ...] = cache_key or (
            inventory.get("house_index"),
            inventory.get("scene_dataset"),
            "fallback",
        )
        if cache is not None and key in cache:
            return cache[key]
        result = fallback
    else:
        obs = _capture_observation(env)
        agent_b64 = _capture_image(env, kind="agent")
        wrist_b64 = _capture_image(env, kind="wrist")
        images = [b for b in (agent_b64, wrist_b64) if b]
        key = cache_key or (
            inventory.get("house_index"),
            inventory.get("scene_dataset"),
            scope.get("room") or "all_rooms",
            _items_fingerprint(scoped_items),
            _image_fingerprint(images),
            "geo_gate",
            bool(use_geometric_visibility_gate),
        )
        if cache is not None and key in cache:
            return cache[key]
        if not images:
            logger.info("VLM grounding skipped: no images captured; using fallback")
            result = fallback
        else:
            try:
                if scope.get("room") is not None and len(scoped_items) < len(items):
                    logger.info(
                        "VLM grounding scoped to room %s via %s: %d/%d items",
                        scope.get("room"),
                        scope.get("anchor_name") or "anchored target",
                        len(scoped_items),
                        len(items),
                    )
                result = _vlm_ground(scoped_items, images, model=model)
                # Backfill any missing names so callers always look up clean.
                # Missing VLM visibility is treated as not visible in the VLM
                # path. Items outside the active room are also backfilled as
                # hidden so visible-only proposer prompts do not see them.
                _merge_with_fallback(result, scoped_fallback, missing_visible=False)
                _merge_out_of_scope_as_hidden(
                    result,
                    fallback,
                    scoped_names={str(it["internal_name"]) for it in scoped_items},
                )
                geometric_visible = (
                    _geometric_visibility(scoped_items, obs)
                    if use_geometric_visibility_gate else {}
                )
                if geometric_visible:
                    _apply_geometric_visibility_gate(result, geometric_visible)
            except Exception as exc:
                logger.warning(
                    "VLM grounding failed (%s: %s); using fallback",
                    type(exc).__name__,
                    exc,
                )
                result = fallback

    _attach_scope_metadata(result, scope, total_items=len(items))

    if cache is not None:
        cache[key] = result
    return result


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

def _flatten_inventory(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    """Collapse pickables/receptacles/articulations into one tagged list."""
    items: list[dict[str, Any]] = []
    for kind in ("pickables", "receptacles", "articulations"):
        for entry in inventory.get(kind, []) or []:
            if not isinstance(entry, dict) or not entry.get("internal_name"):
                continue
            items.append({
                "kind": kind,
                "internal_name": entry["internal_name"],
                "category": entry.get("category", ""),
                "room": entry.get("room", "unknown"),
                "position": entry.get("position", [0.0, 0.0, 0.0]),
                "joints": entry.get("joints", []),
            })
    return items


def _current_task_room_scope(
    inventory: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the live task room scope used to trim VLM grounding input.

    The bridge may switch houses between playtime iterations, so this uses
    the current ``inventory["anchored_target"]`` snapshot rather than a
    startup-time config guess. If the anchor or room cannot be resolved, the
    grounder conservatively keeps the full inventory.
    """
    anchor = inventory.get("anchored_target")
    if not isinstance(anchor, dict):
        return {"room": None, "anchor_name": None, "items": items}

    by_internal = {str(it.get("internal_name")): it for it in items}
    for name in _anchored_room_candidate_names(anchor):
        item = by_internal.get(name) or _find_item_by_joint_name(items, name)
        if not item:
            continue
        room = _normalize_room(item.get("room"))
        if room is None:
            continue
        scoped = [it for it in items if _normalize_room(it.get("room")) == room]
        if scoped:
            return {"room": room, "anchor_name": name, "items": scoped}

    return {"room": None, "anchor_name": None, "items": items}


def _anchored_room_candidate_names(anchor: dict[str, Any]) -> list[str]:
    """Ordered internal-name guesses for the task's main object."""
    task_type = str(anchor.get("task_type") or "").strip().lower()
    if task_type in {"open", "close"}:
        keys = ("joint_name", "pickup_obj_name", "place_receptacle_name")
    elif task_type == "pick_and_place":
        keys = ("pickup_obj_name", "place_receptacle_name", "joint_name")
    else:
        keys = ("pickup_obj_name", "joint_name", "place_receptacle_name")

    names: list[str] = []
    for key in keys:
        val = anchor.get(key)
        if val is None:
            continue
        name = str(val).strip()
        if name and name not in names:
            names.append(name)
    return names


def _find_item_by_joint_name(
    items: list[dict[str, Any]],
    joint_name: str,
) -> dict[str, Any] | None:
    for item in items:
        for joint in item.get("joints") or []:
            if isinstance(joint, dict) and str(joint.get("name") or "") == joint_name:
                return item
    return None


def _normalize_room(value: Any) -> str | None:
    room = str(value or "").strip()
    if not room or room.lower() in {"unknown", "none", "null"}:
        return None
    return room


def _fallback_grounding(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic name templater used when the VLM is unavailable.

    NOTE on visibility default: previously this returned ``visible=True``
    for every item under the reasoning "don't break runs where vision
    is intentionally disabled". That was wrong: defaulting to True
    silently misled the task_proposer into proposing tasks on objects
    no one had visually confirmed, and downstream verifiers had no
    second signal to catch it. We now default to ``False`` — "no VLM
    signal" should NOT be conflated with "all objects confirmed visible".
    Runs that genuinely want to skip vision should either provide the
    geometric visibility gate (`use_geometric_visibility_gate=True`)
    or override `visible` explicitly upstream.


    `the <category> in the <room>` is the safest generic phrase. We
    don't try to disambiguate duplicates — that's what the VLM is for.
    Difficulty cues default to empty since we can't actually see the
    scene; the proposer treats empty as "no known blocker". Visibility
    defaults to ``False`` (see NOTE in the docstring summary above).
    """
    display_names: dict[str, str] = {}
    difficulty: dict[str, str] = {}
    visible: dict[str, bool] = {}
    points: dict[str, list[int] | None] = {}
    for it in items:
        cat = it["category"] or "object"
        room = it["room"] or "unknown"
        if room and room != "unknown":
            display_names[it["internal_name"]] = f"the {cat} in the {room}"
        else:
            display_names[it["internal_name"]] = f"the {cat}"
        difficulty[it["internal_name"]] = ""
        visible[it["internal_name"]] = False
        points[it["internal_name"]] = None
    return {
        "display_names": display_names,
        "difficulty_cues": difficulty,
        "visible": visible,
        "points": points,
    }


def _capture_image(env: Any, *, kind: str) -> str | None:
    """Render and base64-encode either the agentview or the wrist view."""
    low_level = getattr(env, "low_level_env", env)
    fn_name = "render_wrist" if kind == "wrist" else "render"
    fn = getattr(low_level, fn_name, None) or getattr(env, fn_name, None)
    if not callable(fn):
        return None
    try:
        try:
            arr = fn(mode="rgb_array")
        except TypeError:
            arr = fn()
    except Exception as exc:
        logger.debug("Render %s failed: %s", kind, exc)
        return None
    if arr is None:
        return None
    return image_to_data_url(arr)


def _capture_observation(env: Any) -> dict[str, Any] | None:
    """Best-effort current observation for geometric visibility checks."""
    low_level = getattr(env, "low_level_env", env)
    for owner in (low_level, env):
        if owner is None:
            continue
        for attr in ("get_observation", "_get_observation", "build_observation"):
            fn = getattr(owner, attr, None)
            if callable(fn):
                try:
                    obs = fn()
                    if isinstance(obs, dict):
                        return obs
                except Exception:
                    continue
    return None


def _image_fingerprint(images: list[str]) -> str:
    h = hashlib.sha1()
    for img in images:
        h.update(img.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16] if images else "no-images"


def _items_fingerprint(items: list[dict[str, Any]]) -> str:
    h = hashlib.sha1()
    for item in items:
        h.update(str(item.get("internal_name", "")).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16] if items else "no-items"


def _vlm_ground(
    items: list[dict[str, Any]],
    images: list[str],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Single LLM call: emit display_names + difficulty_cues + visible.

    The prompt ships the full item list so the model can't hallucinate
    new internal names — it can only assign phrases to names that
    already exist in the inventory. ``visible`` is a separate REQUIRED
    boolean per item so the model has a dedicated schema slot for "I
    cannot see this object" rather than burying it inside the optional
    free-form ``difficulty_cues`` string. The earlier "cover every
    item" instruction biased the model toward fabricating phrases for
    off-camera objects; removed.
    """
    # Keep each VLM call small enough that reasoning models do not burn the
    # entire completion budget on hidden reasoning and return an empty
    # assistant message.  Agent-IO showed the previous monolithic call ending
    # with finish_reason="length", completion_tokens=8192, reasoning_tokens=8192
    # and content="" on large MolmoSpaces inventories.
    if len(items) > 40:
        merged = {
            "display_names": {},
            "difficulty_cues": {},
            "visible": {},
            "points": {},
        }
        for start in range(0, len(items), 40):
            batch = items[start : start + 40]
            part = _vlm_ground(batch, images, model=model)
            for key in ("display_names", "difficulty_cues", "visible", "points"):
                if isinstance(part.get(key), dict):
                    merged[key].update(part[key])
        return merged

    # Positions are symbolic simulator metadata and were a major source of
    # prompt bloat.  The visibility decision must come from images only.
    prompt_items = [
        {
            "kind": it.get("kind", ""),
            "internal_name": it.get("internal_name", ""),
            "category": it.get("category", ""),
            "room": it.get("room", "unknown"),
        }
        for it in items
    ]
    item_blob = json.dumps(prompt_items, indent=2)
    system_prompt = (
        "You are a vision-language helper for a robot task proposer. "
        "Look at the provided agentview and wrist images of a household "
        "scene, then for each inventory item: (a) assign a short "
        "disambiguating phrase, (b) decide whether it is actually "
        "visible in either image, and (c) optionally flag a concrete "
        "non-visibility difficulty (obscured, far away, behind another "
        "object). Respond ONLY with valid JSON matching the requested "
        "schema. Do NOT invent positions for items you cannot see — "
        "set visible=false for those instead."
    )
    user_prompt = (
        "Inventory items (each has internal_name, category, room):\n"
        f"{item_blob}\n\n"
        "Return JSON with four top-level keys:\n"
        "  display_names: { internal_name: short human phrase },\n"
        "  difficulty_cues: { internal_name: short string OR empty },\n"
        "  visible: { internal_name: true | false },\n"
        "  points: { internal_name: [pixel_x, pixel_y] OR null }.\n"
        "Rules:\n"
        "- Use the EXACT internal_name strings from the input as keys.\n"
        "- visible MUST be true or false for every item. Set true ONLY "
        "if you can point to the item in the agentview or wrist image. "
        "Treat 'probably present off-screen' as visible=false. Do NOT "
        "use the symbolic position to decide; rely on the images.\n"
        "- points is the HARD evidence backing each visible=true claim. "
        "For every internal_name you mark visible=true, you MUST also "
        "give a [pixel_x, pixel_y] integer coordinate that points to "
        "that item in the agentview image (preferred) or the wrist "
        "image. If you cannot pick a specific pixel, set visible=false "
        "and points=null for that item — do NOT claim visible=true with "
        "a missing or null point. Pixel coordinates are in image space "
        "(origin at top-left). For visible=false items, set "
        "points=null.\n"
        "- display_name should be 3-10 words, lowercase, no quotes, "
        "no internal_name. Examples: 'the white cup on the kitchen counter', "
        "'the bottom drawer of the dresser', 'the open shelf above the stove'. "
        "For visible=false items, give a generic phrase like 'the <category>' "
        "with the room — do not invent a spatial location.\n"
        "- difficulty_cues should be a SHORT phrase ONLY when there is a "
        "concrete visual blocker for a visible=true item (object behind "
        "another, joint flush against a wall). For visible=false items, "
        "set difficulty_cue to 'not in current view'."
    )
    # max_tokens=6000 was too tight: each ground call returns 4 dicts
    # (display_names + difficulty_cues + visible + points) keyed by full
    # internal_name (e.g. "Irishpotato_121f03310f70f545adbe1151e4ea4b7f_1_0_2",
    # ~50 chars) for up to 40 items per batch. Even at reasoning_effort="low"
    # the JSON output alone runs ~8-10k tokens for a full batch, so 6000
    # truncated mid-dict and lost the trailing items' visibility / points
    # info. 16000 leaves comfortable headroom for the JSON payload + the
    # small reasoning trace that "low" still produces.
    result = query_llm_json(
        system_prompt,
        user_prompt,
        images=images,
        max_tokens=16000,
        reasoning_effort="low",
        **({"model": model} if model else {}),
    )
    if isinstance(result, list):
        # Gemini sometimes returns a flat list of per-item dicts instead of
        # the requested top-level-key structure. Reshape into expected format.
        display_names: dict[str, str] = {}
        difficulty_cues: dict[str, str] = {}
        visible_raw: dict[str, bool] = {}
        points_raw: dict[str, Any] = {}
        for entry in result:
            if not isinstance(entry, dict):
                continue
            name = entry.get("internal_name", "")
            if not name:
                continue
            if "display_name" in entry:
                display_names[name] = entry["display_name"]
            if "difficulty_cue" in entry or "difficulty_cues" in entry:
                difficulty_cues[name] = entry.get("difficulty_cue") or entry.get("difficulty_cues") or ""
            if "visible" in entry:
                visible_raw[name] = bool(entry["visible"])
            if "points" in entry or "point" in entry:
                points_raw[name] = entry.get("points") or entry.get("point")
        result = {
            "display_names": display_names,
            "difficulty_cues": difficulty_cues,
            "visible": visible_raw,
            "points": points_raw,
        }
    elif not isinstance(result, dict):
        raise ValueError(
            f"VLM grounding returned {type(result).__name__}, expected dict"
        )
    display_names = result.get("display_names") or {}
    difficulty_cues = result.get("difficulty_cues") or {}
    visible_raw = result.get("visible") or {}
    points_raw = result.get("points") or {}
    if (
        not isinstance(display_names, dict)
        or not isinstance(difficulty_cues, dict)
        or not isinstance(visible_raw, dict)
    ):
        raise ValueError(
            "VLM response missing display_names / difficulty_cues / visible dicts"
        )
    if not isinstance(points_raw, dict):
        # Tolerate older / unaware models that skip the new field; treat as
        # all-null. The downgrade path below will then demote every
        # visible=true item to visible=false, which is the safe default
        # under the new contract.
        points_raw = {}
    display_names = {
        str(k): str(v).strip() for k, v in display_names.items() if v
    }
    difficulty_cues = {
        str(k): str(v).strip() for k, v in difficulty_cues.items()
    }
    visible: dict[str, bool] = {}
    for k, v in visible_raw.items():
        if isinstance(v, bool):
            visible[str(k)] = v
        elif isinstance(v, str):
            visible[str(k)] = v.strip().lower() in ("true", "yes", "1", "y")
        else:
            visible[str(k)] = bool(v)

    # Hallucination gate. The grounder previously was allowed to claim
    # ``visible=true`` for an internal_name without producing any visual
    # evidence — this is how v7 iter 2 ended up with
    # ``"the pink soap bar on the toilet tank"`` (an internal barsoap mesh
    # the runtime perception pipeline could not locate). The new ``points``
    # field forces the model to commit to a [pixel_x, pixel_y] for every
    # visible=true claim; if the point is missing / null / non-numeric, we
    # downgrade visible to false and record a difficulty_cue so the
    # downstream proposer sees why.
    points: dict[str, list[int] | None] = {}
    for name in list(visible.keys()):
        raw = points_raw.get(name)
        coord: list[int] | None = None
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            try:
                x, y = int(raw[0]), int(raw[1])
                coord = [x, y]
            except (TypeError, ValueError):
                coord = None
        points[name] = coord
        if visible.get(name) and coord is None:
            visible[name] = False
            cue = difficulty_cues.get(name, "")
            note = "vlm could not commit to a pixel point"
            difficulty_cues[name] = (
                f"{cue}; {note}" if cue and note not in cue else (cue or note)
            )

    return {
        "display_names": display_names,
        "difficulty_cues": difficulty_cues,
        "visible": visible,
        "points": points,
    }


def _merge_with_fallback(
    result: dict[str, Any],
    fallback: dict[str, Any],
    *,
    missing_visible: bool | None = None,
) -> None:
    """Backfill missing internal names from the deterministic fallback."""
    fb_names = fallback.get("display_names", {})
    fb_cues = fallback.get("difficulty_cues", {})
    fb_visible = fallback.get("visible", {})
    result.setdefault("visible", {})
    for name, phrase in fb_names.items():
        result["display_names"].setdefault(name, phrase)
    for name, cue in fb_cues.items():
        result["difficulty_cues"].setdefault(name, cue)
    for name, vis in fb_visible.items():
        result["visible"].setdefault(
            name,
            vis if missing_visible is None else missing_visible,
        )


def _merge_out_of_scope_as_hidden(
    result: dict[str, Any],
    fallback: dict[str, Any],
    *,
    scoped_names: set[str],
) -> None:
    """Backfill non-grounded room items as hidden without sending them to VLM."""
    result.setdefault("display_names", {})
    result.setdefault("difficulty_cues", {})
    result.setdefault("visible", {})
    result.setdefault("points", {})
    for name, phrase in (fallback.get("display_names") or {}).items():
        if name in scoped_names:
            continue
        result["display_names"].setdefault(name, phrase)
        result["difficulty_cues"].setdefault(name, "outside active task room")
        result["visible"][name] = False
        result["points"].setdefault(name, None)


def _attach_scope_metadata(
    result: dict[str, Any],
    scope: dict[str, Any],
    *,
    total_items: int,
) -> None:
    result["scope"] = {
        "room": scope.get("room"),
        "anchor_name": scope.get("anchor_name"),
        "grounded_items": len(scope.get("items") or []),
        "total_items": total_items,
    }


def _geometric_visibility(
    items: list[dict[str, Any]],
    obs: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Conservative current-camera visibility from object positions.

    The VLM grounding pass can hallucinate an inventory item into the scene.
    This pass only vetoes items whose inventory position is outside both
    current camera frusta, or clearly behind nearer depth. Unknown calibration
    is left to the VLM result rather than treated as hidden.
    """
    if not isinstance(obs, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        internal = str(item.get("internal_name") or "")
        pos_raw = item.get("position")
        if not internal or not isinstance(pos_raw, (list, tuple)) or len(pos_raw) < 3:
            continue
        try:
            pos = np.asarray(
                [float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2])],
                dtype=np.float64,
            )
        except Exception:
            continue

        camera_results: list[dict[str, Any]] = []
        visible_any = False
        for cam_name in ("agentview", "robot0_eye_in_hand"):
            cam_result = _project_visibility_for_camera(
                pos,
                obs.get(cam_name),
                cam_name,
            )
            if cam_result:
                camera_results.append(cam_result)
                visible_any = visible_any or bool(cam_result.get("visible"))

        if camera_results:
            result[internal] = {
                "visible": visible_any,
                "cameras": camera_results,
                "reason": (
                    "projects into current camera"
                    if visible_any else "outside/occluded in current cameras"
                ),
            }
    return result


def _project_visibility_for_camera(
    pos_world: np.ndarray,
    cam: Any,
    cam_name: str,
) -> dict[str, Any] | None:
    if not isinstance(cam, dict):
        return None
    images = cam.get("images") if isinstance(cam.get("images"), dict) else {}
    rgb = images.get("rgb")
    if not isinstance(rgb, np.ndarray) or rgb.ndim < 2:
        return None
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    K = cam.get("intrinsics")
    T_cam_to_world = cam.get("pose_mat")
    try:
        K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
        T_arr = np.asarray(T_cam_to_world, dtype=np.float64).reshape(4, 4)
        world_to_cam = np.linalg.inv(T_arr)
        pos_h = np.concatenate([pos_world.reshape(3), [1.0]])
        pos_cam = (world_to_cam @ pos_h)[:3]
    except Exception:
        return None

    z = float(pos_cam[2])
    if not np.isfinite(z) or z <= 1e-4:
        return {
            "camera": cam_name,
            "visible": False,
            "reason": "behind_camera",
            "camera_z": z,
        }

    pix_h = K_arr @ pos_cam
    if abs(float(pix_h[2])) <= 1e-8:
        return {
            "camera": cam_name,
            "visible": False,
            "reason": "invalid_projection",
            "camera_z": z,
        }
    u = float(pix_h[0] / pix_h[2])
    v = float(pix_h[1] / pix_h[2])
    margin_px = 8.0
    in_frame = (
        -margin_px <= u < width + margin_px
        and -margin_px <= v < height + margin_px
    )
    if not in_frame:
        return {
            "camera": cam_name,
            "visible": False,
            "reason": "outside_frame",
            "pixel": [u, v],
            "image_size": [width, height],
            "camera_z": z,
        }

    occluded = _depth_strongly_occludes(images.get("depth"), u, v, z)
    return {
        "camera": cam_name,
        "visible": not occluded,
        "reason": "depth_occluded" if occluded else "in_frame",
        "pixel": [u, v],
        "image_size": [width, height],
        "camera_z": z,
    }


def _depth_strongly_occludes(depth: Any, u: float, v: float, z: float) -> bool:
    if not isinstance(depth, np.ndarray) or depth.size == 0:
        return False
    arr = np.asarray(depth, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return False
    h, w = arr.shape
    x = int(round(u))
    y = int(round(v))
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    y0, y1 = max(0, y - 2), min(h, y + 3)
    x0, x1 = max(0, x - 2), min(w, x + 3)
    patch = arr[y0:y1, x0:x1]
    finite = patch[np.isfinite(patch) & (patch > 1e-4)]
    if finite.size == 0:
        return False
    nearest = float(np.nanmin(finite))
    # Inventory positions are object centers, so a small nearer depth is
    # normal. Veto only when a surface is substantially closer.
    return nearest < (float(z) - 0.35)


def _apply_geometric_visibility_gate(
    result: dict[str, Any],
    geometric_visible: dict[str, dict[str, Any]],
) -> None:
    visible = result.setdefault("visible", {})
    cues = result.setdefault("difficulty_cues", {})
    for internal, geo in geometric_visible.items():
        if not bool(geo.get("visible")):
            visible[internal] = False
            cues[internal] = "not in current camera frustum"
        elif internal not in visible:
            visible[internal] = True


__all__ = ["ground_inventory"]
