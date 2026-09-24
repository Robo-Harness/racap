"""MolmoSpaces task-type catalog: schemas, blocklists, and prompt rendering.

The novel-task proposer for MolmoSpaces (`_propose_novel_molmospaces_open` in
`agents/task_proposer.py`) needs a stable description of the typed-sampler
surface that the bridge actually exposes:

  - what's a valid `(task_type, target_object[, place_receptacle | joint])`
    triple,
  - which object categories each task type accepts,
  - which targets the runtime has historically failed to drive end-to-end.

The MolmoSpaces analogue of LIBERO's `agents/libero_catalog.py`. Kept
small: this file holds the *schema*, not the live inventory — the
inventory is queried per-house by `extract_molmospaces_scene_inventory`.
"""

from __future__ import annotations

from typing import Any

# Articulation categories shared by `open` and `close`. Mirrors
# `EXTENDED_ARTICULATION_TYPES_THOR` (lowercased) from
# `molmo_spaces.utils.constants.object_constants`. Hand-listed here to
# avoid importing molmo_spaces in the rats venv.
_ARTICULATED_OPEN_CATEGORIES: tuple[str, ...] = (
    "cabinet",
    "drawer",
    "oven",
    "dishwasher",
    "showerdoor",
    "fridge",
    "microwave",
    "toilet",
    "doorways",
    "doorway_double",
    "safe",
    "dresser",
    "desk",
    "shelving_unit",
    "side_table",
    "coffee_table",
    "laptop",
    "laundry_hamper",
)

# Task-type schemas -----------------------------------------------------
#
# Each entry describes how the upstream MolmoSpaces sampler is steered
# from a `set_task_from_spec` call:
#   - required:           keys the proposer's JSON output MUST include
#   - optional:           keys the proposer MAY include
#   - target_categories:  category whitelist for `target_object` (matched
#                         case-insensitively against the inventory entry's
#                         `category`). Empty tuple means "any category".
#   - permitted_for_stage: minimum curriculum stage at which this task type
#                         is allowed; matches `_curriculum_stage_molmospaces`.
#   - description:        short sentence the LLM sees in its prompt.
#
# Synced with `_TASK_TYPE_MAP` in
# `rats/envs/simulators/molmospaces_bridge.py`. If you add a task type
# here, also add the matching `_TASK_TYPE_MAP` entry on the bridge side.
TASK_TYPE_SCHEMAS: dict[str, dict[str, Any]] = {
    "pick": {
        "required": ("target_object_display_name",),
        "optional": (),
        "target_categories": (),  # any pickable; bridge filters via PICK_AND_PLACE_OBJECTS + grasp files
        "permitted_for_stage": 1,
        "description": (
            "Lift a small free-bodied object off its current support. "
            "target_object must be a pickable item (cup, mug, bowl, "
            "remote, etc.) that the bridge inventory marks "
            "has_grasp_file=true."
        ),
    },
    "pick_and_place": {
        "required": (
            "target_object_display_name",
            "place_receptacle_display_name",
        ),
        "optional": (),
        "target_categories": (),  # filtered downstream against PICK_AND_PLACE_OBJECTS
        "permitted_for_stage": 2,
        "description": (
            "Lift the target_object and place it on / in the named "
            "place_receptacle. place_receptacle must be present in the "
            "inventory's `receptacles` list."
        ),
    },
    "open": {
        "required": (
            "joint_object_display_name",
            "joint_index",
        ),
        "optional": (),
        "target_categories": _ARTICULATED_OPEN_CATEGORIES,
        "permitted_for_stage": 2,
        "description": (
            "Open an articulated joint of joint_object. joint_object must "
            "appear in the inventory's `articulations` list. joint_index "
            "must reference one of that articulation's `joints` entries."
        ),
    },
    "close": {
        "required": (
            "joint_object_display_name",
            "joint_index",
        ),
        "optional": (),
        "target_categories": _ARTICULATED_OPEN_CATEGORIES,  # same set
        "permitted_for_stage": 3,
        "description": (
            "Close an articulated joint of joint_object. The bridge starts "
            "the joint pre-opened to ~50% so closing has visible effect. "
            "Same category constraint as `open`."
        ),
    },
    "nav": {
        "required": ("target_object_display_name",),
        "optional": (),
        "target_categories": (),  # any visible top-level object
        "permitted_for_stage": 3,
        "description": (
            "Navigate the robot base to within manipulation range of "
            "target_object. No grasp / actuation; success when the object "
            "is in the workspace. Useful for cross-room exploration."
        ),
    },
}


# ----------------------------------------------------------------------
# Reliability blocklist
# ----------------------------------------------------------------------

# Categories whose MolmoSpaces lift / actuation pipeline has historically
# produced 0% success. Add entries here as the failure_memory accumulates
# evidence — DO NOT add speculatively. Mirror of LIBERO's
# UNRELIABLE_PICK_OBJECTS in `agents/libero_catalog.py`.
#
# Each entry is the lowercase category slug as it appears in the
# inventory's `pickables[].category` field. Substring match — `"butter"`
# rejects `"butter"` and `"butter_knife"`. Be specific.
MOLMOSPACES_UNRELIABLE_PICK_OBJECTS: frozenset[str] = frozenset({
    # populated empirically; intentionally empty at first integration
})


# ----------------------------------------------------------------------
# Oversized blocklist
# ----------------------------------------------------------------------

# Categories whose typical asset is wider than the Franka gripper
# (~8 cm jaw opening on the standard hand). The proposer should avoid
# proposing these as graspable targets — touching/pushing them is fine.
# Substring match against the inventory's lowercase ``category`` field,
# same convention as ``MOLMOSPACES_UNRELIABLE_PICK_OBJECTS``.
MOLMOSPACES_OVERSIZED_PICK_CATEGORIES: frozenset[str] = frozenset({
    "lettuce",
    "head_of_lettuce",
    "cabbage",
    "watermelon",
    "pumpkin",
    "pineapple",
    "squash",
    "broccoli",
    "cauliflower",
})

# Specific AI2-THOR / Objaverse asset UIDs that are visually labelled
# under a normally-graspable category (e.g. ``apple``) but whose actual
# mesh is too large for the Franka gripper. Exact-match against
# ``inventory[*]["asset_uid"]``. Add entries here as failure-memory
# accumulates evidence — keep it specific.
MOLMOSPACES_OVERSIZED_PICK_ASSET_UIDS: frozenset[str] = frozenset({
    # Empirically too big for the gripper despite the "apple" label —
    # iter001 of outputs/rats_playtime_gpt_0502 (placed-on-plate task
    # failed at the grasp stage).
    "Apple_24",
})


def is_oversized_pick_target(category: str, asset_uid: str | None = None) -> bool:
    """True iff the (category, asset_uid) pair is on either oversized list.

    Category check is substring (so ``head_of_lettuce`` catches
    ``lettuce``). Asset_uid check is exact match.
    """
    cat = (category or "").lower()
    for bad in MOLMOSPACES_OVERSIZED_PICK_CATEGORIES:
        if bad and bad in cat:
            return True
    if asset_uid and str(asset_uid) in MOLMOSPACES_OVERSIZED_PICK_ASSET_UIDS:
        return True
    return False


# ----------------------------------------------------------------------
# Prompt rendering
# ----------------------------------------------------------------------

def build_molmospaces_taskschema_text() -> str:
    """Render `TASK_TYPE_SCHEMAS` for inclusion in the proposer prompt.

    The proposer sees this once per call so it knows the exact JSON
    shape per task_type. Kept compact to leave token budget for the
    inventory + history blocks.
    """
    lines: list[str] = ["# MolmoSpaces Task-Type Schemas", ""]
    for task_type, info in TASK_TYPE_SCHEMAS.items():
        req = ", ".join(info["required"]) or "(none)"
        opt = ", ".join(info["optional"]) or "(none)"
        cats = info["target_categories"]
        cat_str = ", ".join(cats) if cats else "any inventory category"
        lines.append(f"## {task_type}")
        lines.append(f"- description: {info['description']}")
        lines.append(f"- required keys: {req}")
        if info["optional"]:
            lines.append(f"- optional keys: {opt}")
        lines.append(f"- permitted target categories: {cat_str}")
        lines.append(f"- minimum curriculum stage: {info['permitted_for_stage']}")
        lines.append("")
    if MOLMOSPACES_UNRELIABLE_PICK_OBJECTS:
        lines.append("# Unreliable pick categories (do NOT propose as target_object):")
        for cat in sorted(MOLMOSPACES_UNRELIABLE_PICK_OBJECTS):
            lines.append(f"- {cat}")
    return "\n".join(lines)


def is_blocklisted_pick_target(category: str) -> bool:
    """Substring-match against the unreliable list. Case-insensitive."""
    if not category:
        return False
    cat = category.lower()
    return any(bad in cat for bad in MOLMOSPACES_UNRELIABLE_PICK_OBJECTS)


def task_type_allowed_at_stage(task_type: str, stage: int) -> bool:
    """Curriculum gate: True iff task_type is permitted at this stage."""
    info = TASK_TYPE_SCHEMAS.get(task_type)
    if info is None:
        return False
    return stage >= int(info["permitted_for_stage"])


__all__ = [
    "TASK_TYPE_SCHEMAS",
    "MOLMOSPACES_UNRELIABLE_PICK_OBJECTS",
    "MOLMOSPACES_OVERSIZED_PICK_CATEGORIES",
    "MOLMOSPACES_OVERSIZED_PICK_ASSET_UIDS",
    "build_molmospaces_taskschema_text",
    "is_blocklisted_pick_target",
    "is_oversized_pick_target",
    "task_type_allowed_at_stage",
]
