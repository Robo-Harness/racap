"""LIBERO asset catalog: objects, fixtures, predicates, and scene types.

Used by the Environment Creator to validate and generate BDDL task files.
The LLM sees this catalog in its prompt so it knows what building blocks
are available.
"""

from __future__ import annotations

# ---- Problem classes (scene workspace types) ----
# The (define (problem X)) name in BDDL must match one of these.
PROBLEM_CLASSES: dict[str, dict] = {
    "LIBERO_Tabletop_Manipulation": {
        "workspace": "main_table",
        "workspace_type": "table",
        "description": "Main table workspace with cabinet and stove",
        "default_fixtures": ["wooden_cabinet", "flat_stove"],
    },
    "LIBERO_Kitchen_Tabletop_Manipulation": {
        "workspace": "kitchen_table",
        "workspace_type": "kitchen_table",
        "description": "Kitchen workspace",
        "default_fixtures": [],
    },
    "LIBERO_Living_Room_Tabletop_Manipulation": {
        "workspace": "living_room_table",
        "workspace_type": "living_room_table",
        "description": "Living room table workspace",
        "default_fixtures": [],
    },
    "LIBERO_Study_Tabletop_Manipulation": {
        "workspace": "study_table",
        "workspace_type": "study_table",
        "description": "Study desk workspace",
        "default_fixtures": [],
    },
}

# ---- Manipulable objects ----
# name -> description
OBJECTS: dict[str, str] = {
    # Bowls / plates / containers
    "akita_black_bowl": "black bowl (common target object)",
    "black_bowl": "black bowl (alternative model)",
    "red_akita_black_bowl": "red akita-style bowl",
    "bigger_akita_black_bowl": "larger akita-style bowl",
    "yellow_bowl": "yellow bowl",
    "white_bowl": "white bowl",
    "plate": "flat plate",
    "yellow_plate": "yellow plate",
    "glazed_rim_porcelain_ramekin": "small porcelain ramekin",
    "red_ramekin": "small red ramekin",
    "basket": "woven basket",
    "red_basket": "red woven basket",
    # Food items (HOPE dataset)
    "alphabet_soup": "alphabet soup can",
    "bbq_sauce": "BBQ sauce bottle",
    "butter": "butter box",
    "cream_cheese": "cream cheese box",
    "ketchup": "ketchup bottle",
    "milk": "milk carton",
    "orange_juice": "orange juice carton",
    "chocolate_pudding": "chocolate pudding cup",
    "cookies": "cookies box",
    "popcorn": "popcorn box",
    "macaroni_and_cheese": "macaroni and cheese box",
    "tomato_sauce": "tomato sauce can",
    "salad_dressing": "salad dressing bottle",
    # NOTE: cherries / corn / mayo HOPE classes are defined in
    # rats/third_party/LIBERO-PRO/.../hope_objects.py but the underlying
    # mesh/XML assets are not shipped in any LIBERO-PRO branch. Keep them
    # out of the catalog so the Task Proposer doesn't pick them and trip
    # env_creation_failed. Same applies to: rack, red_box, red_sticker,
    # blue_red_sticker, libero_mug_green, mayo (verified by instantiating
    # every OBJECTS_DICT entry — 8 broken classes total).
    # Kitchen items
    "moka_pot": "moka pot (stovetop coffee)",
    "yellow_moka_pot": "yellow moka pot (stovetop coffee)",
    "chefmate_8_frypan": "frying pan",
    "wine_bottle": "wine bottle",
    "white_bottle": "white bottle",
    "porcelain_mug": "porcelain mug",
    "white_porcelain_mug": "white porcelain mug",
    "white_yellow_mug": "white and yellow mug",
    "libero_mug_yellow": "yellow LIBERO mug",
    "red_coffee_mug": "red coffee mug",
    # Desk / study items
    # NOTE — LIBERO-PRO asset names for the three "book" entries do NOT
    # match the actual rendered texture. All three meshes share or use
    # near-identical off-white / pale-grey covers (verified by inspecting
    # turbosquid_objects/<name>/*.png — mean RGB in [200, 232] across
    # channels with no MuJoCo rgba override). Task proposers and downstream
    # perception (Molmo/SAM3) saw "yellow" / "black" in the language and
    # then "There are none" from the VLM, burning whole iterations on a
    # color the scene never actually contained. Describe them by visible
    # appearance, not by the upstream asset filename, so the proposer +
    # planner ground their prompts in what the camera will actually show.
    "black_book": (
        "small thin book (LIBERO asset filename 'black_book' is a "
        "misnomer; the rendered cover is light grey/off-white, NOT "
        "black — use a neutral / grey / book prompt for perception)"
    ),
    "yellow_book": (
        "small thin book (LIBERO asset filename 'yellow_book' is a "
        "misnomer; the rendered cover is off-white / pale beige, NOT "
        "yellow — use a neutral / grey / book prompt for perception)"
    ),
    "red_yellow_book": (
        "small thin book (LIBERO asset filename 'red_yellow_book' is a "
        "misnomer; this mesh shares yellow_book's off-white / pale cover "
        "texture, NOT red-and-yellow — use a neutral / grey / book "
        "prompt for perception)"
    ),
    "desk_caddy": "desk caddy / organizer",
    "yellow_desk_caddy": "yellow desk caddy",
    # Storage
    "wooden_tray": "wooden tray",
    "white_storage_box": "white storage box",
    "wooden_shelf": "wooden shelf (flat surface)",
    "wooden_two_layer_shelf": "two-layer wooden shelf",
    "brown_rack": "brown rack",
}

# ---- Pick-primitive reliability, per-object ----
# Source: docs/pick-primitive-benchmark.md (2026-04-20), 156-attempt sweep of
# Molmo + SAM3 + contact_graspnet + top-down execute. Numbers are raw SR of
# the NON-privileged pick pipeline — not the full task, just the lift.
#
# Why this lives here: the task proposer keeps burning iterations on
# butter / chocolate_pudding / cream_cheese / wine_bottle in novel-explore
# mode. The primitive itself cannot pick these (Molmo localization dead or
# top-down grasp unsuited for tall cylinders); no amount of policy-writer
# iteration rescues them. The proposer needs explicit evidence to avoid
# picking them as the target object.
#
# How to apply: task_proposer._pick_reliability_block() renders this for
# the novel-proposer prompt, and _validate_pick_reliability() hard-rejects
# proposals whose pick target is in UNRELIABLE_PICK_OBJECTS.
#
# Containers (bowls, plates, trays, baskets) are not pick targets in
# standard pick-and-place — they're destinations, not moved — so they're
# omitted from the unreliable list even if the benchmark measured bowl SR.
RELIABLE_PICK_OBJECTS: tuple[str, ...] = (
    "milk",              # 67% SR
    "alphabet_soup",     # 50%
    "orange_juice",      # 50%
    "akita_black_bowl",  # 27% as pick target (spatial suites)
    # Non-benchmarked but visually-distinct packaged items / mugs;
    # assumed OK until proven otherwise.
    "porcelain_mug",
    "white_yellow_mug",
    "red_coffee_mug",
    "cookies",
    "popcorn",
    "macaroni_and_cheese",
    "black_book",
    "yellow_book",
    "red_yellow_book",
    "moka_pot",
    "yellow_moka_pot",
    "white_porcelain_mug",
    "libero_mug_yellow",
)
MARGINAL_PICK_OBJECTS: tuple[str, ...] = (
    "ketchup",         # 33%
    "salad_dressing",  # 33%
    "tomato_sauce",    # 33%
    "bbq_sauce",       # 17%
)
# 0% SR in the benchmark — hard-reject when proposed as pick target.
UNRELIABLE_PICK_OBJECTS: tuple[str, ...] = (
    "butter",             # Molmo 0/6 hits
    "chocolate_pudding",  # Molmo 0/6 hits
    "cream_cheese",       # Molmo 7/12 miss + grasp 5/12 fail
    "wine_bottle",        # Grasp 0/12 — top-down primitive can't do cylinders
)

# ---- Fixtures (placed on workspace, may have articulated parts) ----
# name -> {description, regions: [sub-region names]}
FIXTURES: dict[str, dict] = {
    "wooden_cabinet": {
        "description": "Wooden cabinet with 3 drawers (top/middle/bottom)",
        "regions": ["top_region", "middle_region", "bottom_region", "top_side"],
        "predicates": ["Open", "Close"],
    },
    "flat_stove": {
        "description": "Flat stovetop with burner",
        "regions": ["cook_region"],
        "predicates": ["Turnon", "Turnoff"],
    },
    "microwave": {
        "description": "Microwave oven (openable door)",
        "regions": ["heating_region"],
        "predicates": ["Open", "Close"],
    },
    "wine_rack": {
        "description": "Wine rack / bottle holder",
        "regions": ["top_region"],
        "predicates": [],
    },
    "bowl_drainer": {
        "description": "Bowl drainer / dish rack",
        "regions": ["left_region", "right_region"],
        "predicates": [],
    },
    "white_cabinet": {
        "description": "White cabinet with drawers",
        "regions": ["top_region", "middle_region", "bottom_region", "top_side"],
        "predicates": ["Open", "Close"],
    },
    "yellow_cabinet": {
        "description": "Yellow cabinet with drawers (color variant)",
        "regions": ["top_region", "middle_region", "bottom_region", "top_side"],
        "predicates": ["Open", "Close"],
    },
    "short_cabinet": {
        "description": "Short cabinet with a door",
        "regions": ["top_region", "bottom_region"],
        "predicates": ["Open", "Close"],
    },
    "short_fridge": {
        "description": "Short fridge with an openable door",
        "regions": ["heating_region"],
        "predicates": ["Open", "Close"],
    },
    "slide_cabinet": {
        "description": "Cabinet with a sliding door",
        "regions": ["top_region", "bottom_region"],
        "predicates": ["Open", "Close"],
    },
    "yellow_stove": {
        "description": "Yellow stovetop with burner (color variant of flat_stove)",
        "regions": ["cook_region"],
        "predicates": ["Turnon", "Turnoff"],
    },
}

# ---- Goal predicates ----
PREDICATES: dict[str, dict] = {
    "On": {
        "arity": 2,
        "args": "(object, location_or_region)",
        "description": "Object is on top of location/region",
        "example": '(On akita_black_bowl_1 plate_1)',
    },
    "In": {
        "arity": 2,
        "args": "(object, container_region)",
        "description": "Object is inside a container region (e.g., cabinet drawer, microwave)",
        "example": '(In akita_black_bowl_1 wooden_cabinet_1_top_region)',
    },
    "Open": {
        "arity": 1,
        "args": "(fixture_region)",
        "description": "Fixture region is open (drawer/door)",
        "example": '(Open wooden_cabinet_1_top_region)',
    },
    "Close": {
        "arity": 1,
        "args": "(fixture_region)",
        "description": "Fixture region is closed",
        "example": '(Close wooden_cabinet_1_top_region)',
    },
    "Turnon": {
        "arity": 1,
        "args": "(fixture)",
        "description": "Fixture is turned on (stove burner)",
        "example": '(Turnon flat_stove_1)',
    },
    "Turnoff": {
        "arity": 1,
        "args": "(fixture)",
        "description": "Fixture is turned off",
        "example": '(Turnoff flat_stove_1)',
    },
    "Stack": {
        "arity": 2,
        "args": "(object_on_top, object_below)",
        "description": "First object stacked on second",
        "example": '(Stack akita_black_bowl_1 plate_1)',
    },
}

# ---- Compact text catalog for LLM prompts ----

def build_pick_reliability_text() -> str:
    """Compact prompt block: which objects the non-privileged pick
    primitive actually lifts, based on the 156-attempt benchmark.

    Rendered into `prompts/task_proposer_novel.txt` at the
    `{pick_reliability}` placeholder so the proposer chooses pick
    targets the primitive can handle.
    """
    lines = [
        "# Pick-Primitive Reliability (non-privileged pipeline)",
        "",
        "Source: docs/pick-primitive-benchmark.md — 156-attempt sweep of the "
        "Molmo + SAM3 + contact_graspnet + top-down pick pipeline on "
        "LIBERO-PRO. These are raw *lift* success rates, not full-task SR.",
        "",
        "## Reliable pick targets (≥25% lift SR, or untested-but-distinctive)",
    ]
    for name in RELIABLE_PICK_OBJECTS:
        desc = OBJECTS.get(name, name)
        lines.append(f"- {name}: {desc}")
    lines.append("")
    lines.append("## Marginal pick targets (17–33% lift SR) — avoid unless no alternative")
    for name in MARGINAL_PICK_OBJECTS:
        desc = OBJECTS.get(name, name)
        lines.append(f"- {name}: {desc}")
    lines.append("")
    lines.append(
        "## Unreliable pick targets (0% lift SR, benchmark-verified) — "
        "DO NOT propose as the object being picked"
    )
    for name in UNRELIABLE_PICK_OBJECTS:
        desc = OBJECTS.get(name, name)
        lines.append(f"- {name}: {desc}")
    lines.append("")
    lines.append(
        "Containers (bowl, plate, tray, basket, ramekin) are destinations "
        "and are not on the unreliable list — you may use them as the "
        "target location regardless of their own pick SR."
    )
    return "\n".join(lines)


def build_catalog_text() -> str:
    """Build a compact text catalog for inclusion in LLM prompts."""
    lines = []
    lines.append("# LIBERO Asset Catalog\n")

    lines.append("## Scene Types (problem class)")
    for name, info in PROBLEM_CLASSES.items():
        lines.append(f"- {name}: workspace={info['workspace']} ({info['description']})")

    lines.append("\n## Manipulable Objects")
    for name, desc in OBJECTS.items():
        lines.append(f"- {name}: {desc}")

    lines.append("\n## Fixtures (placed on workspace)")
    for name, info in FIXTURES.items():
        regions = ", ".join(info["regions"])
        preds = ", ".join(info["predicates"]) if info["predicates"] else "none"
        lines.append(f"- {name}: {info['description']}  regions=[{regions}]  predicates=[{preds}]")

    lines.append("\n## Goal Predicates")
    for name, info in PREDICATES.items():
        lines.append(f"- {name}{info['args']}: {info['description']}  e.g. {info['example']}")

    lines.append("\n## Naming Conventions")
    lines.append("- Object instances: {type}_{n} (e.g., akita_black_bowl_1, plate_1)")
    lines.append("- Fixture instances: {type}_{n} (e.g., wooden_cabinet_1, flat_stove_1)")
    lines.append("- Workspace regions: {workspace}_{region_name} (e.g., main_table_plate_region)")
    lines.append("- Fixture sub-regions: {fixture_instance}_{sub} (e.g., wooden_cabinet_1_top_region)")

    return "\n".join(lines)
