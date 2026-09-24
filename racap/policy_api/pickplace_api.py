"""A hand-tuned `pickplace`, built from measurement rather than intuition.

This exists to answer "how well can this primitive surface actually do?", so
that the evolution loop has a ceiling to be compared against instead of only a
seed to beat. It uses no privileged state: the same primitives, the same
perception, the same lack of any pose handed to it. What it has that the seed
does not is a list of facts, each of which cost an experiment to learn.

**Ground one label at a time.** `localize_many([target])` was correct 5 times in
6 with a 6 mm median error. Adding the destination to the same call dropped it
to 3 in 6, and adding `list_objects()` to 2 in 5. Mutual exclusion only helps
when the names describe different objects, and "blue can" describes the target
just as well as "alphabet soup" does.

**Close the fingers below the top surface.** `grasp` aims at `top_z`, the
highest visible point, which on a bottle is the cap. Grasping there lifted 1 of
5 objects. Two centimetres lower lifted 4 of 5, and the result was flat from
2 cm to 5 cm, so the exact value does not matter much as long as it is not zero.

**A held object is not necessarily the right object.** `verify_grasp` reads the
finger opening, so it reports a firm hold on whatever was grasped. Every false
positive measured came from a grounding 0.2 m off: the gripper closed on a
neighbour, held it perfectly, and reported success.

**The centre is not always the contact.** Open containers are empty at the
centroid, and a book standing on its edge has to be taken across its thickness,
not along its face. The hand is aimed from the perceived footprint: hollow
clouds get ``rim``, elongated ones get ``pca_axis``, everything else starts
straight down.

**The surface height has to be measured next to the object, not under the
destination.** The old estimate took the lowest point of the destination's
cloud, which is the table only when the destination stands on the table. Told
to put a bowl *on top of the cabinet*, it returned the height of the cabinet
top and the grasp aimed a hand's width above the bowl. Probing a ring of pixels
around the object instead reads the surface the object actually stands on.

**A destination is a phrase, not a noun.** Half of libero_90's placements name
a part or a relation -- the bottom drawer of the cabinet, the front compartment
of the caddy, the right of the plate -- and the joint detector, which is built
to put one box on one object, is the wrong tool for them. Those go to the
single-label VLM detector, whose prompt is told to use spatial relations, and
relations of the form "<side> of <object>" are resolved geometrically from the
reference object and the camera's own axes.

**Release just above the surface, not at a fixed height.** `place` drops from
14 cm above the destination's top whatever it is carrying. That is a long fall
into a drawer and a hopeless one into a caddy slot. Measuring how far the
carried object hangs below the hand, and releasing so its base clears the
destination by 3 cm, turns a drop into a placement.
"""

from __future__ import annotations

import re

import numpy as np

from racap.backends.vlm import PerceptionUnavailableError
from racap.backends.llm import LLMError

from racap.contracts import GroundedTarget, SkillResult

# Measured: 1/5 objects lift at the visible top, 4/5 two centimetres below, and
# the curve is flat out to five. Chosen at the near end of the plateau so the
# fingers still straddle the body of a short object.
GRASP_DEPTH = 0.025
# Never take so much off a short object that the fingers reach for the table.
MAX_DEPTH_FRACTION = 0.35
# Keep the fingers above the table.
MIN_CLEARANCE = 0.008
# A single-hand tabletop object can legitimately stand much taller than the
# visible depth cloud suggests.  In particular, a near-vertical slab may give
# SAM/RGB-D only its top-facing pages, whose OBB is a centimetre thick even
# though the support is 10--20 cm below.  The surrounding ring is independent
# support evidence, so do not reject its modal plane merely because that
# incomplete OBB is short.  This bound still excludes the room floor beyond a
# table edge (roughly 40 cm below in LIBERO) and reflects the arm's graspable
# object scale rather than an object category.
MAX_GRASPABLE_SUPPORT_DROP = 0.30

# A negative crop identity now triggers full-scene VLM revision instead of
# either moving the wrong object or blindly vetoing the only proposal. Keep
# this switch for callers that explicitly require unknown evidence to match;
# an exact "different object" answer is always revised below.
VETO_ON_IDENTITY = False

MIN_POINTS = 60
MIN_EXTENT = 0.005
MAX_EXTENT = 1.00
# A grounding this far from the previous one is a different object, not a
# refinement of the same one.
# A failed contact may roll or tip the intended object, so retry grounding is
# allowed to move locally.  It must, however, stay near the *originally
# verified instance*.  Comparing only with the previous retry lets a sequence
# of individually small VLM errors walk across a row of look-alike objects.
MAX_GRASP_REGROUND_STEP_M = 0.06
MAX_GRASP_REGROUND_ANCHOR_M = 0.08
# One API call is one closed-loop manipulation decision, not an unbounded
# benchmark of every grasp primitive.  Failed contacts change the scene, and
# the visual agent is better placed to choose the next family from a fresh
# image.  This also keeps one call from consuming an entire simulator horizon.
MAX_GRASP_ATTEMPTS_PER_CALL = 3

# Never command the hand itself below this above the destination surface.
MIN_HAND_CLEARANCE = 0.02
# Height to cross the scene at, above the destination surface and whatever is
# hanging from the hand.
TRANSIT_CLEARANCE = 0.12
# Gap between the base of the carried object and the destination at release.
PLACE_MARGIN = 0.03
# Beyond this, a held object's apparent offset from the hand is a misdetection.
MAX_CARRY_OFFSET = 0.09
# A destination further than this from the object being placed is a grounding
# that has left the scene.
MAX_DESTINATION_M = 1.2
# Footprints longer than this relative to their width have an orientation that
# matters, both to grasp and to place.
ELONGATED = 2.2

# Words that make a destination a region or a relation rather than an object.
_REGION_WORDS = (
    "left",
    "right",
    "front",
    "back",
    "behind",
    "top",
    "bottom",
    "middle",
    "compartment",
    "drawer",
    "shelf",
    "side",
    "corner",
    "next to",
    "beside",
    "above",
    "under",
    "area",
    "space",
)
# "the right of the plate", "the front of the white mug".
_RELATION = re.compile(r"\b(left|right|front|back|behind)\s+(?:side\s+)?of\s+(?:the\s+)?(.+?)\s*$")
# "top of the cabinet", "the top surface of the shelf".
_ON_TOP = re.compile(r"\btop\s+(?:side\s+|surface\s+)?of\s+(?:the\s+)?(.+?)\s*$")
_ARTICLE = re.compile(r"^(?:the|a|an)\s+")


def _plausible(target) -> bool:
    """Geometric sanity: enough depth support and a believable footprint."""
    if target is None or target.confidence <= 0.0:
        return False
    if int(target.metadata.get("n_points", 0)) < MIN_POINTS:
        return False
    extent = target.metadata.get("extent") or [0.0, 0.0, 0.0]
    largest = max((float(v) for v in extent), default=0.0)
    return MIN_EXTENT < largest < MAX_EXTENT


def _ground(runtime, label: str):
    """One label, on its own, because adding others was measured to hurt."""
    return runtime.localize_many([label]).get(label)


def _ground_visible(runtime, label: str):
    """Re-find an already manipulated object without another VLM decision.

    Initial grounding needs the VLM to disambiguate instruction language.  Once
    the named object has been lifted or released, however, SAM3's text prompt
    is sufficient for measuring whether that same visible shape moved.  Using
    an independent local measurement here also avoids merely repeating the
    exact perception error that selected the object in the first place.
    """
    return runtime.localize(label, detector="sam3")


def _cloud(runtime, label: str) -> np.ndarray:
    try:
        points = np.asarray(runtime.object_points(label)).reshape(-1, 3)
    except Exception:
        return np.zeros((0, 3))
    return points[np.isfinite(points).all(axis=1)] if points.size else points


# ----------------------------------------------------------------- geometry


def _modal_height(values: np.ndarray, bin_m: float = 0.01) -> float | None:
    """The height most of the samples agree on.

    A ring of pixels around an object holds three populations: the surface it
    stands on, whatever else is standing nearby, and -- where the ring runs off
    the edge of the table -- the floor and the far wall, metres away. A median
    is only right when the first population is the majority, and around a plate
    at the edge of a table it is not. The largest cluster is, so the ring is
    binned and the fullest bin wins.
    """
    if values.size < 4:
        return None
    lo, hi = float(values.min()), float(values.max())
    bins = max(1, int(np.ceil((hi - lo) / bin_m)))
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi + 1e-6))
    peak = int(np.argmax(counts))
    if counts[peak] < 3:
        return None
    inside = values[(values >= edges[peak]) & (values <= edges[peak + 1])]
    return float(np.median(inside)) if inside.size else None


def _surface_height(runtime, target, fallback: float = 0.0) -> float:
    """Height of the surface the grounded object is standing on.

    Read from a ring of pixels just outside the object's box, because that is
    the only place in the image guaranteed to show what it stands on. The
    object's own cloud cannot supply this -- a depth camera looking down at a
    flat box sees only its lid -- and the destination's cloud only can when the
    destination happens to share the surface, which "on top of the cabinet"
    does not.
    """
    box = target.metadata.get("bbox") if target is not None else None
    top_z = float(target.metadata.get("top_z", 0.0)) if target is not None else 0.0
    extent = target.metadata.get("extent") if target is not None else None
    try:
        extent_z = abs(float(extent[2]))
    except (IndexError, TypeError, ValueError):
        extent_z = 0.0
    # The OBB bottom is a useful fallback and, more importantly, establishes
    # a physical gate for ring pixels.  A ring near a table edge can mostly
    # see the floor; accepting that mode makes a 2 cm package appear to hang
    # 40 cm below the fingers.  Supports may be a little below the noisy OBB,
    # but cannot be many object-heights below it.
    obb_bottom = float(target.pose[2]) - 0.5 * extent_z if target is not None else float(fallback)
    physical_fallback = obb_bottom if target is not None else float(fallback)
    minimum_support_z = (
        float(target.pose[2]) - max(MAX_GRASPABLE_SUPPORT_DROP, 1.5 * extent_z + 0.03)
        if target is not None
        else -float("inf")
    )
    if box and len(box) == 4:
        x0, y0, x1, y1 = (int(v) for v in box)
        pad = 8
        xs = np.linspace(x0 - pad, x1 + pad, 6).astype(int)
        ys = np.linspace(y0 - pad, y1 + pad, 6).astype(int)
        ring = (
            [(int(x), int(y0 - pad)) for x in xs]
            + [(int(x), int(y1 + pad)) for x in xs]
            + [(int(x0 - pad), int(y)) for y in ys]
            + [(int(x1 + pad), int(y)) for y in ys]
        )
        try:
            probed = np.asarray(runtime.probe_pixels(ring), dtype=float)
        except Exception:
            probed = np.zeros((0, 3))
        if probed.size:
            z = probed[:, 2]
            z = z[np.isfinite(z)]
            # Anything at or above the object's top is another object, and
            # anything far below it is the floor beyond the table's edge.
            z = z[(z < top_z - 0.002) & (z > top_z - 0.60)]
            modal = _modal_height(z)
            if modal is not None and modal >= minimum_support_z:
                return modal
    return physical_fallback


def _robust_object_top(
    target: GroundedTarget,
    points: np.ndarray,
    surface_z: float,
) -> tuple[float, float]:
    """Return a graspable top height and the raw detector maximum.

    Mask-to-depth grounding historically stored the maximum Z of every mask
    pixel as ``top_z``.  One cabinet/background pixel in a bowl mask can then
    put the commanded finger close 15--20 cm above a correctly grounded
    object.  The OBB centre/extent are already robust to that isolated ray, so
    they define a generous physical cap; inlier cloud percentiles refine the
    top below the cap.  This is geometry-only and does not assume an object
    class or simulator pose.
    """
    raw = float(target.metadata.get("top_z", target.pose[2]))
    extent = target.metadata.get("extent") or [0.0, 0.0, 0.0]
    try:
        extent_z = abs(float(extent[2]))
    except (IndexError, TypeError, ValueError):
        extent_z = 0.0
    centre_z = float(target.pose[2])
    cap = centre_z + max(0.75 * extent_z, 0.025) + 0.010
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)] if cloud.size else cloud
    inliers = (
        cloud[(cloud[:, 2] >= float(surface_z) - 0.012) & (cloud[:, 2] <= cap)]
        if cloud.shape[0]
        else cloud
    )
    measured = float(np.percentile(inliers[:, 2], 98)) if inliers.shape[0] >= MIN_POINTS else cap
    robust = min(raw, cap, measured + 0.004)
    return max(robust, float(surface_z) + MIN_CLEARANCE), raw


def _object_body_points(
    target: GroundedTarget,
    points: np.ndarray,
    surface_z: float,
    top_z: float,
) -> np.ndarray:
    """Remove depth rays inconsistent with the grounded object's OBB body."""
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)] if cloud.size else cloud
    if cloud.shape[0] < MIN_POINTS:
        return cloud
    centre = np.asarray(target.pose, dtype=float)
    extent = np.asarray(target.metadata.get("extent") or [0.0, 0.0, 0.0], dtype=float)
    xy_limit = np.maximum(0.035, 0.75 * np.abs(extent[:2]) + 0.012)
    keep = (
        np.all(np.abs(cloud[:, :2] - centre[:2]) <= xy_limit, axis=1)
        & (cloud[:, 2] >= float(surface_z) - 0.012)
        & (cloud[:, 2] <= float(top_z) + 0.012)
    )
    body = cloud[keep]
    return body if body.shape[0] >= MIN_POINTS else cloud


def _grasp_height(
    target,
    table_z: float,
    depth: float | None = None,
    *,
    top_z: float | None = None,
) -> float:
    """Where to close the fingers, as an absolute z.

    Below the visible top by a fixed depth, scaled down for objects too short
    to give that up, and never below the table. How tall the object is comes
    from its top relative to the table, not from ``extent``: that is an
    oriented bounding box whose largest axis is the width for anything flat,
    and scaling the depth by it sent the fingers 5 mm under the table for a
    cream cheese box 2 cm tall.
    """
    top_z = float(target.metadata.get("top_z", target.pose[2])) if top_z is None else float(top_z)
    height = max(top_z - table_z, 0.0)
    wanted = GRASP_DEPTH if depth is None else float(depth)
    depth = min(wanted, height * MAX_DEPTH_FRACTION) if height > 0 else wanted
    return max(top_z - depth, table_z + MIN_CLEARANCE)


def _low_grasp_height(table_z: float, top_z: float) -> float:
    """Low contact that remains below the top even for a thin package."""
    height = max(float(top_z) - float(table_z), 0.0)
    legacy = float(table_z) + float(np.clip(0.20 * height, 0.025, 0.06))
    below_top = float(top_z) - min(0.003, 0.20 * height)
    return max(float(table_z) + MIN_CLEARANCE, min(legacy, below_top))


def _edge_grasp_height(table_z: float, top_z: float) -> float:
    """Contact plane for a tilted pinch on a flush planar object."""
    return max(float(top_z) + 0.0005, float(table_z) + MIN_CLEARANCE)


def _lowered(target, height: float):
    """The same target, asking `grasp` to close at ``height``."""
    from dataclasses import replace

    return replace(target, metadata={**target.metadata, "top_z": height})


def _looks_hollow(points: np.ndarray) -> bool:
    """Whether the object is open at the top, from the depth of its middle.

    Bowls and mugs fail a centre grasp for a geometric reason: the centroid is
    air. The test that says so is not how many points sit in the middle -- a
    tin can seen from the side has a ring of wall points and a sparse lid, and
    a density test calls that hollow, which cost four grasps of tomato sauce
    that a plain top-down hand had been managing. It is how *deep* the middle
    is. Look into a bowl and the depth camera returns its inside floor,
    centimetres below the rim; look at a can and it returns the lid, level with
    it.
    """
    if points.shape[0] < MIN_POINTS:
        return False
    top_z = float(points[:, 2].max())
    centre = points[:, :2].mean(axis=0)
    radii = np.linalg.norm(points[:, :2] - centre, axis=1)
    outer = float(np.percentile(radii, 90))
    if outer < 0.025:
        return False
    core = points[radii <= 0.35 * outer]
    if core.shape[0] < 8:
        return False
    return top_z - float(np.median(core[:, 2])) > 0.02


def _footprint_aspect(points: np.ndarray) -> float:
    """Long over short spread of the footprint, 1.0 when it is round.

    A book standing on its edge has a footprint several times longer than it is
    wide, and the jaws have to close across the short direction. A fixed
    top-down hand closes along whichever world axis it was born pointing at,
    which for half the books in a scene is the wrong one.
    """
    if points.shape[0] < 12:
        return 1.0
    flat = points[:, :2] - points[:, :2].mean(axis=0)
    try:
        spread = np.linalg.svd(flat, compute_uv=False)
    except np.linalg.LinAlgError:
        return 1.0
    if spread.size < 2 or spread[1] <= 1e-6:
        return 1.0
    return float(spread[0] / spread[1])


def _attempts(
    points: np.ndarray,
    *,
    object_height: float | None = None,
) -> tuple[tuple[str, str], ...]:
    """Grasp attempts in the order the perceived shape argues for.

    Each is a strategy and where to close along the object's height. ``top``
    is a couple of centimetres under the visible top, which is right for
    anything solid. ``low`` is just above the surface, which is the only place
    a mug can be taken: it is hollow at the rim, too wide across the handle for
    the jaws, and narrow enough low down on the body to fit them.

    Order matters more than the list does. Leading with ``pca_axis`` on
    elongated objects looked obviously right -- close across a book's thickness
    rather than along its face -- and measured worse: on the caddy books it
    closed on nothing where a plain top-down hand had held them. A failed
    attempt is not free either, since it can push the object over before the
    next one runs, so the strategy measured to work leads and the clever one
    follows.
    """
    aspect = _footprint_aspect(points)
    hollow = _looks_hollow(points)
    if hollow and aspect < ELONGATED and object_height is not None and float(object_height) >= 0.13:
        # A tall open vessel is a genuinely 3-D grasping problem.  Its visual
        # centroid is empty air, its rim can be wider than the Franka jaws,
        # and a world-axis low pinch is easily redirected by the handle.  A
        # depth-conditioned 6-DoF proposal selects an actual side-wall contact
        # and was the only one of four independent contact families to retain
        # and place the tall mug.  Keep the analytic modes as fallbacks: the
        # learned service may have no admissible proposal, and shallow bowls
        # remain better served by the cheap rim grasp.
        return (
            ("graspnet", "top"),
            ("rim", "top"),
            ("top_down", "top"),
            ("top_down", "low"),
            ("pca_axis", "low"),
            ("affordance", "top"),
        )
    if (
        object_height is not None
        and 0.008 <= float(object_height) <= 0.024
        and aspect >= 1.55
        and not hollow
    ):
        # Thin packages are not automatically hard grasps.  A centred vertical
        # pinch is the least destructive probe and succeeds on ordinary boxes
        # that have enough side wall.  Edge scoops are recoveries for a truly
        # flush object: leading with one can push or rotate an otherwise easy
        # target and make every later grasp operate on a damaged scene.
        return (
            ("top_down", "top"),
            ("scoop_edge", "edge"),
            ("tilted_edge", "edge"),
            ("tilted_edge_reverse", "edge"),
            ("top_down", "edge"),
            ("pca_axis", "top"),
            ("rim", "top"),
            ("affordance", "top"),
        )
    # A thin elongated solid can have sparse / lower centre pixels and mimic a
    # cavity in a single depth view.  A rim grasp then lifts one edge and turns
    # a flat book into an upright object whose centre cannot fit below a short
    # containment volume.  Strong footprint elongation is positive evidence
    # for a solid planar object; reserve hollow-first for rounder vessels.
    if hollow and aspect < ELONGATED:
        # A mug is the hard case: hollow at the rim, wider than the jaws
        # across its handle, and grippable only low on the body across its
        # narrow direction, which is ``pca_axis`` low down. Putting that
        # ahead of the plain low grasp was measured and lost more than it
        # won, so it stays available and stays late.
        return (
            ("rim", "top"),
            ("top_down", "top"),
            ("top_down", "low"),
            ("pca_axis", "low"),
            ("affordance", "top"),
        )
    return (
        ("top_down", "top"),
        ("top_down", "low"),
        ("pca_axis", "top"),
        ("rim", "top"),
        ("affordance", "top"),
    )


# -------------------------------------------------------------- destination


_STRATEGIES = (
    "top_down",
    "rim",
    "pca_axis",
    "scoop_edge",
    "tilted_edge",
    "tilted_edge_reverse",
    "affordance",
    "graspnet",
    "wrist_closeloop",
)


def _ladder(
    grasp: str | list[str], default: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    """Put the caller's grasp choice first, keeping the rest as fallback.

    Accepts ``"rim"``, ``"rim@top"``, or a list of either. The height suffix
    is the part callers forget and the part that decides whether a mug is
    taken by its lip or across its body, so an unqualified strategy keeps the
    default height rather than failing.

    A preference rather than a replacement, because a caller naming one
    strategy is saying "try this first", not "and give up if it fails" -- an
    agent that asked for a single grasp got one attempt where the default
    would have made five, and turned a recoverable episode into a dead one.
    """
    items = [grasp] if isinstance(grasp, str) else list(grasp)
    ladder: list[tuple[str, str]] = []
    for item in list(items) + list(default):
        strategy, _, where = (
            str(item).partition("@") if isinstance(item, str) else (item[0], "@", item[1])
        )
        strategy = strategy.strip()
        where = where.strip() or "top"
        if (
            strategy in _STRATEGIES
            and where in ("top", "low", "edge")
            and (strategy, where) not in ladder
        ):
            ladder.append((strategy, where))
    return tuple(ladder) or (("top_down", "top"),)


def _nudge_vector(runtime, nudge: tuple[float, float]) -> np.ndarray:
    """Turn "a bit right and up in the picture" into a world displacement.

    A caller looking at a photograph can see that the object landed to the
    left of where it should have, and cannot see which way world +x points.
    Image directions are the ones it can actually name.
    """
    right_m, up_m = (float(v) for v in nudge)
    if abs(right_m) < 1e-6 and abs(up_m) < 1e-6:
        return np.zeros(2)
    try:
        axes = runtime.image_axes()
    except Exception:
        return np.zeros(2)
    right = np.asarray(axes["right"], dtype=float)[:2]
    down = np.asarray(axes["down"], dtype=float)[:2]
    for vector in (right, down):
        norm = float(np.linalg.norm(vector))
        if norm > 1e-6:
            vector /= norm
    return right * right_m - down * up_m


def _inward_feasible_nudge(
    centre_xy: np.ndarray,
    opening_polygon_xy: np.ndarray,
    outward_axis: np.ndarray,
    object_points: np.ndarray,
    *,
    wall_margin_m: float = 0.008,
    max_shift_m: float = 0.055,
) -> np.ndarray:
    """Move a drawer release deeper within its eroded feasible opening."""
    import cv2

    centre = np.asarray(centre_xy, dtype=float).reshape(2)
    polygon = np.asarray(opening_polygon_xy, dtype=float).reshape(-1, 2)
    outward = np.asarray(outward_axis, dtype=float).reshape(2)
    norm = float(np.linalg.norm(outward))
    cloud = np.asarray(object_points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if polygon.shape[0] < 3 or norm < 1e-6 or cloud.shape[0] < 8:
        return np.zeros(2)
    outward /= norm
    projection = cloud[:, :2] @ outward
    half_extent = 0.5 * float(np.percentile(projection, 95) - np.percentile(projection, 5))
    required_clearance = float(np.clip(half_extent + wall_margin_m, 0.012, 0.065))
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    best = centre.copy()
    for distance in np.linspace(0.0, float(max_shift_m), 12):
        candidate = centre - outward * float(distance)
        clearance = float(
            cv2.pointPolygonTest(contour, (float(candidate[0]), float(candidate[1])), True)
        )
        if clearance >= required_clearance:
            best = candidate
    return best - centre


def _drawer_inward_nudge(
    runtime,
    destination: GroundedTarget,
    object_points: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Find closure-safe depth from the clean handle/opening geometry."""
    if "drawer" not in destination.label.lower():
        return np.zeros(2), {}
    polygon = destination.metadata.get("region_polygon_xy")
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
        return np.zeros(2), {}
    try:
        from racap.policy_api.articulate_api import (
            _ground as _ground_fixture,
            _ground_handle,
            _refine_handle_contact,
            prismatic_axis_from_opening_boundary,
            prismatic_axis_from_opening_fixture,
            prismatic_axis_from_opening_probe,
        )

        lowered = destination.label.lower()
        parent_phrase = lowered.rsplit(" of ", 1)[-1].strip()
        part_phrase = lowered.rsplit(" of ", 1)[0].strip()
        fixture = _ground_fixture(runtime, parent_phrase)
        handle_trace: list[dict] = []
        handle = (
            _ground_handle(runtime, fixture, parent_phrase, part_phrase, handle_trace)
            if _plausible(fixture)
            else None
        )
        fitted = None
        source = ""
        handle_pose = None
        # Prefer the raw RGB-D inside the semantic handle box.  Its 2-D box is
        # often correct even when SAM leaks to the distant background and
        # corrupts the 3-D mask centroid.
        semantic_handle = _ground_fixture(runtime, f"handle of {part_phrase} of {parent_phrase}")
        handle_bbox = semantic_handle.metadata.get("bbox") if semantic_handle is not None else None
        probe_fit = None
        if handle_bbox:
            probed = runtime.probe_bbox(list(handle_bbox))
            probe_fit = prismatic_axis_from_opening_probe(
                np.asarray(probed["points"], dtype=float),
                np.asarray(polygon, dtype=float),
                expected_z=float(destination.metadata.get("top_z", destination.pose[2])),
            )
        if handle is not None:
            handle, _ = _refine_handle_contact(runtime, fixture, handle)
            fitted = prismatic_axis_from_opening_boundary(
                np.asarray(handle.pose[:2], dtype=float),
                np.asarray(polygon, dtype=float),
            )
            if fitted is not None:
                source = "handle_to_opening_boundary"
            handle_pose = [round(float(v), 4) for v in handle.pose[:2]]
        if fitted is None:
            opening_centre = np.mean(np.asarray(polygon, dtype=float), axis=0)
            fixture_distance = (
                float(np.linalg.norm(np.asarray(fixture.pose[:2], dtype=float) - opening_centre))
                if _plausible(fixture)
                else float("inf")
            )
            if 0.03 <= fixture_distance <= 0.50:
                fitted = prismatic_axis_from_opening_fixture(
                    np.asarray(fixture.pose[:2], dtype=float),
                    np.asarray(polygon, dtype=float),
                )
                if fitted is not None:
                    source = "fixture_to_opening_boundary"

        # The raw handle box is useful for finding the *orientation* of a
        # drawer rail, but not its sign: a large VLM box also sees the rear
        # cabinet wall, whose opening-edge normal is exactly opposite the
        # handle/front normal.  A previous implementation let whichever band
        # contained more depth pixels decide the sign and moved releases
        # toward the drawer front.  Require an independently grounded handle
        # or stationary fixture to orient it; otherwise the cavity centre is
        # the maximum-clearance, closure-safe choice.
        if fitted is not None and probe_fit is not None:
            signed_axis = np.asarray(fitted["axis"], dtype=float)
            probe_axis = np.asarray(probe_fit["axis"], dtype=float)
            if abs(float(signed_axis @ probe_axis)) < 0.70:
                fitted = None
                source = "probe_disagreed_with_signed_axis"
    except Exception as exc:
        return np.zeros(2), {"source": f"fit_failed:{type(exc).__name__}"}
    if fitted is None:
        return np.zeros(2), {
            "source": source or "opening_axis_ambiguous_centred",
            "handle_xy": handle_pose,
            "probe_axis": (
                None if probe_fit is None else [round(float(v), 4) for v in probe_fit["axis"]]
            ),
            "inward_shift_cm": 0.0,
        }
    nudge = _inward_feasible_nudge(
        np.asarray(destination.pose[:2], dtype=float),
        np.asarray(polygon, dtype=float),
        np.asarray(fitted["axis"], dtype=float),
        object_points,
    )
    return nudge, {
        "source": source,
        "handle_xy": handle_pose,
        "outward_axis": [round(float(v), 4) for v in fitted["axis"]],
        "inward_shift_cm": round(float(np.linalg.norm(nudge)) * 100, 1),
    }


def _is_region(phrase: str) -> bool:
    lowered = f" {phrase.lower()} "
    return any(f" {word} " in lowered or f" {word}s " in lowered for word in _REGION_WORDS)


def _relation(phrase: str) -> tuple[str, str] | None:
    """Split "<side> of <object>" into the side and the object it is beside."""
    match = _RELATION.search(_ARTICLE.sub("", phrase.strip().lower()))
    if match is None:
        return None
    return match.group(1), _ARTICLE.sub("", match.group(2).strip())


def _beside(runtime, side: str, reference: str, trace: list) -> GroundedTarget | None:
    """A free spot next to ``reference``, on the side the instruction names.

    Asking a detector to box "the right of the plate" returns the plate: the
    models are trained to find objects, and there is no object there. What the
    phrase means is a displacement from the plate, in the direction the scene's
    own viewpoint calls right -- which is a property of the camera, read here
    rather than hard-coded, so it survives a scene that mounts it elsewhere.
    """
    anchor = _ground(runtime, reference)
    if not _plausible(anchor):
        return None
    try:
        axes = runtime.image_axes()
    except Exception:
        return None

    right = np.asarray(axes.get("right") or [0.0, 0.0], dtype=float)
    down = np.asarray(axes.get("down") or [0.0, 0.0], dtype=float)
    direction = {
        "right": right,
        "left": -right,
        "front": down,
        "back": -down,
        "behind": -down,
    }.get(side)
    if direction is None or not np.isfinite(direction).all() or not direction.any():
        return None

    extent = anchor.metadata.get("extent") or [0.1, 0.1, 0.1]
    radius = 0.5 * max(float(extent[0]), float(extent[1]))
    step = float(np.clip(radius + 0.06, 0.10, 0.20))
    xy = np.asarray(anchor.pose[:2], dtype=float) + direction * step

    surface = _surface_height(runtime, anchor, fallback=float(anchor.pose[2]))
    trace.append(
        {
            "event": "destination_beside",
            "side": side,
            "reference": reference,
            "step": round(step, 3),
            "xy": [round(float(v), 3) for v in xy],
            "surface_z": round(surface, 4),
        }
    )
    return GroundedTarget(
        label=f"{side} of {reference}",
        pose=(float(xy[0]), float(xy[1]), surface),
        kind="region",
        confidence=float(anchor.confidence),
        metadata={
            "top_z": surface,
            "n_points": MIN_POINTS,
            "extent": [0.08, 0.08, 0.0],
            "synthetic": True,
        },
    )


def _in_reach(target, near) -> bool:
    """Whether a destination is somewhere the object could plausibly be put.

    Asked for "top of the cabinet" in a scene whose cabinet is a low shelf, the
    detector returned a point 1.5 m behind the robot and below the floor. It
    was a perfectly well-formed grounding -- enough points, believable extent
    -- and the arm spent the episode reaching for it. The object being moved is
    the reference: whatever the destination is, it is in the same room.
    """
    if target is None or near is None:
        return target is not None
    delta = np.asarray(target.pose, dtype=float) - np.asarray(near.pose, dtype=float)
    return bool(np.linalg.norm(delta) <= MAX_DESTINATION_M)


def _upper_surface(runtime, reference: str, trace: list) -> GroundedTarget | None:
    """The middle of the highest flat part of a piece of furniture.

    Asked to box "top of the cabinet" the detector draws the right rectangle
    and then grounds it wrong: the cabinet's top is dark and its outline runs
    into the kitchen units behind, so the mask jumps to the far wall and
    deprojects to a point 1.5 m behind the robot. The object itself grounds
    perfectly -- forty-five thousand points against six hundred -- and its top
    is simply the top of that cloud, which needs no second detection.
    """
    anchor = _ground(runtime, reference)
    if not _plausible(anchor):
        return None
    points = _cloud(runtime, reference)
    if points.shape[0] < MIN_POINTS:
        return None
    top_z = float(np.percentile(points[:, 2], 99))
    surface = points[points[:, 2] >= top_z - 0.02]
    if surface.shape[0] < 12:
        return None
    xy_points = np.asarray(surface[:, :2], dtype=float)
    low, high = np.percentile(xy_points, [4, 96], axis=0)
    robust = xy_points[np.all((xy_points >= low) & (xy_points <= high), axis=1)]
    if robust.shape[0] < 12:
        robust = xy_points
    try:
        import cv2

        polygon = cv2.convexHull(robust.astype("float32")).reshape(-1, 2).astype(float)
    except Exception:
        polygon = np.asarray(
            [
                [low[0], low[1]],
                [high[0], low[1]],
                [high[0], high[1]],
                [low[0], high[1]],
            ],
            dtype=float,
        )
    extent = np.maximum(high - low, 0.0)
    xy = robust.mean(axis=0)
    trace.append(
        {
            "event": "destination_top",
            "reference": reference,
            "top_z": round(top_z, 4),
            "xy": [round(float(v), 3) for v in xy],
            "surface_points": int(surface.shape[0]),
            "extent": [round(float(v), 4) for v in extent],
            "polygon_vertices": int(polygon.shape[0]),
        }
    )
    return GroundedTarget(
        label=f"top of {reference}",
        pose=(float(xy[0]), float(xy[1]), top_z),
        kind="region",
        confidence=float(anchor.confidence),
        metadata={
            "top_z": top_z,
            "n_points": int(surface.shape[0]),
            "extent": [float(extent[0]), float(extent[1]), 0.0],
            "synthetic": True,
            "region_source": "rgbd_top_band",
            "region_polygon_xy": polygon.tolist(),
            "region_area_pixels": int(surface.shape[0]),
            "region_confidence": float(min(1.0, 0.55 + 0.45 * surface.shape[0] / 300.0)),
        },
    )


# Destinations that are holes rather than surfaces, and so have an inside that
# can be found in the depth image.
_CONTAINER_WORDS = (
    "compartment",
    "drawer",
    "slot",
    "basket",
    "bin",
    "tray",
    "box",
    "container",
    "cubby",
)
# Dense cavity recovery is an automatic refinement only for phrases whose
# semantics already select a *local opening*.  A fallback VLM box around a
# broad basket/tray can contain the table behind it, and its deepest connected
# patch is then valid RGB-D geometry for the wrong thing.  Broad containers
# still support an explicit ``destination="inside"`` retry, where the caller
# has visual evidence that the outline grounding needs replacing.
_AUTO_CAVITY_WORDS = ("compartment", "drawer", "slot", "cubby")


def _inside_of(runtime, target, trace: list) -> GroundedTarget | None:
    """Find the opening of a container from the depth inside its box.

    The detector draws the right rectangle around a caddy compartment and then
    segments it badly -- eight hundred points where the neighbouring
    compartment got nine thousand -- and the centroid of a bad mask is
    wherever the mask went. The rectangle is the reliable part, so the inside
    is taken from the depth image directly: a compartment is a hole, its floor
    is the set of pixels in the box that come back furthest from the camera,
    and the middle of that hole is where the object has to be released.

    Refusing when the hole is shallow is what keeps this from firing on a
    tabletop, where the "far" pixels are simply the back of the box.
    """
    box = target.metadata.get("bbox")
    if not box or len(box) != 4:
        return None
    x0, y0, x1, y1 = (int(v) for v in box)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    # Prefer a dense crop.  A 13x13 set of unrelated samples estimates a
    # height, but it cannot distinguish an opening from the walls around it or
    # retain a boundary for the later placement check.
    dense = getattr(runtime, "probe_bbox", None)
    if callable(dense):
        try:
            region = _cavity_region(runtime, target, trace)
        except Exception as exc:
            trace.append({"event": "inside_region", "error": f"{type(exc).__name__}: {exc}"})
            region = None
        if region is not None:
            return region

    xs = np.linspace(x0 + 2, x1 - 2, 13).astype(int)
    ys = np.linspace(y0 + 2, y1 - 2, 13).astype(int)
    grid = [(int(u), int(v)) for v in ys for u in xs]
    try:
        probed = np.asarray(runtime.probe_pixels(grid), dtype=float)
    except Exception:
        return None
    probed = probed[np.isfinite(probed).all(axis=1)]
    if probed.shape[0] < 20:
        return None

    z = probed[:, 2]
    rim = float(np.percentile(z, 90))
    floor = float(np.percentile(z, 10))
    depth = rim - floor
    if not 0.015 < depth < 0.40:
        trace.append({"event": "inside", "rejected": "no hole", "depth": round(depth, 3)})
        return None

    inside = probed[z <= floor + 0.25 * depth]
    if inside.shape[0] < 6:
        return None
    xy = np.median(inside[:, :2], axis=0)
    trace.append(
        {
            "event": "inside",
            "rim_z": round(rim, 4),
            "floor_z": round(floor, 4),
            "xy": [round(float(v), 3) for v in xy],
            "samples": int(inside.shape[0]),
        }
    )
    return GroundedTarget(
        label=target.label,
        pose=(float(xy[0]), float(xy[1]), floor),
        kind="region",
        confidence=float(target.confidence),
        metadata={
            "top_z": rim,
            "floor_z": floor,
            "n_points": MIN_POINTS,
            "extent": [0.08, 0.08, depth],
            "synthetic": True,
            "bbox": list(box),
        },
    )


def _cavity_region(runtime, target, trace: list) -> GroundedTarget | None:
    """Recover a persistent free-space polygon from a container RGB-D crop.

    The semantic grounder only selects *which* compartment the instruction
    names.  Its mask is allowed to include the walls and outer shell.  This
    second stage finds a connected, nearly horizontal surface below the rim,
    retains its boundary, and aims at the point with the greatest clearance
    to that boundary.  No simulator object/site state is consulted.
    """
    import cv2

    box = target.metadata.get("bbox")
    if not box or len(box) != 4:
        return None
    sample = runtime.probe_bbox(list(box))
    points = np.asarray(sample.get("points"), dtype=float)
    pixels = np.asarray(sample.get("pixels"), dtype=int)
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[:2] != pixels.shape[:2]:
        return None

    valid = np.isfinite(points).all(axis=2)
    z = points[..., 2]
    rim = float(target.metadata.get("top_z", np.nan))
    # For an empty opening the VLM box can be excellent while SAM attaches to
    # the distant wall visible through it.  Its 3-D centroid/top then lies far
    # outside the scene.  The bbox is still a valid selector; estimate the rim
    # from its own dense depth crop instead of rejecting the opening before
    # cavity geometry gets a chance to validate it.
    if not _plausible(target) or not np.isfinite(rim):
        rim = float(np.nanpercentile(z[valid], 90))
    below = valid & (z < rim - 0.020) & (z > rim - 0.40)
    if int(np.count_nonzero(below)) < 60:
        return None

    # The visible interior is not necessarily a horizontal floor: a low camera
    # often sees the back wall and only a sliver of the bottom.  Segment all
    # connected pixels appreciably below the rim, then prefer a compact,
    # densely filled, bounded component close to the semantic grounding.
    mask = below.astype("uint8")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    number, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates: list[tuple[float, int, np.ndarray]] = []
    semantic_xy = np.asarray(target.pose[:2], dtype=float)
    vertical_tier = str(target.metadata.get("vertical_tier", ""))
    tier_over_parent_crop = bool(target.metadata.get("vertical_tier_scope") == "parent_crop")
    for component in range(1, number):
        area = int(stats[component, cv2.CC_STAT_AREA])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        if area < 80 or width < 6 or height < 6:
            continue
        pixel_aspect = max(width / max(height, 1), height / max(width, 1))
        if pixel_aspect > 6.0:
            continue
        component_mask = labels == component
        component_points = points[component_mask]
        extent_xy = np.ptp(component_points[:, :2], axis=0)
        if float(np.min(extent_xy)) < 0.012:
            continue
        world_aspect = float(np.max(extent_xy) / max(np.min(extent_xy), 1e-6))
        if world_aspect > 6.0:
            continue
        border = np.zeros_like(component_mask)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        border_fraction = float(np.count_nonzero(component_mask & border)) / max(
            float(2 * (width + height)), 1.0
        )
        fill = area / max(float(width * height), 1.0)
        centre_xy = np.median(component_points[:, :2], axis=0)
        semantic_distance = float(np.linalg.norm(centre_xy - semantic_xy))
        proximity = float(np.exp(-semantic_distance / 0.08))
        score = area * fill * fill * proximity * (1.0 - 0.75 * min(border_fraction, 1.0))
        if tier_over_parent_crop and vertical_tier in {"upper", "lower"}:
            component_rows = np.nonzero(component_mask)[0]
            row_mid = float(np.median(component_rows))
            crop_mid = 0.5 * float(component_mask.shape[0] - 1)
            matches_tier = row_mid <= crop_mid if vertical_tier == "upper" else row_mid >= crop_mid
            score *= 4.0 if matches_tier else 0.10
        candidates.append((score, area, component_mask))
    if not candidates:
        return None

    _, area, component = max(candidates, key=lambda row: row[0])
    component_points = points[component]
    # The lowest stable portion is the support plane.  A percentile is robust
    # to the visible back wall while avoiding a handful of table pixels.
    floor = float(np.percentile(component_points[:, 2], 15))
    depth = rim - floor
    if not 0.015 < depth < 0.40:
        return None

    contours, _ = cv2.findContours(
        component.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    # OpenCV contour coordinates are local (column,row) within the crop.
    boundary_world = points[contour[:, 1], contour[:, 0], :2]
    boundary_pixels = pixels[contour[:, 1], contour[:, 0]]
    finite_boundary = np.isfinite(boundary_world).all(axis=1)
    boundary_world = boundary_world[finite_boundary]
    boundary_pixels = boundary_pixels[finite_boundary]
    if boundary_world.shape[0] < 3:
        return None
    # One depth-edge pixel can deproject metres behind its neighbours; if it
    # is a simplified contour corner, a small compartment becomes a room-sized
    # hull.  Repair only locally inconsistent corners from nearby pixels in the
    # same connected cavity.  Global percentile clipping was deliberately
    # avoided because narrow perspective edges may contain fewer than 2% of
    # the component's pixels yet still be real opening boundary.
    repaired_boundary = boundary_world.copy()
    for index, (column, row) in enumerate(contour[finite_boundary]):
        row0, row1 = max(0, int(row) - 2), min(points.shape[0], int(row) + 3)
        col0, col1 = max(0, int(column) - 2), min(points.shape[1], int(column) + 3)
        local_mask = component[row0:row1, col0:col1]
        local_world = points[row0:row1, col0:col1, :2][local_mask]
        local_world = local_world[np.isfinite(local_world).all(axis=1)]
        if local_world.shape[0] < 4:
            continue
        local_median = np.median(local_world, axis=0)
        local_scale = float(np.median(np.linalg.norm(local_world - local_median, axis=1)))
        consistency_limit = max(0.025, 6.0 * local_scale)
        if float(np.linalg.norm(repaired_boundary[index] - local_median)) > consistency_limit:
            repaired_boundary[index] = local_median

    # A depth discontinuity can occupy the whole local 5x5 neighbourhood, so
    # the local median above may agree with the bad point.  The boundary still
    # exposes that failure geometrically: two adjacent image-contour edges
    # suddenly deproject by tens of centimetres per pixel.  Repair an isolated
    # spike from its two contour neighbours.  Normal long straight contour
    # segments are preserved because the test is normalised by their image
    # length rather than using world edge length alone.
    if repaired_boundary.shape[0] >= 4:
        previous_world = np.roll(repaired_boundary, 1, axis=0)
        next_world = np.roll(repaired_boundary, -1, axis=0)
        previous_pixels = np.roll(boundary_pixels, 1, axis=0)
        next_pixels = np.roll(boundary_pixels, -1, axis=0)
        incoming_rate = np.linalg.norm(repaired_boundary - previous_world, axis=1) / np.maximum(
            np.linalg.norm(boundary_pixels - previous_pixels, axis=1), 1.0
        )
        outgoing_rate = np.linalg.norm(next_world - repaired_boundary, axis=1) / np.maximum(
            np.linalg.norm(next_pixels - boundary_pixels, axis=1), 1.0
        )
        edge_rates = np.concatenate([incoming_rate, outgoing_rate])
        finite_rates = edge_rates[np.isfinite(edge_rates)]
        typical_rate = float(np.median(finite_rates)) if finite_rates.size else 0.0
        jump_rate = max(0.015, 12.0 * typical_rate)
        neighbour_rate = np.linalg.norm(next_world - previous_world, axis=1) / np.maximum(
            np.linalg.norm(next_pixels - previous_pixels, axis=1), 1.0
        )
        isolated = (
            (incoming_rate > jump_rate)
            & (outgoing_rate > jump_rate)
            & (neighbour_rate <= jump_rate)
        )
        repaired_boundary[isolated] = (previous_world[isolated] + next_world[isolated]) / 2.0

    floor_band = component & (np.abs(z - floor) <= 0.030)
    if int(np.count_nonzero(floor_band)) < 40:
        floor_band = component
    repaired_boundary, planar_repair = _planar_boundary_repair(
        points[floor_band, :2],
        pixels[floor_band],
        repaired_boundary,
        boundary_pixels,
    )
    # An apparently dark cavity can deproject entirely onto the table visible
    # through/behind it. Then the whole "floor" is a coherent but wrong plane,
    # so RANSAC cannot call it an outlier. The visible rim is at the container's
    # physical depth and provides an independent image-to-world scale for the
    # same 2-D opening contour.
    if not planar_repair.get("used"):
        rim_band = valid & (np.abs(z - rim) <= 0.025)
        rim_boundary, rim_repair = _planar_boundary_repair(
            points[rim_band, :2],
            pixels[rim_band],
            repaired_boundary,
            boundary_pixels,
        )
        planar_repair = {
            **planar_repair,
            "floor_fit": dict(planar_repair),
            "rim_fit": dict(rim_repair),
            "source": "rim_plane" if rim_repair.get("used") else "floor_plane",
            "used": bool(rim_repair.get("used")),
        }
        if rim_repair.get("used"):
            repaired_boundary = rim_boundary
    if planar_repair.get("used"):
        trace.append({"event": "cavity_planar_repair", **planar_repair})

    # Finally compare the contour with the *bulk* of the same connected
    # component.  A valid narrow perspective edge may contain very few pixels,
    # so ordinary percentile clipping is too aggressive.  We only discard a
    # point when it forms a catastrophic thin spur: beyond the 2--98% bulk box
    # by at least half of that box's own span (and at least 2.5 cm).  This keeps
    # real boundaries while preventing one bad depth ray from turning a
    # 12-centimetre compartment into a 60-centimetre feasible region.
    bulk_low, bulk_high = np.percentile(component_points[:, :2], [2, 98], axis=0)
    bulk_span = np.maximum(bulk_high - bulk_low, 0.0)
    bulk_margin = np.maximum(0.025, 0.5 * bulk_span)
    bulk_consistent = np.all(
        (repaired_boundary >= bulk_low - bulk_margin)
        & (repaired_boundary <= bulk_high + bulk_margin),
        axis=1,
    )
    if int(np.count_nonzero(bulk_consistent)) >= 3:
        repaired_boundary = repaired_boundary[bulk_consistent]
    polygon_xy = cv2.convexHull(repaired_boundary.astype("float32")).reshape(-1, 2).astype(float)

    # Maximise clearance in the *world table plane*.  Pixel distance is
    # perspective-biased: on t80 its apparent centre was the visible back wall
    # of the compartment, six centimetres behind the actual free-space centre.
    contour_xy = polygon_xy.astype("float32").reshape(-1, 1, 2)
    grid_x = np.linspace(float(polygon_xy[:, 0].min()), float(polygon_xy[:, 0].max()), 45)
    grid_y = np.linspace(float(polygon_xy[:, 1].min()), float(polygon_xy[:, 1].max()), 45)
    clearance_candidates: list[tuple[float, float, float]] = []
    for world_x in grid_x:
        for world_y in grid_y:
            signed = float(cv2.pointPolygonTest(contour_xy, (float(world_x), float(world_y)), True))
            if signed >= 0.0:
                clearance_candidates.append((signed, float(world_x), float(world_y)))
    if not clearance_candidates:
        return None
    max_clearance = max(row[0] for row in clearance_candidates)
    plateau = [row for row in clearance_candidates if row[0] >= 0.96 * max_clearance]
    _, safe_x, safe_y = min(
        plateau,
        key=lambda row: float(np.linalg.norm(np.asarray(row[1:]) - semantic_xy)),
    )
    safe = np.asarray([safe_x, safe_y, floor], dtype=float)
    local_world = points[..., :2].reshape(-1, 2)
    local_pixels = pixels.reshape(-1, 2)
    finite_local = np.isfinite(local_world).all(axis=1)
    nearest = int(np.argmin(np.linalg.norm(local_world[finite_local] - safe[:2], axis=1)))
    safe_pixel = local_pixels[finite_local][nearest]
    clearance_m = float(max_clearance)
    extent_xy = np.ptp(polygon_xy, axis=0)
    confidence = float(
        np.clip(
            0.45 + 0.35 * min(area / 500.0, 1.0) + 0.20 * min(clearance_m / 0.04, 1.0), 0.0, 1.0
        )
    )
    metadata = {
        "top_z": rim,
        "floor_z": floor,
        "n_points": int(area),
        "extent": [float(extent_xy[0]), float(extent_xy[1]), depth],
        "synthetic": True,
        "bbox": list(sample.get("bbox", box)),
        "region_source": "rgbd_cavity",
        "region_polygon_xy": [[round(float(v), 5) for v in point] for point in polygon_xy],
        "region_boundary_pixels": [[int(v) for v in point] for point in boundary_pixels],
        "region_safe_pixel": [int(v) for v in safe_pixel],
        "region_area_pixels": int(area),
        "region_clearance_m": clearance_m,
        "region_confidence": confidence,
        "region_planar_repair": planar_repair,
    }
    trace.append(
        {
            "event": "inside_region",
            "source": "rgbd_cavity",
            "rim_z": round(rim, 4),
            "floor_z": round(floor, 4),
            "depth": round(depth, 4),
            "area_pixels": int(area),
            "clearance_cm": round(clearance_m * 100, 1),
            "safe_pixel": metadata["region_safe_pixel"],
            "safe_xy": [round(float(v), 3) for v in safe[:2]],
            "confidence": round(confidence, 2),
        }
    )
    return GroundedTarget(
        label=target.label,
        pose=(float(safe[0]), float(safe[1]), floor),
        kind="region",
        confidence=min(float(target.confidence), confidence),
        metadata=metadata,
    )


def _ground_destination(
    runtime,
    phrase: str,
    trace: list,
    near=None,
    mode: str = "auto",
    inside_validator=None,
) -> GroundedTarget | None:
    """Resolve where to release, from the destination as the instruction words it.

    Four kinds of destination, because they fail differently. A plain noun goes
    to the joint detector, the most accurate thing available for objects. A
    relation is arithmetic on a reference object. A top surface is the upper
    slice of the reference object's own cloud. Everything else -- parts,
    drawers, compartments -- goes to the single-label VLM detector, whose
    prompt asks it to use spatial relations, and which is therefore the only
    one that can tell the top drawer from the bottom one.

    ``mode`` forces the choice. The words are usually enough to pick, and when
    they are not the mistake is invisible from in here and obvious from
    outside: "the cabinet" means the shelf's top surface in one scene and its
    interior in another, and only something that has seen the scene can say
    which.
    """
    if mode in ("auto", "relation"):
        relation = _relation(phrase)
        if relation is not None:
            beside = _beside(runtime, relation[0], relation[1], trace)
            if beside is not None and _in_reach(beside, near):
                return beside

    if mode in ("auto", "top"):
        stripped = _ARTICLE.sub("", phrase.strip().lower())
        on_top = _ON_TOP.search(stripped)
        reference = on_top.group(1) if on_top else (stripped if mode == "top" else None)
        if reference:
            surface = _upper_surface(runtime, _ARTICLE.sub("", reference), trace)
            if surface is not None and _in_reach(surface, near):
                return surface

    # Two detectors, tried in the order the phrase argues for. The joint
    # detector is the most accurate thing available for a plain object; the
    # single-label one is the only thing that can tell the top drawer from the
    # bottom, because its prompt is told to use spatial relations. Whichever
    # runs first, the other is the fallback: each returns nothing often enough
    # on furniture to be worth a second opinion.
    if mode == "object":
        order = ("object",)
    elif mode in ("region", "inside"):
        order = ("region",)
    elif _is_region(phrase):
        order = ("region", "object")
    else:
        order = ("object", "region")
    for source in order:
        grounding_phrases = [phrase]
        lowered_phrase = phrase.lower()
        if source == "region" and mode == "inside" and "shelf" in lowered_phrase:
            tier = "lower" if re.search(r"\b(?:under|lower|bottom)\b", lowered_phrase) else "upper"
            grounding_phrases.append(
                f"empty usable interior of the {tier} open compartment of {phrase}"
            )
        for grounding_phrase in grounding_phrases:
            rejected: list[GroundedTarget] = []
            target = None
            # Region phrases are the failure-prone case: a VLM can ground a
            # semantically related background fixture or a nearby object.  An
            # independent full-scene verifier may veto that proposal, after
            # which the rejected box is shown back to the grounder.  Plain
            # object destinations retain the cheaper established path.
            max_candidates = 3 if source == "region" else 1
            for candidate_attempt in range(1, max_candidates + 1):
                try:
                    if source == "region":
                        feedback = getattr(runtime, "localize_with_feedback", None)
                        if rejected and callable(feedback):
                            candidate = feedback(grounding_phrase, rejected)
                            grounding_mode = "vlm_negative_feedback"
                        else:
                            candidate = runtime.localize(grounding_phrase, detector="vlm_bbox_sam3")
                            grounding_mode = "vlm_bbox_sam3"
                    else:
                        candidate = _ground(runtime, grounding_phrase)
                        grounding_mode = "joint"
                except PerceptionUnavailableError:
                    raise
                except Exception as exc:
                    trace.append(
                        {
                            "event": "destination",
                            "phrase": phrase,
                            "grounding_phrase": grounding_phrase,
                            "source": source,
                            "candidate_attempt": candidate_attempt,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    break

                verifier = getattr(runtime, "verify_destination_grounding", None)
                bbox = None if candidate is None else candidate.metadata.get("bbox")
                if source != "region" or not callable(verifier):
                    target = candidate
                    break
                if not bbox:
                    # A negative-feedback grounder may explicitly decline the
                    # rejected proposal by returning an empty sentinel. That
                    # is not a usable final grounding and must not terminate
                    # the candidate ladder before the next visual proposal.
                    trace.append(
                        {
                            "event": "destination_identity",
                            "grounding_phrase": grounding_phrase,
                            "candidate_attempt": candidate_attempt,
                            "mode": grounding_mode,
                            "answer": "invalid_candidate_without_bbox",
                            "matches": False,
                            "bbox": [],
                        }
                    )
                    if candidate is not None:
                        rejected.append(candidate)
                    continue
                try:
                    verdict = verifier(grounding_phrase, target=candidate)
                except (PerceptionUnavailableError, LLMError):
                    raise
                except Exception as exc:
                    verdict = {"ok": False, "answer": f"verify_failed:{type(exc).__name__}"}
                trace.append(
                    {
                        "event": "destination_identity",
                        "grounding_phrase": grounding_phrase,
                        "candidate_attempt": candidate_attempt,
                        "mode": grounding_mode,
                        "answer": verdict.get("answer", ""),
                        "matches": bool(verdict.get("ok")),
                        "bbox": [int(v) for v in bbox],
                    }
                )
                if verdict.get("ok"):
                    if mode == "inside":
                        # Empty openings often inherit background depth, so
                        # plausibility of the SAM centroid alone is not enough.
                        # Validate the dense cavity geometry now.  A visually
                        # plausible but unreachable background shelf is a
                        # rejected grounding and should trigger another VLM
                        # proposal, not merely fail after the retry loop.
                        geometric_candidate = _inside_of(runtime, candidate, trace)
                        geometry_ok = geometric_candidate is not None and _in_reach(
                            geometric_candidate, near
                        )
                        trace.append(
                            {
                                "event": "destination_geometry",
                                "grounding_phrase": grounding_phrase,
                                "candidate_attempt": candidate_attempt,
                                "usable": geometry_ok,
                                "pose": (
                                    None
                                    if geometric_candidate is None
                                    else [round(float(v), 3) for v in geometric_candidate.pose]
                                ),
                            }
                        )
                        if geometry_ok and inside_validator is not None:
                            try:
                                accepted, reason = inside_validator(geometric_candidate)
                            except Exception as exc:
                                accepted, reason = False, (f"validator_error:{type(exc).__name__}")
                            trace.append(
                                {
                                    "event": "destination_configuration_space",
                                    "grounding_phrase": grounding_phrase,
                                    "candidate_attempt": candidate_attempt,
                                    "usable": bool(accepted),
                                    "reason": str(reason),
                                }
                            )
                            geometry_ok = bool(accepted)
                            if not geometry_ok:
                                # The semantic referent may be right while its
                                # tight 2-D box clips the cavity floor or one
                                # rim. Before telling the VLM to choose a
                                # different referent, expand the *same verified
                                # box* and recompute RGB-D free space. This
                                # separates semantic correction from geometric
                                # crop completion.
                                values = candidate.metadata.get("bbox")
                                if values and len(values) >= 4:
                                    x0, y0, x1, y1 = (int(value) for value in values[:4])
                                    pad = int(
                                        np.clip(
                                            round(0.20 * max(x1 - x0, y1 - y0)),
                                            12,
                                            28,
                                        )
                                    )
                                    expanded = GroundedTarget(
                                        label=candidate.label,
                                        pose=candidate.pose,
                                        kind=candidate.kind,
                                        confidence=candidate.confidence,
                                        metadata={
                                            **candidate.metadata,
                                            "bbox": [
                                                x0 - pad,
                                                y0 - pad,
                                                x1 + pad,
                                                y1 + pad,
                                            ],
                                            "expanded_from_bbox": [
                                                x0,
                                                y0,
                                                x1,
                                                y1,
                                            ],
                                        },
                                    )
                                    expanded_geometry = _inside_of(runtime, expanded, trace)
                                    expanded_ok = bool(
                                        expanded_geometry is not None
                                        and _in_reach(expanded_geometry, near)
                                    )
                                    expanded_reason = "no_expanded_cavity"
                                    if expanded_ok:
                                        try:
                                            expanded_ok, expanded_reason = inside_validator(
                                                expanded_geometry
                                            )
                                        except Exception as exc:
                                            expanded_ok, expanded_reason = (
                                                False,
                                                f"validator_error:{type(exc).__name__}",
                                            )
                                    trace.append(
                                        {
                                            "event": ("destination_expanded_geometry"),
                                            "grounding_phrase": grounding_phrase,
                                            "candidate_attempt": candidate_attempt,
                                            "usable": bool(expanded_ok),
                                            "reason": str(expanded_reason),
                                            "bbox": expanded.metadata["bbox"],
                                        }
                                    )
                                    if expanded_ok:
                                        return expanded_geometry
                        if geometry_ok:
                            return geometric_candidate
                        rejected.append(candidate)
                        continue
                    target = candidate
                    break
                rejected.append(candidate)
            if target is None:
                continue
            # In inside mode the semantic box is only a 2-D selector. Recover
            # and validate its dense RGB-D cavity even when SAM's centroid hit
            # a background surface and is itself implausible.
            inside = None
            if source == "region" and mode == "inside" and target is not None:
                inside = _inside_of(runtime, target, trace)
                if inside is not None and _in_reach(inside, near):
                    if inside_validator is None:
                        return inside
                    try:
                        accepted, reason = inside_validator(inside)
                    except Exception as exc:
                        accepted, reason = False, (f"validator_error:{type(exc).__name__}")
                    trace.append(
                        {
                            "event": "destination_configuration_space",
                            "grounding_phrase": grounding_phrase,
                            "candidate_attempt": 1,
                            "usable": bool(accepted),
                            "reason": str(reason),
                        }
                    )
                    if accepted:
                        return inside
            usable = _plausible(target) and _in_reach(target, near)
            trace.append(
                {
                    "event": "destination",
                    "phrase": phrase,
                    "grounding_phrase": grounding_phrase,
                    "source": source,
                    "usable": usable,
                    "pose": None if target is None else [round(float(v), 3) for v in target.pose],
                }
            )
            if not usable:
                continue
            # Preserve an explicit cavity whenever the crop contains one.  The old
            # sparse inside estimator was only safe on request; the dense version
            # rejects non-flat / unbounded patches and therefore can refine a
            # semantic region without replacing good object grounding.
            words = set(re.findall(r"[a-z]+", phrase.lower()))
            wants_local_cavity = bool(words.intersection(_AUTO_CAVITY_WORDS))
            if mode != "inside" and source == "region" and wants_local_cavity:
                inside = _inside_of(runtime, target, trace)
                if inside is not None and _in_reach(inside, near):
                    return inside
            # In inside mode, a semantic box is not yet a usable free-space
            # destination. Try the next visually explicit paraphrase rather
            # than returning the furniture shell to the motion planner.
            if mode == "inside" and len(grounding_phrases) > 1:
                continue
            return target
    return None


def _planar_boundary_repair(
    component_points: np.ndarray,
    component_pixels: np.ndarray,
    boundary_world: np.ndarray,
    boundary_pixels: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Repair a depth-stretched cavity contour with a robust local plane fit.

    A single invalid depth ray can map two neighbouring boundary pixels tens
    of centimetres apart.  The cavity floor is locally planar, so a robust
    affine image-to-world fit over all floor pixels supplies an independent
    metric scale.  It is used only when the raw contour is catastrophically
    larger, leaving ordinary perspective geometry untouched.
    """
    import cv2

    world = np.asarray(component_points, dtype=float).reshape(-1, 2)
    pixels = np.asarray(component_pixels, dtype=float).reshape(-1, 2)
    measured = np.asarray(boundary_world, dtype=float).reshape(-1, 2)
    contour_pixels = np.asarray(boundary_pixels, dtype=float).reshape(-1, 2)
    finite = np.isfinite(world).all(axis=1) & np.isfinite(pixels).all(axis=1)
    if int(np.count_nonzero(finite)) < 40 or measured.shape[0] < 3:
        return measured, {"used": False, "reason": "insufficient_support"}
    try:
        affine, inliers = cv2.estimateAffine2D(
            pixels[finite].astype("float32"),
            world[finite].astype("float32"),
            method=cv2.RANSAC,
            ransacReprojThreshold=0.008,
            maxIters=2500,
            confidence=0.995,
            refineIters=15,
        )
    except cv2.error:
        affine, inliers = None, None
    if affine is None or inliers is None or int(np.count_nonzero(inliers)) < 30:
        return measured, {"used": False, "reason": "no_stable_plane"}
    predicted = (
        cv2.transform(
            contour_pixels.astype("float32").reshape(-1, 1, 2),
            np.asarray(affine, dtype="float32"),
        )
        .reshape(-1, 2)
        .astype(float)
    )
    measured_span = float(np.max(np.ptp(measured, axis=0)))
    predicted_span = float(np.max(np.ptp(predicted, axis=0)))
    inlier_fraction = float(np.mean(np.asarray(inliers).reshape(-1) > 0))
    catastrophic = bool(
        0.025 <= predicted_span <= 0.40
        # The affine plane is a last-resort repair, not a replacement for the
        # measured contour. Perspective and partially visible cavity walls can
        # make a valid opening disagree with this fit by 50--80%; accepting
        # those disagreements moved otherwise correct caddy sites outside the
        # native compartment. Require an unambiguous factor-of-two failure.
        and measured_span > 2.0 * predicted_span
    )
    return (predicted if catastrophic else measured), {
        "used": catastrophic,
        "measured_span_m": measured_span,
        "fitted_span_m": predicted_span,
        "inlier_fraction": inlier_fraction,
    }


def _release_band(runtime, destination) -> tuple[float, float]:
    """Highest and lowest surface the destination offers, as ``(rim, floor)``.

    A destination is rarely a single height. A basket is a rim at the top and a
    floor 12 cm below it; a caddy compartment is the same shape an inch across;
    a plate is both at once. Approaching to the rim leaves a book standing on
    the edge of the slot, and aiming straight at the floor drives the hand
    through the wall on the way. Knowing both lets the hand start above the rim
    and stop wherever it first meets resistance.
    """
    points = _destination_cloud(runtime, destination)
    if points.shape[0] >= 20:
        return (float(np.percentile(points[:, 2], 95)), float(np.percentile(points[:, 2], 15)))
    top = float(destination.metadata.get("top_z", destination.pose[2]))
    return top, float(destination.metadata.get("floor_z", top))


def _destination_cloud(runtime, destination) -> np.ndarray:
    """The destination's points, or none at all when it was computed.

    ``object_points`` grounds any label it has not seen, so asking it about
    "right of plate" sends a detector after an object that does not exist and
    returns wherever it decided that was.
    """
    if destination.metadata.get("synthetic"):
        return np.zeros((0, 3))
    return _cloud(runtime, destination.label)


def _long_axis(runtime, target) -> float | None:
    """Direction of the destination's opening, when it has one worth matching."""
    axis = target.metadata.get("principal_axis")
    if axis is None:
        polygon = np.asarray(target.metadata.get("region_polygon_xy") or [], dtype=float).reshape(
            -1, 2
        )
        if polygon.shape[0] >= 3:
            flat = polygon - polygon.mean(axis=0)
            try:
                _, _, vt = np.linalg.svd(flat, full_matrices=False)
                axis = vt[0]
            except np.linalg.LinAlgError:
                axis = None
    if target.metadata.get("synthetic") and axis is None:
        return None
    if axis is None:
        points = _destination_cloud(runtime, target)
        if points.shape[0] < 12:
            return None
        flat = points[:, :2] - points[:, :2].mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(flat, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        axis = vt[0]
    return float(np.arctan2(float(axis[1]), float(axis[0])))


def _destination_aspect(runtime, target) -> float:
    """Long/short opening ratio, including synthetic RGB-D polygons."""
    polygon = np.asarray(target.metadata.get("region_polygon_xy") or [], dtype=float).reshape(-1, 2)
    if polygon.shape[0] >= 3:
        flat = polygon - polygon.mean(axis=0)
        try:
            spread = np.linalg.svd(flat, compute_uv=False)
        except np.linalg.LinAlgError:
            spread = np.zeros(2)
        if spread.size >= 2 and spread[1] > 1e-6:
            return float(spread[0] / spread[1])
    points = _destination_cloud(runtime, target)
    if points.shape[0] >= 12:
        return _footprint_aspect(points)
    extent = target.metadata.get("extent") or [0.0, 0.0, 0.0]
    xy = sorted((abs(float(extent[0])), abs(float(extent[1]))))
    return xy[1] / max(xy[0], 1e-6) if xy[1] > 0 else 1.0


def _centre_over(
    runtime,
    label: str,
    goal_xy: np.ndarray,
    z: float,
    trace: list,
    rounds: int = 2,
) -> np.ndarray:
    """Shuffle the hand until the object it holds is over ``goal_xy``.

    Where the object hangs relative to the hand depends on the grasp that
    happened to work: a rim grasp holds a bowl by one side, so driving the hand
    to the centre of a plate leaves the bowl half a bowl-width off it. That
    offset can be measured once and subtracted, and doing so was wrong often
    enough to matter -- a held object is partly behind the gripper and its
    grounding drifts.

    Looking at where the object actually is, moving by the error, and looking
    again converges regardless, and stops as soon as the picture agrees with
    the intention. Two rounds, because the third never moved anything by more
    than a millimetre.
    """
    commanded = np.asarray(goal_xy, dtype=float).copy()
    visual_label = (
        f"the {_appearance_label(label)} held by the robot gripper"
        if _has_transient_spatial_identity(label)
        else label
    )
    for attempt in range(rounds):
        try:
            hand_xy = np.asarray(runtime.ee_pose()["position"][:2], dtype=float)
        except Exception:
            hand_xy = commanded.copy()
        held = _ground(runtime, visual_label)
        source = "agent_rgbd"
        attachment_error = (
            float(np.linalg.norm(np.asarray(held.pose[:2]) - hand_xy))
            if _plausible(held)
            else float("inf")
        )
        # A carried object must remain near the wrist.  In duplicate-object
        # scenes a global attachment query can still bind to a table instance;
        # do not use that centimetre error as control.  Try the close wrist
        # view, whose limited field of view is an independent attachment gate.
        if attachment_error > MAX_CARRY_OFFSET:
            try:
                wrist = runtime.localize(
                    visual_label,
                    close_view=True,
                    detector="vlm_bbox_sam3",
                )
            except Exception:
                wrist = None
            wrist_error = (
                float(np.linalg.norm(np.asarray(wrist.pose[:2]) - hand_xy))
                if _plausible(wrist)
                else float("inf")
            )
            if wrist_error <= MAX_CARRY_OFFSET:
                held = wrist
                attachment_error = wrist_error
                source = "wrist_rgbd"
        if not _plausible(held):
            trace.append(
                {
                    "event": "centre",
                    "round": attempt,
                    "seen": False,
                    "visual_label": visual_label,
                }
            )
            break
        if attachment_error > MAX_CARRY_OFFSET:
            trace.append(
                {
                    "event": "centre",
                    "round": attempt,
                    "seen": True,
                    "attachment_consistent": False,
                    "attachment_error_cm": round(attachment_error * 100, 1),
                    "visual_label": visual_label,
                }
            )
            break
        error = goal_xy - np.asarray(held.pose[:2], dtype=float)
        magnitude = float(np.linalg.norm(error))
        trace.append(
            {
                "event": "centre",
                "round": attempt,
                "error_cm": round(magnitude * 100, 1),
                "attachment_consistent": True,
                "attachment_source": source,
                "visual_label": visual_label,
            }
        )
        # An error this large is the detector having found a different object,
        # not the hand being half a metre out. Correcting by it walks the
        # placement away from a destination that was already right, which is
        # how four working episodes were lost; the commanded position is the
        # thing to keep.
        if magnitude < 0.008 or magnitude > MAX_CARRY_OFFSET:
            break
        proposal = commanded + error
        if float(np.linalg.norm(proposal - goal_xy)) > MAX_CARRY_OFFSET:
            break
        commanded = proposal
        runtime.move_to([commanded[0], commanded[1], z])
    return commanded


def _carry_offset(
    runtime,
    label: str,
    trace: list,
    *,
    source: GroundedTarget | None = None,
) -> np.ndarray:
    """Measure where a held object's centre hangs relative to the hand.

    This helps precise relation placements but was harmful as a universal
    default when the held-object detector latched onto a distractor. It is now
    an explicit ReAct knob used after a measured lateral miss.
    """
    try:
        hand = np.asarray(runtime.ee_pose()["position"][:2], dtype=float)
    except Exception:
        return np.zeros(2)

    # Prefer a fresh wrist-camera observation: it isolates the carried object
    # from identical instances remaining on the table, so relational labels
    # do not become ambiguous after lifting.
    geometric = None
    if source is not None:
        candidate = np.asarray(source.pose[:2], dtype=float) - hand
        if float(np.linalg.norm(candidate)) <= MAX_CARRY_OFFSET:
            geometric = candidate
    observations: list[tuple[str, GroundedTarget | None]] = []
    appearance = _appearance_label(label)
    held_query = f"the {appearance} held by the robot gripper"
    # The home scene supplies better planar metric geometry, while the
    # attachment phrase distinguishes the carried instance from duplicates.
    # Prefer it over a wrist crop whose severe gripper occlusion can pull a
    # rim-grasped bowl's centroid toward the hand by several centimetres.
    for source_name, close_view in (
        ("agent_held_rgbd", False),
        ("wrist_rgbd", True),
    ):
        try:
            observations.append(
                (
                    source_name,
                    runtime.localize(
                        held_query,
                        close_view=close_view,
                        detector="vlm_bbox_sam3",
                    ),
                )
            )
        except Exception:
            pass
    # A front/middle/back selector ceases to identify the carried instance
    # after lift. The scene view can then bind to a duplicate still on the
    # table and manufacture a bowl-width correction. Only the wrist-held query
    # is instance-safe in that case; unique/appearance-only labels retain the
    # broader fallback.
    if not _has_transient_spatial_identity(label):
        try:
            observations.append(("agent_rgbd", _ground(runtime, label)))
        except Exception:
            pass
    for source_name, held in observations:
        if not _plausible(held):
            continue
        visual = np.asarray(held.pose[:2], dtype=float) - hand
        norm = float(np.linalg.norm(visual))
        if norm <= MAX_CARRY_OFFSET:
            trace.append(
                {
                    "event": "carry_offset",
                    "source": source_name,
                    "offset": [round(float(v), 3) for v in visual],
                }
            )
            return visual
        trace.append(
            {
                "event": "carry_offset",
                "source": source_name,
                "rejected": round(norm, 3),
            }
        )
    # The pre-grasp offset is only safe when the grasp was already close to
    # the centre.  A rim contact several centimetres away can pull/rotate the
    # vessel into the fingers, so assuming that large offset stayed rigid was
    # observed to over-correct by an entire bowl width.
    if geometric is not None and float(np.linalg.norm(geometric)) <= 0.03:
        trace.append(
            {
                "event": "carry_offset",
                "source": "pregrasp_rigid_offset",
                "offset": [round(float(v), 3) for v in geometric],
            }
        )
        return geometric
    return np.zeros(2)


def _place_carefully(
    runtime,
    carried: str,
    destination,
    band: tuple[float, float],
    carry: float,
    yaw: float | None,
    trace: list,
    *,
    nudge: np.ndarray,
    carry_offset: np.ndarray,
    margin: float,
    release_on: str,
    centre: bool,
    report: dict,
    transit_yaw: float | None = None,
) -> None:
    """Carry the object over the destination and lower it until it lands.

    The hand is aimed by looking at the object it is carrying rather than by
    assuming it hangs straight down, because a bowl taken by its rim does not.

    It normally releases from just above the destination's rim.  A measured
    cavity with at least 5.5 cm radial hand clearance is the exception: after
    the high transit, the hand can descend inside and release near the floor.
    This avoids dropping a package onto a drawer wall while retaining the
    rim release for narrow baskets/slots where the gripper itself cannot fit.

    ``yaw`` squares the jaws to the destination's opening. A book is held
    across its faces, so its plane is fixed by the hand; a slot it does not
    line up with simply refuses it.
    """
    rim, floor = band
    requested_release_on = release_on
    cavity_clearance = float(destination.metadata.get("region_clearance_m", 0.0) or 0.0)
    wide_cavity = bool(
        destination.metadata.get("region_source") == "rgbd_cavity"
        and cavity_clearance >= 0.055
        and rim - floor >= 0.03
        and re.search(
            r"\b(?:drawer|box|bin|container|cubby|compartment)\b",
            str(destination.label),
            flags=re.IGNORECASE,
        )
    )
    if wide_cavity:
        release_on = "floor"
    report["release_on_requested"] = requested_release_on
    report["release_on_effective"] = release_on
    report["in_container_descent"] = wide_cavity
    report["in_container_hand_clearance_m"] = round(cavity_clearance, 4)
    base = floor if release_on == "floor" else rim
    goal = np.asarray(destination.pose[:2], dtype=float) + nudge - carry_offset
    release_z = max(base + carry + margin, floor + MIN_HAND_CLEARANCE)

    trace.append(
        {
            "event": "place_plan",
            "rim_z": round(rim, 4),
            "floor_z": round(floor, 4),
            "carry": round(carry, 4),
            "yaw": None if yaw is None else round(float(yaw), 3),
            "release_z": round(release_z, 4),
            "nudge": [round(float(v), 3) for v in nudge],
            "carry_offset": [round(float(v), 3) for v in carry_offset],
            "goal_xy": [round(float(v), 3) for v in goal],
        }
    )

    # Clear the obstacle the object was picked from, without blindly retaining
    # an arbitrarily high end-effector pose.  The latter is safe geometrically
    # but makes RGB-D centring ill-conditioned: a small angular error becomes
    # several centimetres at the table.  Source support height is the physical
    # quantity that matters.  Descending vertically above that source to this
    # plane is safe; only then do we translate across the scene.
    try:
        source_surface_z = float(report["surface_under_object_z"])
    except (KeyError, TypeError, ValueError):
        source_surface_z = -float("inf")
    source_clearance_z = source_surface_z + carry + 0.08
    travel_z = max(release_z + TRANSIT_CLEARANCE, source_clearance_z)
    report["transit_z"] = round(float(travel_z), 4)
    report["transit_source_surface_z"] = (
        None if not np.isfinite(source_surface_z) else round(source_surface_z, 4)
    )
    report["transit_preserved_source_clearance"] = bool(
        source_clearance_z > release_z + TRANSIT_CLEARANCE
    )
    # First reach the transit plane *vertically over the source*.  Sending a
    # single Cartesian target at (destination XY, high Z) lets the low-level
    # interpolator take a diagonal shortcut; a carried object can then sweep
    # through the front wall of an open drawer before it has climbed over the
    # rim.  The explicit lift-horizontal-descend decomposition is required for
    # every elevated/container destination.
    try:
        hand = np.asarray(runtime.ee_pose()["position"], dtype=float)
        runtime.move_to([float(hand[0]), float(hand[1]), travel_z])
        report["transit_vertical_lift_completed"] = True
    except Exception:
        report["transit_vertical_lift_completed"] = False
        raise
    # An edge pinch needs a tilted wrist only until the thin object clears its
    # support. Level it in place at the transit plane before translating over
    # a rim; carrying the 25-degree pickup tilt into a drawer made the lower
    # fingertip/object corner strike the wall and eject the load.
    motion_yaw = yaw if yaw is not None else transit_yaw
    if motion_yaw is not None:
        runtime.move_to([float(hand[0]), float(hand[1]), travel_z], yaw=motion_yaw)
        report["transit_wrist_levelled"] = True
        report["transit_wrist_yaw_deg"] = round(float(np.degrees(motion_yaw)), 1)
        runtime.move_to([goal[0], goal[1], travel_z], yaw=motion_yaw)
    else:
        report["transit_wrist_levelled"] = False
    runtime.move_to([goal[0], goal[1], travel_z])
    xy = _centre_over(runtime, carried, goal, travel_z, trace) if centre else goal
    runtime.move_to([xy[0], xy[1], release_z])
    runtime.open_gripper()
    runtime.move_to([xy[0], xy[1], travel_z])
    report["release_xy"] = [round(float(v), 3) for v in xy]
    report["release_z"] = round(release_z, 4)


def _appearance_label(label: str) -> str:
    """Remove pre-action spatial qualifiers that cannot survive transport.

    ``front butter`` and ``back butter`` are selected by scene geometry before
    grasping; once either one is carried into a drawer, a tight crop can only
    establish that it is butter.  Asking a VLM whether that crop is still
    "the butter at the front" turns correct action continuity into a false
    identity veto.  Colour, material, size and other visible modifiers remain.
    """
    text = re.sub(r"^\s*the\s+", "", str(label), flags=re.IGNORECASE)
    patterns = (
        r"\s+at\s+the\s+(?:front|back|left|right|middle|center|centre)\b.*$",
        r"\s+in\s+the\s+(?:front|back|middle|center|centre)\b.*$",
        r"\s+on\s+the\s+(?:left|right)\b.*$",
        r"\s+(?:nearest|closest|farthest)\s+to\b.*$",
    )
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip() or str(label).strip()


def _has_transient_spatial_identity(label: str) -> bool:
    """Whether the label ceases to identify the same object after transport.

    In a scene with two identical packages, ``the butter at the front`` is a
    valid pre-grasp selector.  Once that package is lifted, re-running the same
    relation can bind to the remaining package, so treating its unchanged Z as
    proof of a wrong grasp actively undoes the correct action.  Appearance
    modifiers remain stable; only relation/ordering selectors are transient.
    """
    return (
        _appearance_label(label).strip().lower()
        != re.sub(r"^\s*the\s+", "", str(label), flags=re.IGNORECASE).strip().lower()
    )


def _confirm_identity(runtime, label: str, target) -> dict:
    """Is the thing we grounded actually the thing we were asked for?

    Cropping to the grounding and enlarging it recovers label text that is a
    few pixels wide in the full frame. This is the only check measured to
    separate a correct grounding from a confident wrong one -- and it is not
    repeatable, which is why it does not veto by default.
    """
    # A relational phrase can only be verified in the full scene.  The normal
    # crop verifier deliberately removes that relation because it is useful
    # after transport; using it before grasping accepted any same-looking
    # neighbour as the requested front/middle/back instance.
    if _has_transient_spatial_identity(label):
        verifier = getattr(runtime, "verify_pick_grounding", None)
        if callable(verifier):
            try:
                verdict = dict(verifier(label, target=target) or {})
            except (PerceptionUnavailableError, LLMError):
                raise
            except Exception as exc:
                verdict = {
                    "ok": False,
                    "matches": None,
                    "reason": f"relation_inspect_failed:{type(exc).__name__}",
                }
            verdict.setdefault("appearance_label", _appearance_label(label))
            verdict["relation_preserved"] = True
            return verdict

    appearance = _appearance_label(label)
    other = "a different object"
    try:
        inspect_kwargs = {
            "options": [appearance, other],
            "target": target,
        }
        target_view = str(target.metadata.get("view", "agentview"))
        if target_view != "agentview":
            inspect_kwargs["view"] = target_view
        try:
            verdict = runtime.inspect(
                "identity of the tightly cropped candidate; judge visible "
                "appearance only, independent of the commanded action",
                **inspect_kwargs,
            )
        except TypeError:
            # Lightweight test/evolution runtimes predate explicit view
            # selection; retain their ordinary crop-verifier contract.
            inspect_kwargs.pop("view", None)
            verdict = runtime.inspect(
                "identity of the tightly cropped candidate; judge visible "
                "appearance only, independent of the commanded action",
                **inspect_kwargs,
            )
    except (PerceptionUnavailableError, LLMError):
        raise
    except Exception as exc:
        return {"ok": False, "reason": f"inspect_failed:{type(exc).__name__}"}
    answer = str(verdict.get("answer", "")).strip().lower()
    expected = appearance.lower()
    different = other.lower()
    # Closed-set calls occasionally return prose or an empty answer despite
    # ``ok=True``.  Such a response is unknown evidence, not a match.  Only an
    # exact requested-label choice accepts the crop and only the explicit
    # alternative vetoes it; callers may safely retry a second view on veto.
    verdict["matches"] = (
        True
        if verdict.get("ok") and answer == expected
        else False
        if verdict.get("ok") and answer == different
        else None
    )
    verdict["appearance_label"] = appearance
    return verdict


def _ground_verified_pick(
    runtime,
    label: str,
    trace: list,
    *,
    strict_unknown: bool = False,
    max_attempts: int = 4,
    max_planar_extent: float | None = None,
) -> tuple[GroundedTarget | None, dict[str, object], int]:
    """Ground one referent, revising any independently rejected candidate.

    Joint detection is the accurate default on ordinary tabletop objects, but
    its one-box-per-label prompt can omit a thin, edge-on object entirely.  An
    absent box is not negative category evidence.  In that case, change the
    *visual prompt family* before giving up: ask for a single-object box, then
    for a graspable point.  Every plausible proposal still goes through the
    independent identity verifier below, so this improves recall without
    turning an open-vocabulary inventory guess into permission to move.
    """
    rejected: list[GroundedTarget] = []
    feedback_attempted = False
    evidence: dict[str, object] = {"used": False, "matches": None}
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        try:
            feedback = getattr(runtime, "localize_with_feedback", None)
            if rejected and callable(feedback) and not feedback_attempted:
                candidate = feedback(label, rejected)
                feedback_attempted = True
                grounding_mode = "vlm_negative_feedback"
                close_view = False
            else:
                close_view = False
                if attempt == 1:
                    candidate = runtime.localize_many([label], close_view=False).get(label)
                    grounding_mode = "joint_bbox"
                else:
                    locator = getattr(runtime, "localize", None)
                    if not callable(locator):
                        # Compatibility for lightweight/offline runtimes.
                        close_view = attempt > 1
                        candidate = runtime.localize_many([label], close_view=close_view).get(label)
                        grounding_mode = "close_joint_bbox" if close_view else "joint_bbox"
                    elif attempt <= 3:
                        detector = "vlm_bbox_sam3" if attempt == 2 else "vlm_search_bbox_sam3"
                        candidate = locator(label, detector=detector)
                        grounding_mode = detector
                    else:
                        multi_view = getattr(runtime, "localize_across_views", None)
                        if callable(multi_view):
                            candidate = multi_view(label, detector="vlm_point_sam3")
                            grounding_mode = "multi_view_point_sam3"
                        else:
                            candidate = locator(label, detector="vlm_search_bbox_sam3")
                            grounding_mode = "vlm_search_bbox_sam3"
        except PerceptionUnavailableError:
            raise
        except Exception as exc:
            trace.append(
                {
                    "event": "grounded",
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        trace.append(
            {
                "event": "grounded",
                "attempt": attempt,
                "close_view": close_view,
                "mode": grounding_mode,
                "pick_points": (candidate.metadata.get("n_points") if candidate else 0),
                "confidence": (round(float(candidate.confidence), 4) if candidate else 0.0),
                "extent": (
                    [round(float(value), 4) for value in candidate.metadata.get("extent", [])]
                    if candidate
                    else []
                ),
                "bbox": (list(candidate.metadata.get("bbox", [])) if candidate else []),
                "reason": (candidate.metadata.get("reason") if candidate else "no_candidate"),
            }
        )
        if not _plausible(candidate):
            # A failed negative-feedback query is not evidence that the object
            # is absent.  Continue into the independent exhaustive prompt on
            # the final rung instead of replaying this marked cached image or
            # terminating the whole source search early.
            continue
        if max_planar_extent is not None:
            extent = np.asarray(candidate.metadata.get("extent") or [], dtype=float).reshape(-1)
            planar = float(np.max(np.abs(extent[:2]))) if extent.size >= 2 else 0.0
            if planar > float(max_planar_extent):
                trace.append(
                    {
                        "event": "grounding_rejected_geometry",
                        "attempt": attempt,
                        "reason": "planar_extent_exceeds_single_grasp_limit",
                        "planar_extent_m": round(planar, 4),
                        "limit_m": round(float(max_planar_extent), 4),
                    }
                )
                # This is geometric negative evidence, not an identity veto.
                # Do not feed it to the semantic exclusion prompt; simply use
                # the next independent grounding family / viewpoint.
                continue
        verdict = _confirm_identity(runtime, label, candidate)
        trace.append(
            {
                "event": "identity",
                "attempt": attempt,
                **{key: verdict.get(key) for key in ("answer", "matches")},
            }
        )
        if verdict.get("ok") and verdict.get("matches") is False:
            if candidate.metadata.get("bbox"):
                rejected.append(candidate)
                trace.append(
                    {
                        "event": "grounding_rejected",
                        "attempt": attempt,
                        "bbox": [int(v) for v in candidate.metadata["bbox"]],
                    }
                )
                evidence = {
                    "used": True,
                    "matches": False,
                    "answer": verdict.get("answer", ""),
                    "view": grounding_mode,
                    "veto_enabled": True,
                    "rejected_candidates": len(rejected),
                }
                continue
            if VETO_ON_IDENTITY or strict_unknown:
                continue
            verdict["matches"] = None
        if (VETO_ON_IDENTITY or strict_unknown) and verdict.get("matches") is not True:
            continue
        evidence = {
            "used": True,
            "matches": verdict.get("matches"),
            "answer": verdict.get("answer", ""),
            "view": grounding_mode,
            "veto_enabled": bool(VETO_ON_IDENTITY or strict_unknown),
            "rejected_candidates": len(rejected),
        }
        return candidate, evidence, attempt
    return None, evidence, max(1, int(max_attempts))


def _refresh_pick_after_contact(runtime, label: str, current, trace: list):
    """Measure the same established instance after a failed grasp contact.

    Prefer local visual tracking around the verified pre-contact box. Global
    language re-grounding is retained only as a compatibility fallback for
    runtimes without local tracking; it can otherwise rebind transient
    front/back selectors to an untouched duplicate and adds several VLM calls
    to every rung of the grasp ladder.
    """
    tracker = getattr(runtime, "track_target", None)
    if callable(tracker) and _plausible(current):
        try:
            refreshed = tracker(current, appearance_label=_appearance_label(label))
        except Exception as exc:
            trace.append(
                {
                    "event": "local_pick_track",
                    "accepted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            trace.append(
                {
                    "event": "local_pick_track",
                    "accepted": bool(_plausible(refreshed)),
                    "detector": (refreshed.metadata.get("detector") if refreshed else None),
                }
            )
            if _plausible(refreshed):
                evidence = dict(current.metadata.get("pregrasp_identity_evidence") or {})
                evidence.update(
                    {
                        "continued_by": "local_sam3_rgbd",
                        "global_vlm_recheck": False,
                    }
                )
                refreshed.metadata["pregrasp_identity_evidence"] = evidence
                return refreshed, evidence

    refreshed, evidence, _ = _ground_verified_pick(runtime, label, trace, max_attempts=3)
    if evidence is not None:
        evidence = dict(evidence)
        evidence["continued_by"] = "global_vlm_fallback"
    return refreshed, evidence


def _held_object_left_the_table(runtime, label: str, before) -> bool:
    """Did the object we asked for actually come up with the gripper?

    `verify_grasp` only reports that the fingers are apart. A failed rim
    contact can shove a flat object many centimetres while it stays on the
    table, so planar displacement is not lift evidence. Prefer a measured Z
    rise in the scene view; if that view rebinds to an identical distractor or
    loses the object, require a second wrist-view grounding that is both raised
    and physically near the hand.
    """

    def scene_verdict(after) -> bool | None:
        if after is None or after.confidence <= 0:
            return None
        rise = float(after.pose[2]) - float(before.pose[2])
        planar = float(
            np.linalg.norm(
                np.asarray(after.pose[:2], dtype=float) - np.asarray(before.pose[:2], dtype=float)
            )
        )
        if rise > 0.02:
            extent = np.asarray(
                before.metadata.get("extent") or [0.0, 0.0, 0.0], dtype=float
            ).reshape(-1)
            broad_object = bool(max(np.abs(extent[:2]), default=0.0) >= 0.16)
            if not broad_object:
                return True
            # A wide or irregular body can be tipped upright by a failed
            # contact, raising its scene centroid while remaining on the table.
            # Defer these ambiguous rises to the wrist-view near-hand check.
            return None
        # This is decisive even for a transient selector: if the same phrase
        # still grounds a well-supported object at the original contact pose,
        # the load did not follow the hand.  The old unconditional transient
        # acceptance turned table/finger collision into a false grasp.
        if planar <= 0.045 and int(after.metadata.get("n_points", 0)) >= MIN_POINTS:
            # With an instance-qualified label, a global category prompt can
            # bind to a neighbouring lookalike that stayed at (or close to)
            # the selected source pose.  It is not valid negative evidence
            # against the mechanically held load.  Defer to the independent
            # wrist / near-hand observation.  For an unqualified unique
            # object, seeing it still supported at its original pose remains
            # a decisive empty-grasp signal.
            if _has_transient_spatial_identity(label):
                return None
            return False
        # Once a front/back/left/right/middle referent moves, the same phrase
        # can rebind to a remaining lookalike. That rebind is ambiguity, not
        # evidence that the load followed the hand: a failed edge grasp can
        # shove the intended book aside and produce the same observation.
        # Defer to the wrist-view held-object check below.
        if _has_transient_spatial_identity(label) and planar >= 0.08:
            return None
        return None

    # For an instance-qualified source, preserve the already verified visual
    # instance before asking any global category prompt.  Local SAM tracking
    # around its pre-grasp box cannot jump to a distant sibling.  A sizeable
    # rise, small planar drift, and proximity to the hand are independent
    # geometric evidence that the same body followed the gripper.
    if _has_transient_spatial_identity(label):
        try:
            tracked = runtime.track_target(
                before,
                appearance_label=_appearance_label(label),
                padding_px=30,
            )
            hand = np.asarray(runtime.ee_pose()["position"], dtype=float)
            tracked_pose = np.asarray(tracked.pose, dtype=float)
            rise = float(tracked_pose[2] - float(before.pose[2]))
            planar = float(
                np.linalg.norm(tracked_pose[:2] - np.asarray(before.pose[:2], dtype=float))
            )
            near_hand = float(np.linalg.norm(tracked_pose - hand))
            if _plausible(tracked) and rise > 0.03 and planar <= 0.06 and near_hand <= 0.20:
                return True
        except Exception:
            pass

    # Motion continuity is geometric, so start with the local SAM3 mask.  A
    # language VLM is reserved for an absent or ambiguous mask; it is not
    # needed merely to prove that a visible body rose several centimetres.
    for locator in (_ground_visible, _ground):
        try:
            verdict = scene_verdict(locator(runtime, label))
        except Exception:
            verdict = None
        if verdict is not None:
            return verdict
    try:
        wrist = runtime.localize(
            f"the {_appearance_label(label)} held by the robot gripper",
            close_view=True,
            detector="vlm_bbox_sam3",
        )
        hand = np.asarray(runtime.ee_pose()["position"], dtype=float)
    except Exception:
        return False
    if not _plausible(wrist):
        return False
    near_hand = float(np.linalg.norm(np.asarray(wrist.pose, dtype=float) - hand)) <= 0.18
    raised = float(wrist.pose[2]) - float(before.pose[2]) > 0.02
    return bool(near_hand and raised)


def _return_unintended_grasp(runtime, target, trace: list) -> bool:
    """Put an unidentified load back where the gripper made contact.

    Finger opening only proves that *something* was caught.  When the named
    object is still at its original pose, returning a failure while retaining
    that neighbour makes all later perception and motion inconsistent.  The
    contact pose is visual evidence for a conservative return location and
    does not require knowing the neighbour's identity.
    """
    try:
        if not runtime.verify_grasp(""):
            return True
    except Exception:
        return False
    pose = np.asarray(target.pose, dtype=float)
    top = float(target.metadata.get("top_z", pose[2]))
    try:
        runtime.move_to([pose[0], pose[1], top + 0.06])
        runtime.open_gripper()
        runtime.move_to([pose[0], pose[1], top + 0.14])
        trace.append({"event": "return_unintended_grasp", "released": True})
        return True
    except Exception as exc:
        trace.append(
            {
                "event": "return_unintended_grasp",
                "released": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        try:
            runtime.recover()
            return True
        except Exception:
            return False


# -------------------------------------------------------------------- skill


def pickplace(
    runtime,
    pick_label: str,
    place_label: str,
    *,
    grasp: str | list[str] | None = None,
    grasp_depth: float | None = None,
    destination: str = "auto",
    nudge: tuple[float, float] = (0.0, 0.0),
    place_margin: float = PLACE_MARGIN,
    release_on: str = "rim",
    yaw_deg: float | str | None = "auto",
    centre: bool = False,
    compensate: bool = False,
    identity_check: bool = False,
    _pick_target: GroundedTarget | None = None,
    _destination_target: GroundedTarget | None = None,
) -> SkillResult:
    """Move one object onto or into another, with every judgement overridable.

    Called with no options this behaves as the tuned default: it picks a grasp
    ladder from the shape of the object, resolves the destination phrase
    according to what kind of phrase it is, and releases just above the rim.
    Those defaults are right more often than any single fixed choice, and they
    are still wrong perhaps a third of the time, in ways that are obvious from
    a photograph of the result and not from inside the skill.

    So each of them is an argument. A caller that can look at the scene -- an
    agent with the failed attempt in front of it -- can say *take it by the rim
    instead*, *that is the drawer, not the cabinet top*, *you released two
    centimetres to the left*, and get a different attempt rather than the same
    one. The skill stays general; the knowledge of which setting this scene
    needs lives with whatever can see the scene.

    Args:
        grasp: Strategy to try first, or an ordered list of them, as
            ``"rim@top"``, ``"top_down@low"``, ``"pca_axis"``. The shape-chosen
            ladder still follows as fallback. ``None`` uses it alone.
        grasp_depth: Metres below the object's highest point to close the
            fingers. Raise it for something tall and tapered, lower it for
            something flat.
        destination: How to read ``place_label``. ``"auto"`` decides from the
            words; ``"object"`` grounds it as a thing; ``"region"`` as a part
            of a thing; ``"inside"`` finds the opening of a container;
            ``"top"`` finds the upper surface of one.
        nudge: Metres to shift the release point, as seen in the camera image:
            positive right, positive up.
        place_margin: Gap left between the base of the carried object and the
            destination when the fingers open.
        release_on: ``"rim"`` to clear the highest part of the destination,
            ``"floor"`` to aim at the lowest -- the inside of a drawer rather
            than its front edge.
        yaw_deg: Wrist angle for the release. ``"auto"`` squares a long object
            to a long opening, ``None`` keeps the grasp angle, a number sets it.
        centre: Whether to look at the carried object and correct the aim
            before opening the fingers. Off by default: it is worth several
            centimetres when the carried object grounds cleanly and worth
            losing the episode when it does not, and only a caller looking at
            the scene can tell those apart.
        compensate: Whether to measure and subtract the object's lateral
            offset from the hand. Useful on a precision retry; off by default
            because a wrong held-object grounding can point the wrong way.
        identity_check: Verify the pre-grasp crop against the requested label.
            This costs an extra visual call and is therefore reserved for a
            retry after evidence of a wrong-object grasp.
    """
    trace: list[dict[str, object]] = []

    # Validate every public knob before moving the arm.  ReAct models are good
    # at naming the correction they want and much less reliable at preserving
    # a JSON scalar type (one earlier run supplied ``[0.1, 0, 0]`` as a
    # margin, after the object had already been grasped).  A bad option must be
    # a cheap, retryable call rather than a half-executed manipulation.
    valid_destinations = {"auto", "object", "region", "inside", "top", "relation"}
    if destination not in valid_destinations:
        destination = "auto"
    if release_on not in {"rim", "floor"}:
        release_on = "rim"
    try:
        place_margin = float(np.clip(float(place_margin), 0.0, 0.10))
    except (TypeError, ValueError):
        place_margin = PLACE_MARGIN
    try:
        values = list(nudge)
        if len(values) != 2:
            raise ValueError("nudge needs two values")
        nudge = tuple(float(np.clip(float(v), -0.08, 0.08)) for v in values)
    except (TypeError, ValueError):
        nudge = (0.0, 0.0)
    if grasp_depth is not None:
        try:
            grasp_depth = float(np.clip(float(grasp_depth), 0.005, 0.08))
        except (TypeError, ValueError):
            grasp_depth = None
    if yaw_deg not in (None, "auto"):
        try:
            yaw_deg = float(yaw_deg)
        except (TypeError, ValueError):
            yaw_deg = "auto"
    centre = bool(centre)
    compensate = bool(compensate)
    identity_check = bool(identity_check)
    report: dict[str, object] = {
        "params": {
            "grasp": grasp,
            "grasp_depth": grasp_depth,
            "destination": destination,
            "nudge": list(nudge),
            "place_margin": place_margin,
            "release_on": release_on,
            "yaw_deg": yaw_deg,
            "centre": centre,
            "compensate": compensate,
            "identity_check": identity_check,
        }
    }

    def fail(mode: str, message: str, attempts: int = 1) -> SkillResult:
        return SkillResult(
            success=False,
            status="retryable_failure",
            pick_label=pick_label,
            place_label=place_label,
            attempts=attempts,
            strategy="phase1_policy_api",
            failure_mode=mode,
            message=message,
            trace=trace,
            report=report,
        )

    # Both groundings happen before the arm moves. Resolving the destination
    # afterwards costs nothing extra in calls and looked cheaper, but by then
    # the hand is holding the object over the table: it occludes part of the
    # scene, and the ring of pixels used to find how high a surface is starts
    # returning the gripper.
    pick = None
    pregrasp_identity: dict[str, object] = {
        "used": False,
        "matches": None,
    }
    if _pick_target is not None and _plausible(_pick_target):
        pick = _pick_target
        pregrasp_identity = dict(
            pick.metadata.get("pregrasp_identity_evidence")
            or {
                "used": True,
                "matches": None,
                "answer": "persistent task referent",
                "view": "locked_pre_action_grounding",
                "veto_enabled": False,
            }
        )
        trace.append(
            {
                "event": "pick_lock",
                "pose": [round(float(v), 3) for v in pick.pose],
                "points": int(pick.metadata.get("n_points", 0)),
            }
        )
        report["pick_locked"] = True
    else:
        pick, pregrasp_identity, grounding_attempts = _ground_verified_pick(
            runtime,
            pick_label,
            trace,
            strict_unknown=identity_check,
            max_attempts=3,
        )
        report["grounding_attempts"] = grounding_attempts

    if pick is None:
        report["grounding_failure"] = "pick"
        report["failed_label"] = pick_label
        return fail("not_grounded", f"no usable grounding for {pick_label!r}", attempts=3)
    report["pregrasp_identity_evidence"] = pregrasp_identity
    report["pick_confidence"] = round(float(pick.confidence), 4)
    report["pick_top_z"] = round(float(pick.metadata.get("top_z", pick.pose[2])), 5)
    report["pick_n_points"] = int(pick.metadata.get("n_points", 0))
    report["pick_principal_axis"] = pick.metadata.get("principal_axis")
    if pick.metadata.get("bbox"):
        report["pick_bbox"] = [int(v) for v in pick.metadata["bbox"]]

    goal = None
    if (
        _destination_target is not None
        and _plausible(_destination_target)
        and _in_reach(_destination_target, pick)
    ):
        goal = _destination_target
        trace.append(
            {
                "event": "destination_lock",
                "pose": [round(float(v), 3) for v in goal.pose],
                "source": goal.metadata.get("region_source"),
            }
        )
        report["destination_locked"] = True
    if goal is None:
        goal = _ground_destination(runtime, place_label, trace, near=pick, mode=destination)
    if goal is None:
        report["grounding_failure"] = "destination"
        report["failed_label"] = place_label
        return fail("not_grounded", f"no usable grounding for {place_label!r}")
    band = _release_band(runtime, goal)
    slot_yaw = _long_axis(runtime, goal)
    report["destination_pose"] = [round(float(v), 3) for v in goal.pose]
    report["destination_confidence"] = round(float(goal.confidence), 4)
    report["destination_rim_z"] = round(band[0], 4)
    report["destination_floor_z"] = round(band[1], 4)
    report["destination_kind"] = goal.kind
    report["destination_extent"] = [
        round(float(v), 4) for v in (goal.metadata.get("extent") or [0.0, 0.0, 0.0])
    ]
    report["destination_synthetic"] = bool(goal.metadata.get("synthetic"))
    if goal.metadata.get("bbox"):
        report["destination_bbox"] = [int(v) for v in goal.metadata["bbox"]]
    for source_key, report_key in (
        ("region_source", "destination_region_source"),
        ("region_polygon_xy", "destination_region_polygon_xy"),
        ("region_boundary_pixels", "destination_region_boundary_pixels"),
        ("region_safe_pixel", "destination_region_safe_pixel"),
        ("region_area_pixels", "destination_region_area_pixels"),
        ("region_clearance_m", "destination_region_clearance_m"),
        ("region_confidence", "destination_region_confidence"),
        ("region_planar_repair", "destination_region_planar_repair"),
    ):
        if source_key in goal.metadata:
            report[report_key] = goal.metadata[source_key]

    # A post-release continuity lock may use a relational visual query to
    # retain the same physical instance after its original "front/back"
    # qualifier stopped being meaningful.  Reuse that query's cached cloud
    # together with its locked pose; re-grounding the original phrase here
    # would undo the identity lock.
    points_label = str(pick.metadata.get("points_label", pick_label))
    points = _cloud(runtime, points_label)
    closure_nudge, closure_depth = _drawer_inward_nudge(runtime, goal, points)
    if closure_depth:
        report["closure_safe_depth"] = closure_depth
    report["centre_effective"] = centre
    report["centre_reason"] = "requested" if centre else "disabled"
    table_z = _surface_height(runtime, pick)
    top_z, raw_top_z = _robust_object_top(pick, points, table_z)
    points = _object_body_points(pick, points, table_z, top_z)
    object_aspect = _footprint_aspect(points)
    heights = {
        "top": _grasp_height(pick, table_z, depth=grasp_depth, top_z=top_z),
        "low": _low_grasp_height(table_z, top_z),
        "edge": _edge_grasp_height(table_z, top_z),
    }
    attempts = _attempts(points, object_height=top_z - table_z)
    if grasp is not None:
        attempts = _ladder(grasp, attempts)
    complete_ladder = attempts
    attempts = attempts[:MAX_GRASP_ATTEMPTS_PER_CALL]
    report["pick_pose"] = [round(float(v), 3) for v in pick.pose]
    report["pick_extent"] = [
        round(float(v), 4) for v in (pick.metadata.get("extent") or [0.0, 0.0, 0.0])
    ]
    report["object_top_z"] = round(top_z, 4)
    report["object_top_raw_z"] = round(raw_top_z, 4)
    report["object_top_repaired"] = bool(raw_top_z - top_z > 0.015)
    report["surface_under_object_z"] = round(table_z, 4)
    report["hollow"] = _looks_hollow(points)
    report["aspect"] = round(object_aspect, 2)
    report["ladder"] = [f"{s}@{h}" for s, h in attempts]
    report["complete_ladder"] = [f"{s}@{h}" for s, h in complete_ladder]
    report["grasp_attempt_limit"] = MAX_GRASP_ATTEMPTS_PER_CALL
    trace.append(
        {
            "event": "grasp_plan",
            "top_z": top_z,
            "table_z": round(table_z, 4),
            "close_at": {k: round(v, 4) for k, v in heights.items()},
            "hollow": _looks_hollow(points),
            "aspect": round(object_aspect, 2),
            "attempts": [f"{s}@{h}" for s, h in attempts],
        }
    )

    held = False
    height = heights["top"]
    used = attempts[0][0]
    # Keep an immutable spatial identity anchor for the whole grasp ladder.
    # ``pick`` itself is updated after legitimate contact-induced motion.
    pick_identity_anchor_xy = np.asarray(pick.pose[:2], dtype=float).copy()
    for index, (used, where) in enumerate(attempts):
        height = heights[where]
        try:
            runtime.open_gripper()
            runtime.grasp(_lowered(pick, height), strategy=used)
        except Exception as exc:
            trace.append(
                {
                    "event": "grasp_attempt",
                    "strategy": used,
                    "at": where,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if not runtime.verify_grasp(pick_label):
            trace.append(
                {
                    "event": "grasp_attempt",
                    "strategy": used,
                    "at": where,
                    "held": False,
                    "reason": "empty_close",
                }
            )
            # Contact without retention can still slide or roll the object.
            # Reusing the original target then turns every remaining ladder
            # entry into an air grasp.  Clear the camera, re-ground, and update
            # both XY and grasp height before trying a materially different
            # contact mode.  Reject large jumps so a distractor cannot hijack
            # the retry.
            try:
                runtime.open_gripper()
                runtime.recover()
                refreshed, refresh_identity = _refresh_pick_after_contact(
                    runtime, pick_label, pick, trace
                )
            except Exception as exc:
                trace.append(
                    {
                        "event": "grasp_reground",
                        "accepted": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            step_shift = (
                float(
                    np.linalg.norm(
                        np.asarray(refreshed.pose[:2], dtype=float)
                        - np.asarray(pick.pose[:2], dtype=float)
                    )
                )
                if _plausible(refreshed)
                else float("inf")
            )
            anchor_shift = (
                float(
                    np.linalg.norm(
                        np.asarray(refreshed.pose[:2], dtype=float) - pick_identity_anchor_xy
                    )
                )
                if _plausible(refreshed)
                else float("inf")
            )
            if (
                _plausible(refreshed)
                and step_shift <= MAX_GRASP_REGROUND_STEP_M
                and anchor_shift <= MAX_GRASP_REGROUND_ANCHOR_M
            ):
                pick = refreshed
                points_label = str(pick.metadata.get("points_label", pick_label))
                refreshed_points = _cloud(runtime, points_label)
                table_z = _surface_height(runtime, pick)
                top_z, refreshed_raw_top = _robust_object_top(pick, refreshed_points, table_z)
                heights = {
                    "top": _grasp_height(pick, table_z, depth=grasp_depth, top_z=top_z),
                    "low": _low_grasp_height(table_z, top_z),
                    "edge": _edge_grasp_height(table_z, top_z),
                }
                report.setdefault("grasp_regroundings", []).append(
                    {
                        "shift_cm": round(step_shift * 100, 1),
                        "anchor_shift_cm": round(anchor_shift * 100, 1),
                        "pose": [round(float(v), 3) for v in pick.pose],
                        "object_top_z": round(top_z, 4),
                        "object_top_raw_z": round(refreshed_raw_top, 4),
                        "identity": refresh_identity,
                    }
                )
                trace.append(
                    {
                        "event": "grasp_reground",
                        "accepted": True,
                        "shift_cm": round(step_shift * 100, 1),
                        "anchor_shift_cm": round(anchor_shift * 100, 1),
                    }
                )
            else:
                trace.append(
                    {
                        "event": "grasp_reground",
                        "accepted": False,
                        "shift_cm": None
                        if not np.isfinite(step_shift)
                        else round(step_shift * 100, 1),
                        "anchor_shift_cm": None
                        if not np.isfinite(anchor_shift)
                        else round(anchor_shift * 100, 1),
                        "max_step_shift_cm": round(MAX_GRASP_REGROUND_STEP_M * 100, 1),
                        "max_anchor_shift_cm": round(MAX_GRASP_REGROUND_ANCHOR_M * 100, 1),
                    }
                )
            continue
        if not _held_object_left_the_table(runtime, pick_label, pick):
            # Finger/table collision can leave the opening sensor in the
            # nominal "holding" band even though the visually grounded target
            # stayed put.  Open and restore any unidentified contact, then try
            # the next materially different grasp instead of ending the whole
            # skill on that weak proprioceptive signal.
            restored = _return_unintended_grasp(runtime, pick, trace)
            report.setdefault("rejected_false_holds", []).append(
                {
                    "strategy": used,
                    "restored": bool(restored),
                }
            )
            continue
        if _has_transient_spatial_identity(pick_label):
            # The relational phrase (front/back/left/right) names an instance
            # only in the pre-action scene.  Once that grounded instance has
            # been retained and lifted, a post-release crop cannot re-prove
            # the old relation and often calls the partly occluded object a
            # different item.  Persist the causal transport evidence instead.
            report["transport_identity_established"] = True
        held = True
        trace.append({"event": "grasp_attempt", "strategy": used, "at": where, "held": True})
        break

    if not held:
        # The desired postcondition of a failed transport is an empty gripper.
        # ToolSession uses this marker to reassert opening and clear the camera
        # instead of trusting a stale finger-opening "holding" signal.
        report["force_empty_after_contact"] = True
        return fail(
            "empty_grasp",
            f"no attempt in {[f'{s}@{h}' for s, h in attempts]} lifted {pick_label!r}",
            attempts=len(attempts),
        )

    # How far the object's base hangs below the hand. The hand closed at
    # ``height`` on an object standing on ``table_z``, so that difference is
    # the object below the fingers -- which is what has to clear the
    # destination.
    carry = float(np.clip(height - table_z, 0.01, 0.30))
    report["grasp_strategy"] = f"{used}@{where}"
    report["carry_below_hand"] = round(carry, 4)
    carry_offset = (
        _carry_offset(runtime, pick_label, trace, source=pick)
        if compensate and not centre
        else np.zeros(2)
    )
    report["carry_offset"] = [round(float(v), 3) for v in carry_offset]

    if yaw_deg == "auto":
        # Turning the wrist is only worth its risk when both shapes have a
        # direction: a long object going into a long opening.
        yaw = None
        if (
            slot_yaw is not None
            and object_aspect >= ELONGATED
            and _destination_aspect(runtime, goal) >= 1.35
        ):
            yaw = slot_yaw + 0.5 * np.pi
    elif yaw_deg is None:
        yaw = None
    else:
        yaw = float(yaw_deg) * np.pi / 180.0
    report["place_yaw_deg"] = None if yaw is None else round(float(np.degrees(yaw)), 1)

    # Keep a tilted edge pinch rigid during transport.  Levelling it after the
    # lift removes the very vertical opposition that holds a flush thin object
    # and was observed to return the package to its source pose.
    transit_yaw = None

    try:
        combined_nudge = _nudge_vector(runtime, nudge) + closure_nudge
        _place_carefully(
            runtime,
            pick_label,
            goal,
            band,
            carry,
            yaw,
            trace,
            nudge=combined_nudge,
            carry_offset=carry_offset,
            margin=place_margin,
            release_on=release_on,
            centre=centre,
            report=report,
            transit_yaw=transit_yaw,
        )
    except Exception as exc:
        return fail("placement_failed", f"place raised {type(exc).__name__}: {exc}")

    # Re-grounding both object and destination here duplicated the mandatory
    # external ``check`` and sometimes disagreed with it after an occluding
    # container placement.  The transport skill owns execution evidence only:
    # it reached the planned release and opened the gripper.  Placement truth
    # belongs to the fresh post-home check / visual agent.
    report["placement_verification"] = "deferred_to_external_check"
    trace.append({"event": "released", "verification": "deferred"})
    return SkillResult(
        success=True,
        status="released",
        pick_label=pick_label,
        place_label=place_label,
        attempts=1,
        strategy="phase1_policy_api",
        trace=trace,
        report=report,
    )
