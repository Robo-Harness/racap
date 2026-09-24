"""VLM grounding primitives: bounding boxes and affordance points from language.

SAM3's text prompt segments whatever matches a phrase, which breaks down exactly
where LIBERO is hardest: `libero_spatial` puts two identical black bowls in one
scene and distinguishes them only by a spatial relation. A VLM can resolve
"the bowl between the plate and the ramekin" to a region; SAM3 can then produce a
precise mask inside it. This module supplies the first half of that pipeline.

Which VLM to use is a parameter, not a constant, so that model choice becomes
one of the things the coding agent can tune during evolution.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# The grounding model is configurable independently from the runtime planner.
DEFAULT_GROUNDER = os.environ.get("RACAP_GROUNDER_MODEL", "gpt-5.5")


class PerceptionUnavailableError(RuntimeError):
    """The visual service produced no usable evidence for a grounding call.

    This is deliberately different from a valid ``found: false`` answer.  A
    policy may recover from an absent or occluded object by looking elsewhere;
    it cannot reason its way around an HTTP failure or malformed model reply.
    """

    def __init__(self, kind: str, operation: str, model: str, detail: str):
        self.kind = kind
        self.operation = operation
        self.model = model
        self.detail = detail[:500]
        super().__init__(f"{kind} during {operation} with {model}: {self.detail}")

    def report(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "operation": self.operation,
            "model": self.model,
            "detail": self.detail,
        }


def _perception_error(operation: str, model: str, exc: Exception) -> PerceptionUnavailableError:
    from racap.backends.llm import LLMError, LLMQuotaError

    if isinstance(exc, PerceptionUnavailableError):
        return exc
    if isinstance(exc, LLMQuotaError):
        kind = "provider_quota"
    elif isinstance(exc, LLMError):
        kind = "provider_unavailable"
    else:
        kind = "invalid_response"
    return PerceptionUnavailableError(kind, operation, model, f"{type(exc).__name__}: {exc}")

BBOX_SYSTEM = """\
You locate objects in robot camera images. You answer only with JSON.
"""

BBOX_PROMPT = """\
Find the single object best described by: "{label}"

The image is {width} pixels wide and {height} pixels tall.

Reply with exactly this JSON and nothing else:
{{"found": true/false, "x0": <int>, "y0": <int>, "x1": <int>, "y1": <int>, "confidence": <0..1>}}

Coordinates are pixels in the image, with (0,0) at the top-left corner.
(x0,y0) is the top-left of the box and (x1,y1) the bottom-right.
If the description mentions a spatial relation, use it to choose between similar
objects. If it names a part, opening, drawer, shelf, or compartment, tightly box
only that named subregion rather than the whole parent fixture. When a shelf,
rack, or compartment is a placement destination, box its empty usable interior
opening in the robot's manipulable foreground workspace, not architectural
storage in the distant background, a nearby stored object, its outer wall, or
the whole furniture.
If no object matches, reply {{"found": false}}.
"""

SEARCH_BBOX_PROMPT = """\
Find the single physical object best described by: "{label}"

The ordinary detector did not return a usable region, so inspect the complete
image carefully before deciding that the object is absent. Include thin,
edge-on, dark, partially occluded, upright, and visually plain objects. A book
may present only its cover edge or spine and need not contain readable text.
Use the task's ordinary household category, geometry, and scene context; do not
silently rename a narrow object as a brush or generic rectangular item.

The image is {width} pixels wide and {height} pixels tall.

Reply with exactly this JSON and nothing else:
{{"found": true/false, "x0": <int>, "y0": <int>, "x1": <int>, "y1": <int>, "confidence": <0..1>}}

Coordinates are pixels in the image, with (0,0) at the top-left corner.
(x0,y0) is the top-left of a tight object box and (x1,y1) the bottom-right.
Box only the requested object, not the support surface or a neighbouring item.
If no visible physical object matches after careful inspection, reply
{{"found": false}}.
"""

REVISED_BBOX_PROMPT = """\
Find the single object best described by: "{label}"

An independent close-up visual check established that every red crossed box in
the image is a wrong candidate. Locate a DIFFERENT physical object that matches
the complete description. The red marks are feedback annotations, not objects
or part of the scene. Preserve every spatial or relational qualifier in the
description, and do not return a box overlapping a red crossed region.
For a robot placement destination, prefer the manipulable foreground workspace
fixture over architectural cabinets or shelves in the distant background. If
the label denotes a shelf, rack, drawer, or compartment, box the requested
empty usable opening/tier rather than a nearby object or the whole furniture.

The image is {width} pixels wide and {height} pixels tall.

Reply with exactly this JSON and nothing else:
{{"found": true/false, "x0": <int>, "y0": <int>, "x1": <int>, "y1": <int>, "confidence": <0..1>}}

Coordinates are pixels in the image, with (0,0) at the top-left corner.
(x0,y0) is the top-left of the box and (x1,y1) the bottom-right.
If there is no different visible object satisfying the full description, reply
{{"found": false}} rather than selecting a rejected or merely similar object.
"""

POINT_PROMPT = """\
Point at the best place for a parallel-jaw gripper to grasp: "{label}"

The image is {width} pixels wide and {height} pixels tall.

Reply with exactly this JSON and nothing else:
{{"found": true/false, "x": <int>, "y": <int>, "confidence": <0..1>}}

Choose a point on a graspable part of the object, not on its shadow or on the
surface beneath it.
"""


def _four_numbers(item: dict) -> list[float] | None:
    """Pull a box out of whatever shape the model chose to emit.

    Models disagree on how to serialise a box and several ignore the requested
    schema in favour of the one they were trained on. Qwen returns the four
    corners as a single array, sometimes under `bbox_2d`, sometimes packed into
    the first key of the requested schema. Rejecting those replies would
    discard an otherwise accurate detection, so read them all.
    """
    for key in ("bbox_2d", "bbox", "box"):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            try:
                return [float(v) for v in value[:4]]
            except (TypeError, ValueError):
                return None

    first = item.get("x0")
    if isinstance(first, (list, tuple)) and len(first) >= 4:
        try:
            return [float(v) for v in first[:4]]
        except (TypeError, ValueError):
            return None

    try:
        return [float(item["x0"]), float(item["y0"]), float(item["x1"]), float(item["y1"])]
    except (KeyError, TypeError, ValueError):
        return None


def _rescale_if_normalised(
    boxes: list[list[float]], width: int, height: int
) -> tuple[list[list[float]], bool]:
    """Convert 0-1000 normalised coordinates to pixels when that is what we got.

    Qwen models emit boxes on a 0-1000 grid regardless of the true image size,
    a convention they were trained with and do not abandon when asked. The
    decision is made once for the whole reply rather than per box: a single
    out-of-frame corner is much more likely to mean the wrong convention than a
    genuinely out-of-frame object, and mixing conventions within one reply does
    not happen.
    """
    if not boxes:
        return boxes, False
    overshoot = any(
        b[0] > width * 1.02 or b[2] > width * 1.02 or b[1] > height * 1.02 or b[3] > height * 1.02
        for b in boxes
    )
    if not overshoot:
        return boxes, False
    sx, sy = width / 1000.0, height / 1000.0
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes], True


@dataclass(frozen=True)
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float

    def clipped(self, width: int, height: int, margin: int = 8) -> "BBox":
        """Widen slightly and clamp, so SAM3 sees a little context around the object."""
        return BBox(
            max(0, min(self.x0, self.x1) - margin),
            max(0, min(self.y0, self.y1) - margin),
            min(width, max(self.x0, self.x1) + margin),
            min(height, max(self.y0, self.y1) + margin),
            self.confidence,
        )

    @property
    def area(self) -> int:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)


def encode_png(rgb: np.ndarray) -> str:
    from PIL import Image

    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        array = np.clip(array * 255, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _ask_json(
    system: str,
    prompt: str,
    rgb: np.ndarray,
    model: str,
    *,
    max_tokens: int = 1024,
    cache: bool = True,
) -> dict:
    from racap.backends.llm import ask, cached_answer

    # Routed through racap.backends.llm so every grounding call is cached and a
    # replayed episode sees byte-identical perception. temperature=0 alone does
    # not give that: hosted models still vary between identical requests, which
    # is precisely why measuring how repeatable a grounding is needs
    # ``cache=False`` -- otherwise all the repeats are one draw.
    images = [encode_png(rgb)]
    reply = (
        cached_answer(
            system,
            prompt,
            images=images,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        if cache
        else None
    )
    if reply is None:
        # Evaluation hosts occasionally exhaust one provider while the local
        # gateway remains healthy.  An opt-in cache-miss route lets existing
        # high-quality groundings stay byte-identical and sends only genuinely
        # unseen frames to an available fallback, instead of waiting through
        # repeated quota failures.  It is deliberately not enabled by default.
        fallback = os.environ.get("RACAP_GROUNDER_CACHE_MISS_MODEL", "").strip()
        effective_model = fallback or model
        reply = ask(
            system,
            prompt,
            images=images,
            model=effective_model,
            max_tokens=max_tokens,
            temperature=0.0,
            cache=cache,
        )
    text = (reply or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"grounder returned no JSON: {text[:200]!r}")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError(f"grounder returned {type(value).__name__}, expected a JSON object")
    return value


def identify(
    rgb: np.ndarray,
    *,
    options: list[str] | None = None,
    model: str | None = None,
    upscale: int = 4,
    cache: bool = True,
) -> str:
    """Name the object in a crop, optionally from a closed set.

    Meant for a small patch cut out of a scene, which is why it upsamples: a
    30-pixel-wide can carries its label in a handful of pixels, and enlarging
    it is what lets the model read the text rather than guess from colour.

    Offering ``options`` and requiring a choice matters more than it sounds.
    Asked openly, a model invents plausible product names; invited to abstain,
    it abstains even when the label is legible. Forcing a pick from a known
    list turns the question into the one actually being asked -- which of these
    is it -- and it answers that one well.
    """
    from PIL import Image

    model = model or DEFAULT_GROUNDER
    array = np.asarray(rgb)
    if upscale > 1:
        image = Image.fromarray(array.astype(np.uint8))
        array = np.asarray(
            image.resize((image.width * upscale, image.height * upscale), Image.LANCZOS)
        )

    if options:
        listing = ", ".join(f'"{o}"' for o in options)
        prompt = (
            f"This is a close crop of one object. Which of these is it: {listing}?\n"
            "Answer with exactly one of those strings and nothing else. You must "
            "choose the closest match even if you are unsure."
        )
    else:
        prompt = (
            "This is a close crop of one object. Name it in at most four words, "
            "using the wording a household inventory would use. Answer with the "
            "name only."
        )

    from racap.backends.llm import ask

    reply = ask(
        "You identify objects in close-up crops. You answer with a name and nothing else.",
        prompt,
        images=[encode_png(array)],
        model=model,
        max_tokens=200,
        temperature=0.0,
        cache=cache,
    )
    answer = (reply or "").strip().strip('".')
    if options:
        lowered = answer.lower()
        for option in options:
            if option.lower() in lowered or lowered in option.lower():
                return option
    return answer


def bbox(
    rgb: np.ndarray,
    label: str,
    *,
    model: str = DEFAULT_GROUNDER,
    exhaustive: bool = False,
) -> BBox | None:
    """Ask a VLM for the bounding box of the object described by ``label``."""
    height, width = np.asarray(rgb).shape[:2]
    try:
        template = SEARCH_BBOX_PROMPT if exhaustive else BBOX_PROMPT
        data = _ask_json(
            BBOX_SYSTEM,
            template.format(label=label, width=width, height=height),
            rgb,
            model,
        )
    except Exception as exc:
        raise _perception_error("bbox", model, exc) from exc
    if not data.get("found"):
        return None
    corners = _four_numbers(data)
    if corners is None:
        raise PerceptionUnavailableError(
            "invalid_response", "bbox", model, "found=true but no valid box coordinates"
        )
    (corners,), _ = _rescale_if_normalised([corners], width, height)
    box = BBox(
        int(corners[0]),
        int(corners[1]),
        int(corners[2]),
        int(corners[3]),
        float(data.get("confidence", 0.5)),
    ).clipped(width, height)
    return box if box.area >= 16 else None


def _marked_rejections(rgb: np.ndarray, rejected: list[BBox]) -> np.ndarray:
    """Mark wrong candidates while retaining the scene's relational context."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(rgb).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(3, int(round(min(image.size) / 90)))
    for box in rejected:
        x0 = max(0, min(box.x0, box.x1))
        y0 = max(0, min(box.y0, box.y1))
        x1 = min(image.width - 1, max(box.x0, box.x1))
        y1 = min(image.height - 1, max(box.y0, box.y1))
        draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=line_width)
        draw.line((x0, y0, x1, y1), fill=(255, 0, 0), width=line_width)
        draw.line((x0, y1, x1, y0), fill=(255, 0, 0), width=line_width)
    return np.asarray(image)


def _substantial_overlap(box: BBox, rejected: BBox) -> bool:
    """Whether a revision still names the already rejected physical patch."""
    ix0, iy0 = max(box.x0, rejected.x0), max(box.y0, rejected.y0)
    ix1, iy1 = min(box.x1, rejected.x1), min(box.y1, rejected.y1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if intersection / max(1, box.area) >= 0.40:
        return True
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    return rejected.x0 <= cx <= rejected.x1 and rejected.y0 <= cy <= rejected.y1


def bbox_with_feedback(
    rgb: np.ndarray,
    label: str,
    rejected: list[BBox],
    *,
    model: str = DEFAULT_GROUNDER,
    cache: bool = True,
) -> BBox | None:
    """Re-ground ``label`` after visual verification rejected prior boxes.

    This stays open-vocabulary: the controller neither enumerates instances
    nor interprets front/back with a hand-written camera-axis rule. The VLM
    receives the complete phrase plus visually explicit negative evidence.
    """
    if not rejected:
        return bbox(rgb, label, model=model)
    height, width = np.asarray(rgb).shape[:2]
    try:
        data = _ask_json(
            BBOX_SYSTEM,
            REVISED_BBOX_PROMPT.format(
                label=label,
                width=width,
                height=height,
            ),
            _marked_rejections(rgb, rejected),
            model,
            cache=cache,
        )
    except Exception as exc:
        raise _perception_error("bbox_with_feedback", model, exc) from exc
    if not data.get("found"):
        return None
    corners = _four_numbers(data)
    if corners is None:
        raise PerceptionUnavailableError(
            "invalid_response",
            "bbox_with_feedback",
            model,
            "found=true but no valid box coordinates",
        )
    (corners,), _ = _rescale_if_normalised([corners], width, height)
    revised = BBox(
        int(corners[0]),
        int(corners[1]),
        int(corners[2]),
        int(corners[3]),
        float(data.get("confidence", 0.5)),
    ).clipped(width, height)
    if revised.area < 16 or any(_substantial_overlap(revised, old) for old in rejected):
        return None
    return revised


INVENTORY_PROMPT = """\
List every manipulable object on the surface in this robot camera image.

Use short, concrete noun phrases a person would use, for example "milk carton",
"wicker basket", "blue can". Include small or partially occluded objects: look
behind and between the prominent ones. Exclude the robot arm, the floor, walls,
and shadows.

Reply with exactly this JSON and nothing else:
{"objects": ["<name>", "<name>", ...]}
"""


def inventory(rgb: np.ndarray, *, model: str = DEFAULT_GROUNDER) -> list[str]:
    """Open-vocabulary list of the manipulable objects the camera can see.

    Gives a policy the scene contents it needs to disambiguate one object from
    its lookalikes, without any privileged access to the simulator's object
    list.
    """
    try:
        data = _ask_json(BBOX_SYSTEM, INVENTORY_PROMPT, rgb, model)
    except Exception as exc:
        raise _perception_error("inventory", model, exc) from exc
    if not isinstance(data.get("objects"), list):
        raise PerceptionUnavailableError(
            "invalid_response", "inventory", model, "missing JSON objects list"
        )
    names = []
    for item in data.get("objects", []):
        name = str(item).strip()
        if name and name.lower() not in {n.lower() for n in names}:
            names.append(name)
    return names


MANY_PROMPT = """\
This robot camera image contains these objects, each appearing exactly once:

{inventory}

Draw one bounding box per object. The image is {width} pixels wide and {height} \
pixels tall, with (0,0) at the top-left.

Every object must get a different box: no two objects share a location. Some \
objects partially occlude others, so look carefully for small or half-hidden \
items rather than assigning two names to the same prominent object.

Reply with exactly this JSON and nothing else:
{{"objects": [{{"name": "<one of the names above>", "x0": <int>, "y0": <int>, \
"x1": <int>, "y1": <int>, "confidence": <0..1>}}, ...]}}

Omit an object only if it is genuinely not visible.
"""


def detect_many(
    rgb: np.ndarray,
    labels: list[str],
    *,
    model: str = DEFAULT_GROUNDER,
    cache: bool = True,
) -> dict[str, BBox]:
    """Box every named object in one call, under a no-shared-box constraint.

    The constraint helps only when the names describe genuinely different
    objects. Measured on libero_object, one label alone was correct 5 times in
    6, two labels 3 in 6, and a full inventory 2 in 5, because generic names
    like "blue can" alias the target and exclusion then pushes it elsewhere.

    Note also what the prompt asserts: that every listed object appears exactly
    once. A model that cannot find a small or occluded target is being told one
    is definitely present, and it will box the most target-like thing it can
    see rather than omit it. Two of the three grounding failures measured take
    this shape.
    """
    height, width = np.asarray(rgb).shape[:2]
    inventory = "\n".join(f"- {name}" for name in labels)
    try:
        data = _ask_json(
            BBOX_SYSTEM,
            MANY_PROMPT.format(inventory=inventory, width=width, height=height),
            rgb,
            model,
            # One box costs roughly 60 tokens. A reply truncated mid-JSON parses
            # as nothing at all, which reads as "the model found no objects" and
            # silently blames the model for our budget.
            # Reasoning-capable VLMs may spend most of the completion budget on
            # hidden reasoning before emitting the short JSON payload.  Keep a
            # generous envelope so a syntactically correct answer is not cut
            # mid-object; models still stop as soon as the JSON is complete.
            max_tokens=max(4096, 600 + 120 * len(labels)),
            cache=cache,
        )
    except Exception as exc:
        raise _perception_error("detect_many", model, exc) from exc
    if not isinstance(data.get("objects"), list):
        raise PerceptionUnavailableError(
            "invalid_response", "detect_many", model, "missing JSON objects list"
        )

    known = {name.lower(): name for name in labels}
    parsed: list[tuple[str, list[float], float]] = []
    for item in data.get("objects", []):
        if not isinstance(item, dict):
            continue
        name = known.get(str(item.get("name", "")).strip().lower())
        corners = _four_numbers(item) if name else None
        if name is None or corners is None or name in {p[0] for p in parsed}:
            continue
        try:
            score = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            score = 0.5
        parsed.append((name, corners, score))

    corners_only, _ = _rescale_if_normalised([p[1] for p in parsed], width, height)
    found: dict[str, BBox] = {}
    for (name, _, score), corners in zip(parsed, corners_only):
        box = BBox(
            int(corners[0]), int(corners[1]), int(corners[2]), int(corners[3]), score
        ).clipped(width, height)
        if box.area >= 16:
            found[name] = box
    return found


def grasp_point(
    rgb: np.ndarray, label: str, *, model: str = DEFAULT_GROUNDER
) -> tuple[int, int, float] | None:
    """Ask a VLM for a pixel to grasp on the object described by ``label``."""
    height, width = np.asarray(rgb).shape[:2]
    try:
        data = _ask_json(
            BBOX_SYSTEM,
            POINT_PROMPT.format(label=label, width=width, height=height),
            rgb,
            model,
        )
    except Exception as exc:
        raise _perception_error("grasp_point", model, exc) from exc
    if not data.get("found"):
        return None
    try:
        x, y = int(data["x"]), int(data["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PerceptionUnavailableError(
            "invalid_response", "grasp_point", model, f"invalid point coordinates: {exc}"
        ) from exc
    if not (0 <= x < width and 0 <= y < height):
        return None
    return x, y, float(data.get("confidence", 0.5))


def probe_grounder(model: str = DEFAULT_GROUNDER) -> dict[str, Any]:
    """Make one uncached visual request before an expensive rollout batch."""
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[6:18, 6:18, 0] = 255
    try:
        data = _ask_json(
            "You are a health probe. Return JSON only.",
            'Ignore image content and reply exactly {"status":"ok"}.',
            rgb,
            model,
            max_tokens=80,
            cache=False,
        )
    except Exception as exc:
        raise _perception_error("health_probe", model, exc) from exc
    if str(data.get("status", "")).lower() != "ok":
        raise PerceptionUnavailableError(
            "invalid_response", "health_probe", model, f"unexpected payload: {data!r}"
        )
    return {"status": "ok", "model": model}
