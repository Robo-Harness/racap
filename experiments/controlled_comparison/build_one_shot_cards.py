#!/usr/bin/env python3
"""Build bounded, visual one-trial experience cards from calibration rollouts."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from racap.backends.llm import LLMQuotaError, ask

try:
    from .analyze_results import (
        METHODS,
        _manifest_index,
        _parse_capx,
        _parse_racap,
        _parse_rats,
    )
except ImportError:  # direct script execution
    from analyze_results import (  # type: ignore[no-redef]
        METHODS,
        _manifest_index,
        _parse_capx,
        _parse_racap,
        _parse_rats,
    )

FAMILY_SUITE = {
    "spatial": "libero_spatial_swap",
    "goal": "libero_goal_swap",
    "object": "libero_object_swap",
}
FORBIDDEN_TRACE_PATTERNS = (
    "predicate",
    "satisfied_conditions",
    "unsatisfied_conditions",
    "goal_state",
    "private_native_audit",
    "parsed_problem",
    "_eval_predicate",
    "native_state",
    "native_success",
    "object_pose",
    "ground_truth",
)
CARD_HEADINGS = (
    "## Observed mechanism",
    "## Transferable guidance",
    "## Avoid",
    "## Uncertainty",
)
CARD_FALLBACKS = {
    "## Observed mechanism": "The retained public evidence is incomplete.",
    "## Transferable guidance": "Re-observe the current scene before acting.",
    "## Avoid": "Do not treat this single interaction as a fixed action sequence.",
    "## Uncertainty": (
        "This is one calibration interaction; current visual evidence must override it."
    ),
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibration_rows(
    input_root: Path,
    input_cohort: str = "one_shot_calibration",
) -> list[dict[str, Any]]:
    manifest, _ = _manifest_index()
    rows: list[dict[str, Any]] = []
    rows.extend(_parse_capx(input_root / "capx", manifest))
    rows.extend(_parse_rats(input_root / "rats_base", "rats_base", manifest))
    rows.extend(_parse_rats(input_root / "rats_90", "rats_90", manifest))
    rows.extend(_parse_racap(input_root / "racap_phase1", "racap_phase1", manifest))
    rows.extend(_parse_racap(input_root / "racap_phase2", "racap_phase2", manifest))
    selected = [row for row in rows if row.get("cohort") == input_cohort]
    if input_cohort == "libero_pro_zero_shot":
        selected = [
            row
            for row in selected
            if row.get("suite") in FAMILY_SUITE.values()
            and int(row.get("task_id", -1)) == 0
            and int(row.get("seed", -1)) == 0
        ]
    return selected


def _calibration_rows_from_index(
    path: Path,
    input_cohort: str,
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[dict[str, Any]] = []
    for raw in rows:
        if raw.get("cohort") != input_cohort:
            continue
        row: dict[str, Any] = dict(raw)
        row["task_id"] = int(raw.get("task_id") or -1)
        row["seed"] = int(raw.get("seed") or -1)
        row["native_success"] = str(raw.get("native_success") or "").lower() in {
            "1",
            "true",
            "yes",
        }
        selected.append(row)
    return selected


def _episode_root(row: dict[str, Any]) -> Path:
    artifact = Path(str(row["artifact_path"]))
    root = artifact if artifact.is_dir() else artifact.parent
    # RATS normalization points at ``task_root/artifacts/iteration_*.json``.
    # The public run log is one directory above ``artifacts`` and the videos
    # are below it, so use the full task directory as the evidence root.
    if root.name == "artifacts":
        root = root.parent
    marker = f"t{int(row['task_id']):03d}_seed{int(row['seed'])}"
    matches = [path for path in root.rglob(marker) if path.is_dir()]
    return matches[0] if matches else root


def _select_video(root: Path) -> Path | None:
    videos = [path for path in root.rglob("*.mp4") if path.is_file()]
    if not videos:
        return None
    return max(videos, key=lambda path: path.stat().st_size)


def _frame_urls(video: Path | None, output_dir: Path) -> tuple[list[str], list[str]]:
    if video is None:
        return [], []
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        capture.release()
        return [], []
    indices = sorted({0, count // 2, max(0, count - 1)})
    urls: list[str] = []
    paths: list[str] = []
    for order, index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 260, 28), fill=(0, 0, 0))
        draw.text((8, 7), f"CALIBRATION {order + 1}/{len(indices)}", fill=(255, 255, 255))
        path = output_dir / f"frame_{order}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        urls.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
        paths.append(str(path.resolve()))
    capture.release()
    return urls, paths


def _public_trace(root: Path) -> str:
    candidates = [
        *root.rglob("trace.md"),
        *root.rglob("trajectory.md"),
        *root.rglob("run.log"),
    ]
    chunks: list[str] = []
    for path in sorted(set(candidates))[:8]:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        public = [
            line
            for line in lines
            if not any(pattern in line.lower() for pattern in FORBIDDEN_TRACE_PATTERNS)
        ]
        if public:
            chunks.append(f"FILE {path.name}:\n" + "\n".join(public[-250:]))
    trace = "\n\n".join(chunks)
    # Defense in depth for exact goal-condition strings that may be printed
    # without the literal word "predicate" on the same line.
    trace = re.sub(
        r"\[(?:on|in|inside|open|close|closed|turnon|turnoff|stack)\s+[^\]]+\]",
        "<redacted-evaluator-condition>",
        trace,
        flags=re.IGNORECASE,
    )
    # Remove obvious absolute paths and keep the critic context bounded.
    trace = re.sub(r"/mnt/data/\S+", "<artifact-path>", trace)
    return trace[-24_000:]


def _card_prompt(row: dict[str, Any], trace: str) -> str:
    return f"""Create a transferable one-trial robot experience card.

Target family: {str(row['suite']).split('_')[1]}
Calibration instruction: {row.get('instruction', '')}
Final benchmark success bit: {bool(row.get('native_success'))}

PUBLIC AGENT TRACE (may be incomplete):
{trace or '(no readable text trace; rely on the ordered frames)'}

Use the ordered first/middle/final frames and public trace to identify what
worked, the earliest visible failure if any, and advice likely to transfer to
other tasks in the same family. Do not mention hidden predicates, simulator
state, exact coordinates, task IDs, seeds, or a memorized action sequence.
Current observations at test time must override this single-example prior.

Return Markdown under these exact headings, at most 400 words total:
## Observed mechanism
## Transferable guidance
## Avoid
## Uncertainty
"""


def _bounded_card(card: str, maximum_words: int = 400) -> str:
    """Preserve the registered four-section schema under the word budget."""

    sections: dict[str, str] = {}
    positions: list[tuple[int, str]] = []
    for heading in CARD_HEADINGS:
        match = re.search(re.escape(heading), card, flags=re.IGNORECASE)
        if match:
            positions.append((match.start(), heading))
    positions.sort()
    for index, (start, heading) in enumerate(positions):
        content_start = start + len(heading)
        content_end = positions[index + 1][0] if index + 1 < len(positions) else len(card)
        sections[heading] = card[content_start:content_end].strip()

    contents = {
        heading: (sections.get(heading) or CARD_FALLBACKS[heading]).split()
        for heading in CARD_HEADINGS
    }
    heading_words = sum(len(heading.split()) for heading in CARD_HEADINGS)
    full = "\n\n".join(
        heading + "\n" + " ".join(contents[heading])
        for heading in CARD_HEADINGS
    ).strip()
    if len(full.split()) <= maximum_words:
        return full

    # Reserve a common minimum for every section, then distribute the
    # remaining words round-robin. Short sections return unused capacity to
    # longer ones, preserving more evidence than four rigid equal caps.
    content_budget = max(len(CARD_HEADINGS), maximum_words - heading_words)
    allocations = {
        heading: min(40, len(contents[heading])) for heading in CARD_HEADINGS
    }
    remaining = content_budget - sum(allocations.values())
    while remaining > 0:
        advanced = False
        for heading in CARD_HEADINGS:
            if allocations[heading] < len(contents[heading]):
                allocations[heading] += 1
                remaining -= 1
                advanced = True
                if remaining == 0:
                    break
        if not advanced:
            break
    rendered: list[str] = []
    for heading in CARD_HEADINGS:
        rendered.append(
            heading + "\n" + " ".join(contents[heading][: allocations[heading]])
        )
    bounded = "\n\n".join(rendered).strip()
    # The arithmetic above should already enforce the limit; retain this
    # assertion so a future schema edit cannot silently violate the protocol.
    if len(bounded.split()) > maximum_words:
        raise RuntimeError("section-aware experience-card bound was exceeded")
    return bounded


def _normalize_existing_cards(
    cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in cards:
        path = Path(str(row.get("card") or ""))
        if not path.is_file():
            continue
        bounded = _bounded_card(path.read_text(encoding="utf-8"))
        if path.read_text(encoding="utf-8").strip() != bounded:
            path.write_text(bounded + "\n", encoding="utf-8")
        current = dict(row)
        current["word_count"] = len(bounded.split())
        current["card_sha256"] = _sha256(path)
        normalized.append(current)
    return normalized


def _manifest_payload(
    *,
    args: argparse.Namespace,
    calibration_identity: list[dict[str, str]],
    cards: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "model": args.model,
        "temperature": 0.0,
        "input_root": str(args.input_root.resolve()),
        "input_index": (
            str(args.input_index.resolve()) if args.input_index is not None else None
        ),
        "input_index_sha256": (
            _sha256(args.input_index) if args.input_index is not None else None
        ),
        "input_cohort": args.input_cohort,
        "calibration_selector": {
            "suites": sorted(FAMILY_SUITE.values()),
            "task_id": 0,
            "seed": 0,
        },
        "calibration_identity": calibration_identity,
        "predicate_text_visible_to_critic": False,
        "complete": complete,
        "cards": cards,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "one_shot" / "calibration",
    )
    parser.add_argument(
        "--input-index",
        type=Path,
        help="optional normalized 15-row calibration index to avoid scanning full runs",
    )
    parser.add_argument(
        "--input-cohort",
        choices=["one_shot_calibration", "libero_pro_zero_shot"],
        default="one_shot_calibration",
        help=(
            "cohort containing the registered task-0/seed-0 calibration episodes; "
            "zero-shot reuse preserves their original measured resource budget"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "one_shot" / "cards",
    )
    parser.add_argument("--model", default="gpt-5.5")
    args = parser.parse_args()
    if not os.environ.get("RACAP_VAPI_KEY") or not os.environ.get("RACAP_VAPI_BASE"):
        raise SystemExit("RACAP_VAPI_KEY and RACAP_VAPI_BASE must be configured")
    if args.input_index is not None:
        if not args.input_index.is_file():
            raise SystemExit(f"calibration index does not exist: {args.input_index}")
        rows = _calibration_rows_from_index(args.input_index, args.input_cohort)
    else:
        rows = (
            _calibration_rows(args.input_root)
            if args.input_cohort == "one_shot_calibration"
            else _calibration_rows(args.input_root, args.input_cohort)
        )
    index = {(row["method"], row["suite"]): row for row in rows}
    missing = [
        f"{method}/{suite}"
        for method in METHODS
        for suite in FAMILY_SUITE.values()
        if (method, suite) not in index
    ]
    if missing:
        raise SystemExit("missing calibration results: " + ", ".join(missing))
    if len(rows) != len(METHODS) * len(FAMILY_SUITE):
        raise SystemExit(
            f"calibration index must contain exactly 15 selected rows, got {len(rows)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    calibration_identity = []
    for row in rows:
        evidence = Path(str(row["evidence_path"]))
        if not evidence.is_file():
            raise SystemExit(f"missing calibration evidence: {evidence}")
        calibration_identity.append(
            {
                "method": str(row["method"]),
                "suite": str(row["suite"]),
                "episode_key": str(row["episode_key"]),
                "evidence_path": str(evidence.resolve()),
                "evidence_sha256": _sha256(evidence),
            }
        )
    calibration_identity.sort(key=lambda row: (row["method"], row["suite"]))
    manifests: list[dict[str, Any]] = []
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_binding = {
            "model": args.model,
            "input_root": str(args.input_root.resolve()),
            "input_index": (
                str(args.input_index.resolve()) if args.input_index is not None else None
            ),
            "input_index_sha256": (
                _sha256(args.input_index) if args.input_index is not None else None
            ),
            "input_cohort": args.input_cohort,
            "calibration_identity": calibration_identity,
        }
        actual_binding = {key: previous.get(key) for key in expected_binding}
        if actual_binding != expected_binding:
            raise SystemExit("refusing incompatible one-trial card resume")
        manifests = _normalize_existing_cards([
            row
            for row in previous.get("cards") or []
            if Path(str(row.get("card") or "")).is_file()
        ])
        if previous.get("complete") is True and len(manifests) == len(METHODS) * 3:
            _write_json(
                manifest_path,
                _manifest_payload(
                    args=args,
                    calibration_identity=calibration_identity,
                    cards=manifests,
                    complete=True,
                ),
            )
            return 0

    os.environ["RACAP_LLM_TELEMETRY_PATH"] = str(args.output_dir / "critic_calls.jsonl")
    completed = {(str(row["method"]), str(row["family"])) for row in manifests}
    for method in METHODS:
        for family, suite in FAMILY_SUITE.items():
            if (method, family) in completed:
                continue
            row = index[(method, suite)]
            episode_root = _episode_root(row)
            video = _select_video(episode_root)
            if video is None:
                raise SystemExit(
                    f"registered visual calibration episode has no video: "
                    f"{method}/{family} at {episode_root}"
                )
            media_dir = args.output_dir / "evidence" / method / family
            images, frame_paths = _frame_urls(video, media_dir)
            os.environ["RACAP_EPISODE_KEY"] = f"one_shot_card/{method}/{family}"
            try:
                card = ask(
                    "You are a rigorous visual robot-learning critic. Use only public evidence.",
                    _card_prompt(row, _public_trace(episode_root)),
                    images=images,
                    model=args.model,
                    max_tokens=1_200,
                    temperature=0.0,
                    cache=False,
                    attempts=2,
                    timeout_s=240,
                ).strip()
            except LLMQuotaError as exc:
                _write_json(
                    args.output_dir / "ABORTED.json",
                    {
                        "reason": "quota",
                        "method": method,
                        "family": family,
                        "completed_cards": len(manifests),
                        "error": str(exc),
                    },
                )
                _write_json(
                    manifest_path,
                    _manifest_payload(
                        args=args,
                        calibration_identity=calibration_identity,
                        cards=manifests,
                        complete=False,
                    ),
                )
                return 75
            card = _bounded_card(card)
            word_count = len(card.split())
            card_path = args.output_dir / f"{method}_{family}.md"
            card_path.write_text(card + "\n", encoding="utf-8")
            manifests.append(
                {
                    "method": method,
                    "family": family,
                    "suite": suite,
                    "instruction": row.get("instruction", ""),
                    "native_success_bit": bool(row.get("native_success")),
                    "episode_artifact": str(episode_root.resolve()),
                    "video": str(video.resolve()) if video else None,
                    "video_sha256": _sha256(video),
                    "keyframes": frame_paths,
                    "card": str(card_path.resolve()),
                    "word_count": word_count,
                    "card_sha256": _sha256(card_path),
                }
            )
            _write_json(
                manifest_path,
                _manifest_payload(
                    args=args,
                    calibration_identity=calibration_identity,
                    cards=manifests,
                    complete=False,
                ),
            )
    manifests.sort(key=lambda row: (str(row["method"]), str(row["family"])))
    _write_json(
        manifest_path,
        _manifest_payload(
            args=args,
            calibration_identity=calibration_identity,
            cards=manifests,
            complete=True,
        ),
    )
    aborted = args.output_dir / "ABORTED.json"
    if aborted.is_file():
        archive = args.output_dir / "interrupted" / f"ABORTED_{int(time.time())}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        aborted.replace(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
