"""Evaluator-owned two-layer experience memory for controller evolution.

Raw critic episodes remain immutable evidence.  Reusable lessons are compact
advisory hypotheses with explicit provenance and empirical usefulness.  The
coding agent can read them but candidate code cannot edit this store.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

from .schema import CriticReport, StageSpec


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", value.lower()) if len(token) > 2}


def _lesson_key(report: CriticReport) -> str:
    basis = "|".join(
        (
            report.failure_phase.lower().strip(),
            report.suggested_layer.lower().strip(),
            report.lesson_condition.lower().strip(),
            report.lesson_remedy.lower().strip(),
        )
    )
    return "lesson_" + hashlib.sha256(basis.encode()).hexdigest()[:12]


class ExperienceMemory:
    """Persistent failure episodes plus retrieved, outcome-tracked lessons."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes_path = root / "failure_episodes.jsonl"
        self.lessons_path = root / "lessons.json"
        self.applications_path = root / "applications.jsonl"
        self.capabilities_path = root / "CAPABILITIES.json"
        self.clusters_path = root / "FAILURE_CLUSTERS.json"
        self.ledger_path = root / "EXPERIMENT_LEDGER.jsonl"
        self.gaps_path = root / "OPEN_GAPS.json"

    @staticmethod
    def _read_object(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _write_object(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _cluster_key(report: CriticReport) -> str:
        mechanism = report.failure_mechanism.strip().lower() or report.failure_phase.strip().lower()
        layer = report.suggested_layer.strip().lower() or "unknown"
        tags = sorted(_tokens(" ".join(report.applicable_tags)))[:5]
        suffix = "+".join(tags) or "untagged"
        return f"{mechanism}|{layer}|{suffix}"

    def _lessons(self) -> list[dict[str, Any]]:
        if not self.lessons_path.is_file():
            return []
        try:
            value = json.loads(self.lessons_path.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_lessons(self, lessons: list[dict[str, Any]]) -> None:
        temporary = self.lessons_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(lessons, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self.lessons_path)

    def ingest(
        self,
        reports: Iterable[CriticReport],
        *,
        stage: StageSpec,
        commit: str,
        iteration: int,
        rollout_label: str,
    ) -> None:
        lessons = self._lessons()
        by_id = {lesson["lesson_id"]: lesson for lesson in lessons}
        now = time.time()
        clusters = self._read_object(self.clusters_path)
        gaps = self._read_object(self.gaps_path)
        for report in reports:
            episode = {
                "time": now,
                "stage": stage.id,
                "commit": commit,
                "iteration": iteration,
                "rollout": rollout_label,
                **report.to_dict(),
            }
            with self.episodes_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

            cluster_id = self._cluster_key(report)
            cluster = clusters.setdefault(
                cluster_id,
                {
                    "failure_mechanism": report.failure_mechanism or report.failure_phase,
                    "failure_phase": report.failure_phase,
                    "suggested_layer": report.suggested_layer,
                    "tags": sorted(set(report.applicable_tags)),
                    "count": 0,
                    "episodes": [],
                    "first_divergences": [],
                    "candidate_remedies": [],
                    "instrumentation_requests": [],
                    "last_iteration": iteration,
                },
            )
            cluster["count"] = int(cluster.get("count", 0)) + 1
            cluster["last_iteration"] = iteration
            for key, value in (
                ("episodes", report.episode_key),
                ("first_divergences", report.first_divergence),
                ("candidate_remedies", report.lesson_remedy),
            ):
                items = list(cluster.get(key) or [])
                if value and value not in items:
                    items.append(value)
                cluster[key] = items[-12:]
            requests = list(cluster.get("instrumentation_requests") or [])
            for request in report.recommended_instrumentation:
                if request and request not in requests:
                    requests.append(request)
            cluster["instrumentation_requests"] = requests[-12:]
            gaps[cluster_id] = {
                "cluster": cluster_id,
                "count": cluster["count"],
                "owning_layer": report.suggested_layer,
                "mechanism": report.failure_mechanism or report.hypothesis,
                "latest_hypothesis": report.hypothesis,
                "next_observation": report.recommended_observation,
                "instrumentation": list(report.recommended_instrumentation),
                "last_iteration": iteration,
                "status": "open",
            }

            # A report without a concrete condition/remedy remains raw evidence;
            # it is deliberately not converted into vague reusable advice.
            if not report.lesson_condition or not report.lesson_remedy:
                continue
            lesson_id = _lesson_key(report)
            lesson = by_id.get(lesson_id)
            evidence = {
                "episode_key": report.episode_key,
                "commit": commit,
                "iteration": iteration,
                "rollout": rollout_label,
            }
            if lesson is None:
                lesson = {
                    "lesson_id": lesson_id,
                    "condition": report.lesson_condition,
                    "antipattern": report.lesson_antipattern,
                    "remedy": report.lesson_remedy,
                    "failure_phase": report.failure_phase,
                    "suggested_layer": report.suggested_layer,
                    "scope": report.generality_scope,
                    "tags": sorted(set(report.applicable_tags)),
                    "critic_confidence": report.confidence,
                    "evidence": [evidence],
                    "times_served": 0,
                    "times_helped": 0,
                    "times_hurt": 0,
                    "recent_outcomes": [],
                }
                lessons.append(lesson)
                by_id[lesson_id] = lesson
            elif evidence not in lesson["evidence"]:
                lesson["evidence"].append(evidence)
                count = len(lesson["evidence"])
                lesson["critic_confidence"] = (
                    (float(lesson.get("critic_confidence", 0.0)) * (count - 1))
                    + report.confidence
                ) / count
                lesson["tags"] = sorted(set(lesson.get("tags", ())) | set(report.applicable_tags))
        self._save_lessons(lessons)
        self._write_object(self.clusters_path, clusters)
        self._write_object(self.gaps_path, gaps)
        self._render_markdown()

    def retrieve(
        self,
        stage: StageSpec,
        reports: Iterable[CriticReport],
        *,
        limit: int = 8,
    ) -> tuple[dict[str, Any], ...]:
        query = " ".join(
            (
                stage.id,
                stage.title,
                stage.capability,
                stage.prompt,
                *(report.failure_phase for report in reports),
                *(report.generality_scope for report in reports),
                *(report.hypothesis for report in reports),
            )
        )
        query_tokens = _tokens(query)
        ranked: list[tuple[float, dict[str, Any]]] = []
        lessons = self._lessons()
        for lesson in lessons:
            lesson_text = " ".join(
                str(lesson.get(key, ""))
                for key in ("condition", "antipattern", "remedy", "failure_phase", "scope")
            ) + " " + " ".join(lesson.get("tags", ()))
            overlap = len(query_tokens & _tokens(lesson_text))
            served = int(lesson.get("times_served", 0))
            helped = int(lesson.get("times_helped", 0))
            hurt = int(lesson.get("times_hurt", 0))
            reliability = (helped + 1.0) / (served + 2.0)
            evidence_bonus = min(3, len(lesson.get("evidence", ()))) * 0.5
            score = overlap + evidence_bonus + reliability - 0.5 * hurt
            if overlap or lesson.get("failure_phase") in {
                report.failure_phase for report in reports
            }:
                ranked.append((score, lesson))
        ranked.sort(key=lambda item: (item[0], item[1]["lesson_id"]), reverse=True)
        selected = [dict(lesson) for _, lesson in ranked[:limit]]
        selected_ids = {lesson["lesson_id"] for lesson in selected}
        if selected_ids:
            for lesson in lessons:
                if lesson["lesson_id"] in selected_ids:
                    lesson["times_served"] = int(lesson.get("times_served", 0)) + 1
            self._save_lessons(lessons)
        return tuple(selected)

    def record_application(
        self,
        lesson_ids: Iterable[str],
        *,
        iteration: int,
        candidate: str,
        native_delta: int,
        new_wins: Iterable[str],
        regressions: Iterable[str],
    ) -> None:
        ids = set(lesson_ids)
        if not ids:
            return
        outcome = {
            "time": time.time(),
            "iteration": iteration,
            "candidate": candidate,
            "lesson_ids": sorted(ids),
            "native_delta": native_delta,
            "new_wins": list(new_wins),
            "regressions": list(regressions),
            "attribution": "correlational: lessons were supplied to the coding agent",
        }
        with self.applications_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(outcome, ensure_ascii=False) + "\n")
        lessons = self._lessons()
        for lesson in lessons:
            if lesson["lesson_id"] not in ids:
                continue
            if native_delta > 0:
                lesson["times_helped"] = int(lesson.get("times_helped", 0)) + 1
            elif native_delta < 0:
                lesson["times_hurt"] = int(lesson.get("times_hurt", 0)) + 1
            recent = list(lesson.get("recent_outcomes", ()))
            recent.append(
                {
                    "iteration": iteration,
                    "native_delta": native_delta,
                    "candidate": candidate,
                }
            )
            lesson["recent_outcomes"] = recent[-12:]
        self._save_lessons(lessons)

    def record_candidate(
        self,
        *,
        stage: StageSpec,
        iteration: int,
        parent: str,
        candidate: str,
        title: str,
        mechanism: str,
        status: str,
        native_delta: int,
        new_wins: Iterable[str],
        regressions: Iterable[str],
        changed_files: Iterable[str],
    ) -> None:
        """Append the durable experiment ledger and update capability evidence."""
        row = {
            "time": time.time(),
            "iteration": iteration,
            "stage": stage.id,
            "parent": parent,
            "candidate": candidate,
            "title": title,
            "mechanism": mechanism,
            "status": status,
            "native_delta": int(native_delta),
            "new_wins": list(new_wins),
            "regressions": list(regressions),
            "changed_files": list(changed_files),
        }
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        capabilities = self._read_object(self.capabilities_path)
        capability = capabilities.setdefault(
            stage.capability,
            {
                "stage": stage.id,
                "status": "observed",
                "best_native_delta": 0,
                "successful_mechanisms": [],
                "failed_or_neutral_mechanisms": [],
                "wins": [],
                "regressions": [],
                "last_iteration": iteration,
            },
        )
        capability["last_iteration"] = iteration
        capability["best_native_delta"] = max(
            int(capability.get("best_native_delta", 0)), int(native_delta)
        )
        target_key = "successful_mechanisms" if native_delta > 0 else "failed_or_neutral_mechanisms"
        mechanisms = list(capability.get(target_key) or [])
        summary = mechanism or title
        if summary and summary not in mechanisms:
            mechanisms.append(summary)
        capability[target_key] = mechanisms[-16:]
        if native_delta > 0:
            capability["status"] = "empirically_improved"
        for key, values in (("wins", new_wins), ("regressions", regressions)):
            existing = list(capability.get(key) or [])
            for value in values:
                if value not in existing:
                    existing.append(value)
            capability[key] = existing[-24:]
        self._write_object(self.capabilities_path, capabilities)
        self._render_markdown()

    def snapshot(
        self,
        *,
        stage: StageSpec | None = None,
        reports: Iterable[CriticReport] = (),
        recent_ledger: int = 12,
        cluster_limit: int = 12,
    ) -> dict[str, Any]:
        ledger: list[dict[str, Any]] = []
        if self.ledger_path.is_file():
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines()[-recent_ledger:]:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    ledger.append(value)
        clusters = self._read_object(self.clusters_path)
        gaps = self._read_object(self.gaps_path)
        if stage is not None:
            report_values = tuple(reports)
            query_tokens = _tokens(
                " ".join(
                    (
                        stage.id,
                        stage.title,
                        stage.capability,
                        stage.prompt,
                        *(report.failure_phase for report in report_values),
                        *(report.failure_mechanism for report in report_values),
                        *(" ".join(report.applicable_tags) for report in report_values),
                    )
                )
            )
            ranked = sorted(
                clusters.items(),
                key=lambda pair: (
                    len(query_tokens & _tokens(pair[0] + " " + json.dumps(pair[1]))),
                    int(pair[1].get("count", 0)),
                    pair[0],
                ),
                reverse=True,
            )[: max(0, int(cluster_limit))]
            clusters = dict(ranked)
            gaps = {key: gaps[key] for key in clusters if key in gaps}
        return {
            "capabilities": self._read_object(self.capabilities_path),
            "failure_clusters": clusters,
            "open_gaps": gaps,
            "recent_experiments": ledger,
        }

    def _render_markdown(self) -> None:
        """Keep the machine state easy for humans and coding agents to audit."""
        capabilities = self._read_object(self.capabilities_path)
        clusters = self._read_object(self.clusters_path)
        gaps = self._read_object(self.gaps_path)

        capability_lines = ["# Capability map", ""]
        for name, item in sorted(capabilities.items()):
            capability_lines.extend(
                [
                    f"## {name}",
                    "",
                    f"- Status: {item.get('status', 'observed')}",
                    f"- Best native delta: {item.get('best_native_delta', 0)}",
                    f"- Successful mechanisms: {json.dumps(item.get('successful_mechanisms', []), ensure_ascii=False)}",
                    f"- Neutral/regressed mechanisms: {json.dumps(item.get('failed_or_neutral_mechanisms', []), ensure_ascii=False)}",
                    "",
                ]
            )
        (self.root / "CAPABILITIES.md").write_text("\n".join(capability_lines), encoding="utf-8")

        cluster_lines = ["# Failure clusters", ""]
        for name, item in sorted(clusters.items(), key=lambda pair: int(pair[1].get("count", 0)), reverse=True):
            cluster_lines.extend(
                [
                    f"## {name}",
                    "",
                    f"- Count: {item.get('count', 0)}",
                    f"- First divergences: {json.dumps(item.get('first_divergences', []), ensure_ascii=False)}",
                    f"- Candidate remedies: {json.dumps(item.get('candidate_remedies', []), ensure_ascii=False)}",
                    "",
                ]
            )
        (self.root / "FAILURE_CLUSTERS.md").write_text("\n".join(cluster_lines), encoding="utf-8")

        gap_lines = ["# Open capability gaps", ""]
        for name, item in sorted(gaps.items(), key=lambda pair: int(pair[1].get("count", 0)), reverse=True):
            gap_lines.extend(
                [
                    f"## {name}",
                    "",
                    f"- Owning layer: {item.get('owning_layer', 'unknown')}",
                    f"- Latest hypothesis: {item.get('latest_hypothesis', '')}",
                    f"- Next observation: {item.get('next_observation', '')}",
                    "",
                ]
            )
        (self.root / "OPEN_GAPS.md").write_text("\n".join(gap_lines), encoding="utf-8")
