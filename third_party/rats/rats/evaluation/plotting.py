"""Plotting utilities for RATS evaluation metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_metrics(summary: dict[str, Any], output_dir: str = "outputs/rats_lifelong") -> None:
    """Generate metric plots from lifelong loop summary."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics = summary.get("metrics", {})

    # 1. Cumulative success rate
    cumulative = metrics.get("cumulative_success_rates", [])
    if cumulative:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cumulative Success Rate")
        ax.set_title("RATS Cumulative Success Rate")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.savefig(output_path / "cumulative_success_rate.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 2. Skill library growth
    growth = metrics.get("skill_library_growth", [])
    if growth:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(growth) + 1), growth, marker="s", color="green")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total Learned Skills")
        ax.set_title("Skill Library Growth")
        ax.grid(True, alpha=0.3)
        fig.savefig(output_path / "skill_library_growth.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Per-task success rate bar chart
    per_task = metrics.get("per_task_stats", {})
    if per_task:
        fig, ax = plt.subplots(figsize=(max(8, len(per_task) * 1.5), 4))
        tasks = sorted(per_task.keys())
        rates = [per_task[t]["success_rate"] for t in tasks]
        counts = [per_task[t]["attempts"] for t in tasks]
        bars = ax.bar(range(len(tasks)), rates, color="steelblue")
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8)
        ax.set_ylabel("Success Rate")
        ax.set_title("Per-Task Success Rate")
        ax.set_ylim(0, 1.05)
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={count}", ha="center", va="bottom", fontsize=7,
            )
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(output_path / "per_task_success_rate.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. Retry efficiency over time
    retry_eff = metrics.get("retry_efficiency", [])
    if retry_eff:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(retry_eff) + 1), retry_eff, marker="^", color="orange")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Avg Retries (rolling window)")
        ax.set_title("Retry Efficiency Over Time")
        ax.grid(True, alpha=0.3)
        fig.savefig(output_path / "retry_efficiency.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 5. Skill reuse frequency
    skill_reuse = metrics.get("skill_reuse", {})
    per_skill = skill_reuse.get("per_skill", {})
    failed_per_skill = skill_reuse.get("failed_per_skill", {})
    if per_skill or failed_per_skill:
        all_names = set(per_skill) | set(failed_per_skill)
        skills = sorted(
            all_names,
            key=lambda k: per_skill.get(k, 0) + failed_per_skill.get(k, 0),
            reverse=True,
        )[:15]
        fig, ax = plt.subplots(figsize=(max(8, len(skills) * 1.2), 4))
        success_counts = [per_skill.get(s, 0) for s in skills]
        failed_counts = [failed_per_skill.get(s, 0) for s in skills]
        ax.barh(
            range(len(skills)),
            success_counts,
            color="mediumseagreen",
            label="used in final success",
        )
        ax.barh(
            range(len(skills)),
            failed_counts,
            left=success_counts,
            color="indianred",
            label="implicated in final failure",
        )
        ax.set_yticks(range(len(skills)))
        ax.set_yticklabels(skills, fontsize=8)
        ax.set_xlabel("Iteration Count")
        ax.set_title("Skill Usage Frequency (top 15)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(fontsize=8)
        fig.savefig(output_path / "skill_reuse.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Plots saved to {output_path}")
