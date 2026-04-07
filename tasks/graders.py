"""
Task Graders — Deterministic scoring functions for each task.

Each grader evaluates agent performance on a 0.0–1.0 scale based on:
1. SLA Compliance Rate (drop rate + latency compliance)
2. Cost Efficiency (vs worst/best case baselines)
3. Latency Performance (average and P99)
4. Scaling Stability (action frequency / oscillation)

Graders are deterministic and reproducible given the same episode history.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.models import EpisodeInfo


@dataclass
class GradingConfig:
    """Weights and thresholds loaded from task YAML."""

    sla_weight: float = 0.40
    cost_weight: float = 0.30
    latency_weight: float = 0.20
    stability_weight: float = 0.10
    worst_case_cost: float = 180.0
    best_case_cost: float = 36.0
    latency_threshold_ms: float = 200.0
    max_acceptable_drop_rate: float = 0.05


def load_grading_config(task_yaml_path: str) -> GradingConfig:
    """Load grading configuration from a task YAML file."""
    with open(task_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    grading = data.get("grading", {})
    metrics = grading.get("metrics", {})

    return GradingConfig(
        sla_weight=metrics.get("sla_weight", 0.40),
        cost_weight=metrics.get("cost_weight", 0.30),
        latency_weight=metrics.get("latency_weight", 0.20),
        stability_weight=metrics.get("stability_weight", 0.10),
        worst_case_cost=grading.get("worst_case_cost", 180.0),
        best_case_cost=grading.get("best_case_cost", 36.0),
        latency_threshold_ms=grading.get("latency_threshold_ms", 200.0),
        max_acceptable_drop_rate=grading.get("max_acceptable_drop_rate", 0.05),
    )


def grade_episode(
    episode_info: EpisodeInfo,
    history: Dict[str, list],
    config: GradingConfig,
) -> Dict[str, Any]:
    """
    Grade an episode deterministically on a 0.0–1.0 scale.

    Args:
        episode_info: Summary statistics from the episode
        history: Full history dict from env.get_history()
        config: Grading weights and thresholds

    Returns:
        Dict with total_score (0.0-1.0) and component scores
    """
    # ── 1. SLA Compliance Score ──────────────────────────────────────────
    # Based on: drop rate + latency compliance rate
    drop_rate = episode_info.drop_rate
    if drop_rate <= 0.0:
        drop_score = 1.0
    elif drop_rate <= config.max_acceptable_drop_rate:
        drop_score = 1.0 - (drop_rate / config.max_acceptable_drop_rate) * 0.5
    else:
        # Rapid degradation beyond threshold
        drop_score = max(0.0, 0.5 - (drop_rate - config.max_acceptable_drop_rate) * 5.0)

    latency_compliance = episode_info.sla_compliance_rate
    sla_score = 0.6 * drop_score + 0.4 * latency_compliance

    # ── 2. Cost Efficiency Score ─────────────────────────────────────────
    # Normalized between worst-case and best-case costs
    total_cost = episode_info.total_cost
    if total_cost <= config.best_case_cost:
        cost_score = 1.0
    elif total_cost >= config.worst_case_cost:
        cost_score = 0.0
    else:
        cost_score = 1.0 - (total_cost - config.best_case_cost) / (config.worst_case_cost - config.best_case_cost)

    # ── 3. Latency Performance Score ─────────────────────────────────────
    avg_latency = episode_info.avg_latency
    max_latency = episode_info.max_latency
    threshold = config.latency_threshold_ms

    # Average latency score
    if avg_latency <= threshold * 0.3:
        avg_lat_score = 1.0
    elif avg_latency <= threshold:
        avg_lat_score = 1.0 - 0.5 * ((avg_latency - threshold * 0.3) / (threshold * 0.7))
    else:
        avg_lat_score = max(0.0, 0.5 - (avg_latency - threshold) / (threshold * 2))

    # Max latency score (penalty for extreme spikes)
    if max_latency <= threshold:
        max_lat_score = 1.0
    elif max_latency <= threshold * 3:
        max_lat_score = 1.0 - (max_latency - threshold) / (threshold * 2)
    else:
        max_lat_score = 0.0

    latency_score = 0.7 * avg_lat_score + 0.3 * max_lat_score

    # ── 4. Scaling Stability Score ───────────────────────────────────────
    # Penalize excessive oscillation (rapid up/down/up/down)
    servers = history.get("servers", [])
    if len(servers) >= 2:
        changes = [abs(servers[i] - servers[i - 1]) for i in range(1, len(servers))]
        avg_change = np.mean(changes)
        change_count = sum(1 for c in changes if c > 0)
        change_ratio = change_count / len(changes)

        # Good stability: few changes, small magnitudes
        if avg_change <= 0.1:
            stability_score = 1.0
        elif avg_change <= 0.5:
            stability_score = 0.8
        elif avg_change <= 1.0:
            stability_score = 0.6
        else:
            stability_score = max(0.0, 0.6 - (avg_change - 1.0) * 0.2)

        # Also penalize rapid oscillation (direction changes)
        if len(servers) >= 3:
            direction_changes = 0
            for i in range(2, len(servers)):
                d1 = servers[i - 1] - servers[i - 2]
                d2 = servers[i] - servers[i - 1]
                if d1 * d2 < 0:  # Direction changed
                    direction_changes += 1
            oscillation_ratio = direction_changes / (len(servers) - 2)
            if oscillation_ratio > 0.3:
                stability_score *= max(0.5, 1.0 - oscillation_ratio)
    else:
        stability_score = 0.5

    # ── Composite Score ────────────────────────────────────────────────
    total_score = (
        config.sla_weight * sla_score
        + config.cost_weight * cost_score
        + config.latency_weight * latency_score
        + config.stability_weight * stability_score
    )

    # Clamp to [0, 1]
    total_score = max(0.0, min(1.0, total_score))

    return {
        "total_score": round(total_score, 4),
        "sla_score": round(sla_score, 4),
        "cost_score": round(cost_score, 4),
        "latency_score": round(latency_score, 4),
        "stability_score": round(stability_score, 4),
        "details": {
            "drop_rate": round(drop_rate, 6),
            "total_cost": round(total_cost, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "max_latency_ms": round(max_latency, 2),
            "sla_compliance_rate": round(episode_info.sla_compliance_rate, 4),
            "total_served": round(episode_info.total_served, 0),
            "total_dropped": round(episode_info.total_dropped, 0),
            "avg_servers": round(episode_info.avg_active_servers, 2),
            "peak_servers": episode_info.peak_servers,
        },
        "weights": {
            "sla": config.sla_weight,
            "cost": config.cost_weight,
            "latency": config.latency_weight,
            "stability": config.stability_weight,
        },
    }


def grade_task(
    task_yaml_path: str,
    episode_info: EpisodeInfo,
    history: Dict[str, list],
) -> Dict[str, Any]:
    """
    High-level grading function: loads config from YAML and grades.

    Returns dict with total_score (0.0-1.0) and breakdown.
    """
    config = load_grading_config(task_yaml_path)
    return grade_episode(episode_info, history, config)
