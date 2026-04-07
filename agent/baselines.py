"""
Baseline Agents — Simple heuristic policies for comparison.

These baselines are MANDATORY for hackathon scoring:
1. Static Scaling — Fixed number of servers (naive approach)
2. Threshold-Based Scaling — Rule-based reactive scaling (industry standard)
3. Predictive Heuristic — Trend-following proactive scaling
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.models import Action, ActionType, Observation


class BaselineAgent(ABC):
    """Abstract base for all baseline agents."""

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Select an action given an observation."""
        ...

    @abstractmethod
    def name(self) -> str: ...


class StaticAgent(BaselineAgent):
    """
    Baseline 1: Static Scaling.

    Keeps a fixed number of servers regardless of traffic.
    This is the "lazy IT admin" approach — simple but wasteful.
    """

    def __init__(self, target_servers: int = 5):
        self.target_servers = target_servers

    def act(self, observation: Observation) -> Action:
        current = observation.active_servers
        if current < self.target_servers:
            return Action(action_type=ActionType.SCALE_UP_1, reasoning="Below target count")
        elif current > self.target_servers:
            return Action(action_type=ActionType.SCALE_DOWN_1, reasoning="Above target count")
        return Action(action_type=ActionType.NO_OP, reasoning="At target count")

    def name(self) -> str:
        return f"Static({self.target_servers})"


class ThresholdAgent(BaselineAgent):
    """
    Baseline 2: Threshold-Based Reactive Scaling.

    Classic auto-scaler logic (similar to Kubernetes HPA):
    - If CPU > high_threshold → scale up
    - If CPU < low_threshold → scale down
    - Otherwise → hold

    This is the industry-standard reactive approach.
    """

    def __init__(
        self,
        high_threshold: float = 0.75,
        low_threshold: float = 0.30,
        scale_up_amount: int = 1,
        scale_down_amount: int = 1,
        cooldown_steps: int = 3,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.scale_up_amount = scale_up_amount
        self.scale_down_amount = scale_down_amount
        self.cooldown_steps = cooldown_steps
        self._last_scale_step = -100

    def act(self, observation: Observation) -> Action:
        step = observation.timestep

        # Cooldown: don't scale too rapidly
        if step - self._last_scale_step < self.cooldown_steps:
            return Action(action_type=ActionType.NO_OP, reasoning="Cooldown period")

        cpu = observation.cpu_load

        if cpu > self.high_threshold:
            self._last_scale_step = step
            if self.scale_up_amount >= 3:
                return Action(action_type=ActionType.SCALE_UP_3, reasoning=f"CPU {cpu:.0%} > {self.high_threshold:.0%}")
            return Action(action_type=ActionType.SCALE_UP_1, reasoning=f"CPU {cpu:.0%} > {self.high_threshold:.0%}")

        if cpu < self.low_threshold and observation.active_servers > 1:
            self._last_scale_step = step
            return Action(action_type=ActionType.SCALE_DOWN_1, reasoning=f"CPU {cpu:.0%} < {self.low_threshold:.0%}")

        return Action(action_type=ActionType.NO_OP, reasoning=f"CPU {cpu:.0%} in safe range")

    def name(self) -> str:
        return f"Threshold({self.high_threshold:.0%}/{self.low_threshold:.0%})"


class PredictiveHeuristicAgent(BaselineAgent):
    """
    Baseline 3: Predictive Heuristic Scaling.

    Uses traffic trend to proactively scale:
    - Positive trend + high load → scale up aggressively
    - Negative trend + low load → scale down cautiously
    - Also considers time of day for daily patterns

    This is a smarter heuristic that anticipates demand changes.
    """

    def __init__(
        self,
        trend_threshold: float = 5.0,
        high_cpu: float = 0.65,
        low_cpu: float = 0.25,
        capacity_per_server: float = 50.0,
    ):
        self.trend_threshold = trend_threshold
        self.high_cpu = high_cpu
        self.low_cpu = low_cpu
        self.capacity_per_server = capacity_per_server

    def act(self, observation: Observation) -> Action:
        cpu = observation.cpu_load
        trend = observation.traffic_trend
        incoming = observation.incoming_requests
        active = observation.active_servers
        warming = observation.warming_up_servers

        # Estimate needed capacity
        needed_servers = max(1, int(np.ceil(incoming / self.capacity_per_server)))
        total_planned = active + warming

        # Proactive: if traffic is rising and we're getting loaded
        if trend > self.trend_threshold and cpu > self.high_cpu:
            if total_planned < needed_servers + 2:
                return Action(
                    action_type=ActionType.SCALE_UP_3, reasoning=f"Rising traffic (trend={trend:+.1f}), high CPU"
                )
            return Action(action_type=ActionType.SCALE_UP_1, reasoning=f"Rising traffic, approaching capacity")

        # Reactive: CPU is already high
        if cpu > 0.85:
            return Action(action_type=ActionType.SCALE_UP_3, reasoning=f"Critical CPU load: {cpu:.0%}")
        if cpu > self.high_cpu:
            return Action(action_type=ActionType.SCALE_UP_1, reasoning=f"High CPU: {cpu:.0%}")

        # Proactive scale down: traffic is falling and we have excess
        if trend < -self.trend_threshold and cpu < self.low_cpu and active > needed_servers + 1:
            return Action(action_type=ActionType.SCALE_DOWN_1, reasoning=f"Falling traffic, low CPU")

        # Conservative scale down
        if cpu < 0.15 and active > 2:
            return Action(action_type=ActionType.SCALE_DOWN_1, reasoning=f"Very low utilization: {cpu:.0%}")

        return Action(action_type=ActionType.NO_OP, reasoning="Stable conditions")

    def name(self) -> str:
        return "PredictiveHeuristic"


# ─────────────────────────────────────────────────────────────────────────────
# Runner for baseline evaluation
# ─────────────────────────────────────────────────────────────────────────────


def run_baseline(
    agent: BaselineAgent,
    env,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a baseline agent through a full episode.

    Returns:
        Dict with episode info, history, and grading results
    """
    obs_arr, info = env.reset(seed=seed)
    obs = env._get_observation()

    total_reward = 0.0
    steps = 0

    while True:
        action = agent.act(obs)
        obs_arr, reward, terminated, truncated, info = env.step(action)
        obs = env._get_observation()
        total_reward += reward
        steps += 1

        if verbose and steps % 50 == 0:
            print(
                f"  Step {steps}: servers={obs.active_servers}, "
                f"cpu={obs.cpu_load:.1%}, dropped={obs.dropped_requests:.0f}, "
                f"reward={reward:.4f}"
            )

        if terminated or truncated:
            break

    episode_info = env.get_episode_info()
    history = env.get_history()

    return {
        "agent_name": agent.name(),
        "total_reward": total_reward,
        "episode_info": episode_info,
        "history": history,
        "steps": steps,
    }
