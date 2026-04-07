"""
Reward Engineering — The mathematical grader for the auto-scaler agent.

This module implements a multi-objective reward function that balances:
1. Positive signal for serving requests (revenue proxy)
2. Cost penalty for running servers (efficiency)
3. Heavy penalty for dropping requests (SLA violation)
4. Soft penalty for high latency (quality of service)
5. Bonus for maintaining optimal utilization (sweet spot)
"""

from __future__ import annotations

from dataclasses import dataclass

from env.models import Reward


@dataclass
class RewardConfig:
    """Weights and thresholds for the reward function."""

    # Reward weights
    w_served: float = 0.01  # Reward per served request (normalized)
    w_cost: float = 0.5  # Penalty weight for server costs
    w_dropped: float = 5.0  # Penalty weight per dropped request (HEAVY)
    w_latency: float = 0.002  # Penalty weight for latency above threshold
    w_efficiency: float = 0.1  # Bonus weight for optimal utilization

    # Thresholds
    latency_threshold_ms: float = 100.0  # Latency above this incurs penalty
    optimal_cpu_low: float = 0.4  # Optimal utilization lower bound
    optimal_cpu_high: float = 0.75  # Optimal utilization upper bound

    # Normalization
    max_requests_for_norm: float = 500.0  # Used to normalize served requests


def compute_reward(
    served: float,
    dropped: float,
    cost: float,
    latency_ms: float,
    cpu_load: float,
    config: RewardConfig = RewardConfig(),
) -> Reward:
    """
    Compute the structured reward for a single time step.

    The reward function is designed to provide:
    - Dense signal (not sparse) so the agent learns continuously
    - Clear penalties for undesirable behavior
    - Bonus for efficient operation

    Args:
        served: Number of successfully served requests
        dropped: Number of dropped requests
        cost: Operational cost this step
        latency_ms: Average latency in milliseconds
        cpu_load: Current CPU utilization (0-1)
        config: Reward weight configuration

    Returns:
        Reward object with total and component breakdown
    """
    # 1) Served requests — positive signal (normalized)
    served_reward = config.w_served * (served / max(config.max_requests_for_norm, 1.0))

    # 2) Cost penalty — continuous bleed for running servers
    cost_penalty = -config.w_cost * cost

    # 3) Dropped requests — HEAVY penalty (the most important signal)
    drop_penalty = -config.w_dropped * dropped

    # 4) Latency penalty — soft penalty above threshold
    latency_penalty = 0.0
    if latency_ms > config.latency_threshold_ms:
        excess = latency_ms - config.latency_threshold_ms
        latency_penalty = -config.w_latency * excess

    # 5) Efficiency bonus — reward for staying in the "sweet spot"
    efficiency_bonus = 0.0
    if config.optimal_cpu_low <= cpu_load <= config.optimal_cpu_high:
        # Peak bonus at the midpoint of the optimal range
        midpoint = (config.optimal_cpu_low + config.optimal_cpu_high) / 2
        closeness = 1.0 - abs(cpu_load - midpoint) / (config.optimal_cpu_high - config.optimal_cpu_low)
        efficiency_bonus = config.w_efficiency * closeness

    total = served_reward + cost_penalty + drop_penalty + latency_penalty + efficiency_bonus

    return Reward(
        total=total,
        served_reward=served_reward,
        cost_penalty=cost_penalty,
        drop_penalty=drop_penalty,
        latency_penalty=latency_penalty,
        efficiency_bonus=efficiency_bonus,
    )
