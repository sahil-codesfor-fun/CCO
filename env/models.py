"""
Typed Pydantic models for the OpenEnv interface.

Defines Observation, Action, and Reward types with full validation.
"""

from __future__ import annotations

import enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Action Space
# ─────────────────────────────────────────────────────────────────────────────
class ActionType(str, enum.Enum):
    """Discrete action space for the auto-scaler."""

    SCALE_UP_3 = "scale_up_3"
    SCALE_UP_1 = "scale_up_1"
    NO_OP = "no_op"
    SCALE_DOWN_1 = "scale_down_1"
    SCALE_DOWN_3 = "scale_down_3"


ACTION_MAP = {
    ActionType.SCALE_UP_3: 3,
    ActionType.SCALE_UP_1: 1,
    ActionType.NO_OP: 0,
    ActionType.SCALE_DOWN_1: -1,
    ActionType.SCALE_DOWN_3: -3,
}

ACTION_INDEX_MAP = {
    0: ActionType.SCALE_UP_3,
    1: ActionType.SCALE_UP_1,
    2: ActionType.NO_OP,
    3: ActionType.SCALE_DOWN_1,
    4: ActionType.SCALE_DOWN_3,
}


class Action(BaseModel):
    """An action taken by the agent at time step t."""

    action_type: ActionType = Field(
        ...,
        description="The scaling decision: scale_up_3, scale_up_1, no_op, scale_down_1, scale_down_3",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional reasoning for the action (used by LLM agents)",
    )

    @property
    def delta_servers(self) -> int:
        """Number of servers to add (positive) or remove (negative)."""
        return ACTION_MAP[self.action_type]


# ─────────────────────────────────────────────────────────────────────────────
# Observation Space
# ─────────────────────────────────────────────────────────────────────────────
class Observation(BaseModel):
    """
    The telemetry snapshot the agent receives at each time step.

    This represents what a real monitoring system (Prometheus/CloudWatch)
    would expose to an auto-scaling controller.
    """

    timestep: int = Field(
        ...,
        ge=0,
        description="Current discrete time step in the episode",
    )
    incoming_requests: float = Field(
        ...,
        ge=0,
        description="Volume of traffic arriving at this timestep (requests/sec)",
    )
    active_servers: int = Field(
        ...,
        ge=0,
        description="Current number of provisioned server instances",
    )
    warming_up_servers: int = Field(
        ...,
        ge=0,
        description="Servers currently in warm-up phase (not yet serving traffic)",
    )
    cpu_load: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized CPU utilization across the fleet (0.0 to 1.0)",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Current average response latency in milliseconds",
    )
    dropped_requests: float = Field(
        ...,
        ge=0.0,
        description="Number of requests dropped due to capacity overflow",
    )
    served_requests: float = Field(
        ...,
        ge=0.0,
        description="Number of requests successfully served",
    )
    cost_so_far: float = Field(
        ...,
        ge=0.0,
        description="Cumulative operational cost incurred so far",
    )
    traffic_trend: float = Field(
        ...,
        description="Rate of change of traffic (positive = increasing, negative = decreasing)",
    )
    time_of_day: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized time-of-day for seasonality awareness (0.0 = midnight, 0.5 = noon)",
    )

    def to_flat_array(self) -> list[float]:
        """Convert to a flat numeric array for RL algorithms."""
        return [
            float(self.timestep),
            self.incoming_requests,
            float(self.active_servers),
            float(self.warming_up_servers),
            self.cpu_load,
            self.latency_ms,
            self.dropped_requests,
            self.served_requests,
            self.cost_so_far,
            self.traffic_trend,
            self.time_of_day,
        ]

    def to_prompt(self) -> str:
        """Convert to a human-readable prompt string for LLM-based agents."""
        return (
            f"=== Cloud Fleet Telemetry (Step {self.timestep}) ===\n"
            f"  Incoming Requests:   {self.incoming_requests:.1f} req/s\n"
            f"  Active Servers:      {self.active_servers}\n"
            f"  Warming Up Servers:  {self.warming_up_servers}\n"
            f"  CPU Load:            {self.cpu_load:.2%}\n"
            f"  Avg Latency:         {self.latency_ms:.1f} ms\n"
            f"  Dropped Requests:    {self.dropped_requests:.0f}\n"
            f"  Served Requests:     {self.served_requests:.0f}\n"
            f"  Cumulative Cost:     ${self.cost_so_far:.2f}\n"
            f"  Traffic Trend:       {self.traffic_trend:+.1f} req/s/step\n"
            f"  Time of Day:         {self.time_of_day:.2f} (0=midnight, 0.5=noon)\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────────────────────────────────────
class Reward(BaseModel):
    """
    Structured reward signal with breakdown for interpretability.

    The total reward balances:
    - Positive signal for serving requests
    - Negative signal for server costs (efficiency)
    - Heavy penalty for dropping requests (SLA violation)
    - Soft penalty for high latency
    """

    total: float = Field(
        ...,
        description="The scalar reward signal for RL training",
    )
    served_reward: float = Field(
        default=0.0,
        description="Reward component from successfully served requests",
    )
    cost_penalty: float = Field(
        default=0.0,
        description="Penalty component from server operational costs",
    )
    drop_penalty: float = Field(
        default=0.0,
        description="Penalty component from dropped requests (SLA violation)",
    )
    latency_penalty: float = Field(
        default=0.0,
        description="Penalty component from high latency",
    )
    efficiency_bonus: float = Field(
        default=0.0,
        description="Bonus for maintaining optimal server utilization (60-80%)",
    )

    def to_prompt(self) -> str:
        """Human-readable reward breakdown."""
        return (
            f"=== Reward Breakdown ===\n"
            f"  Total Reward:      {self.total:+.4f}\n"
            f"  ├─ Served:         {self.served_reward:+.4f}\n"
            f"  ├─ Cost Penalty:   {self.cost_penalty:+.4f}\n"
            f"  ├─ Drop Penalty:   {self.drop_penalty:+.4f}\n"
            f"  ├─ Latency Pen.:   {self.latency_penalty:+.4f}\n"
            f"  └─ Efficiency:     {self.efficiency_bonus:+.4f}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Episode Info
# ─────────────────────────────────────────────────────────────────────────────
class EpisodeInfo(BaseModel):
    """Summary statistics for a completed episode."""

    total_steps: int = 0
    total_reward: float = 0.0
    total_requests: float = 0.0
    total_served: float = 0.0
    total_dropped: float = 0.0
    total_cost: float = 0.0
    avg_latency: float = 0.0
    max_latency: float = 0.0
    avg_cpu_load: float = 0.0
    sla_compliance_rate: float = 0.0
    avg_active_servers: float = 0.0
    peak_servers: int = 0

    @property
    def drop_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_dropped / self.total_requests

    @property
    def cost_per_request(self) -> float:
        if self.total_served == 0:
            return float("inf")
        return self.total_cost / self.total_served


# ─────────────────────────────────────────────────────────────────────────────
# State (Full internal state for serialization)
# ─────────────────────────────────────────────────────────────────────────────
class EnvironmentState(BaseModel):
    """Full internal state for save/load and debugging."""

    timestep: int = 0
    active_servers: int = 1
    warming_servers: List[int] = Field(default_factory=list)
    total_cost: float = 0.0
    total_served: float = 0.0
    total_dropped: float = 0.0
    total_requests: float = 0.0
    latency_history: List[float] = Field(default_factory=list)
    cpu_history: List[float] = Field(default_factory=list)
    traffic_history: List[float] = Field(default_factory=list)
    server_history: List[int] = Field(default_factory=list)
    reward_history: List[float] = Field(default_factory=list)
    episode_done: bool = False
