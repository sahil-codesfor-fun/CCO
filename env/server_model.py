"""
Server Fleet Model — Simulates cloud server infrastructure.

Models realistic server characteristics:
- Maximum capacity per server (requests/sec)
- Operational cost per time step
- Warm-up delay (servers take time to become active)
- Exponential latency near saturation
- Request dropping when capacity is exceeded
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ServerConfig:
    """Configuration for the server fleet."""

    capacity_per_server: float = 50.0  # Max requests/sec per server
    cost_per_server_per_step: float = 0.1  # Operational cost per server per time step
    warmup_steps: int = 3  # Steps needed for a new server to become active
    base_latency_ms: float = 10.0  # Base latency when load is near 0
    latency_growth_factor: float = 5.0  # k in latency = base * exp(load * k)
    max_servers: int = 50  # Hard cap on total servers
    min_servers: int = 1  # Minimum servers (always at least 1)
    sla_latency_threshold_ms: float = 200.0  # SLA: latency above this = violation


class ServerFleet:
    """
    Simulates a fleet of cloud servers.

    Handles:
    - Server provisioning with warm-up delay
    - De-provisioning (instant)
    - Load distribution across active servers
    - Latency modeling (exponential near saturation)
    - Request serving and dropping
    - Cost calculation
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.active_servers: int = config.min_servers
        self._warming_queue: List[int] = []  # remaining warmup steps for each pending server

    @property
    def warming_up_count(self) -> int:
        return len(self._warming_queue)

    @property
    def total_capacity(self) -> float:
        return self.active_servers * self.config.capacity_per_server

    def scale(self, delta: int) -> int:
        """
        Apply a scaling action.

        Args:
            delta: Number of servers to add (positive) or remove (negative).

        Returns:
            Actual delta applied (may be clipped to min/max bounds).
        """
        if delta > 0:
            # Scale UP: new servers go into warm-up queue
            can_add = self.config.max_servers - self.active_servers - self.warming_up_count
            actual_add = min(delta, max(0, can_add))
            for _ in range(actual_add):
                self._warming_queue.append(self.config.warmup_steps)
            return actual_add
        elif delta < 0:
            # Scale DOWN: instantly de-provision (but respect minimum)
            can_remove = self.active_servers - self.config.min_servers
            actual_remove = min(abs(delta), can_remove)
            self.active_servers -= actual_remove
            return -actual_remove
        return 0

    def tick(self) -> None:
        """Advance one time step: process warm-up queue."""
        new_queue = []
        for remaining in self._warming_queue:
            remaining -= 1
            if remaining <= 0:
                # Server is now active
                self.active_servers = min(self.active_servers + 1, self.config.max_servers)
            else:
                new_queue.append(remaining)
        self._warming_queue = new_queue

    def process_requests(self, incoming: float) -> Tuple[float, float, float, float]:
        """
        Process incoming requests through the fleet.

        Args:
            incoming: Number of incoming requests/sec.

        Returns:
            Tuple of (served, dropped, cpu_load, latency_ms)
        """
        capacity = self.total_capacity

        if capacity <= 0:
            return 0.0, incoming, 1.0, float("inf")

        # CPU load = utilization ratio
        cpu_load = min(incoming / capacity, 1.0)

        # Served vs dropped
        served = min(incoming, capacity)
        dropped = max(0.0, incoming - capacity)

        # Latency model: exponential growth near saturation
        # latency = base * exp(cpu_load * k)
        # At low load: ~base_latency; at high load: explodes
        latency = self.config.base_latency_ms * math.exp(cpu_load * self.config.latency_growth_factor)

        # Cap latency to something reasonable for display
        latency = min(latency, 10000.0)

        return served, dropped, cpu_load, latency

    def get_cost(self) -> float:
        """Get operational cost for this time step."""
        # Active servers cost money; warming servers cost half (they're booting)
        active_cost = self.active_servers * self.config.cost_per_server_per_step
        warming_cost = self.warming_up_count * self.config.cost_per_server_per_step * 0.5
        return active_cost + warming_cost

    def is_sla_violated(self, latency_ms: float) -> bool:
        """Check if the latency exceeds SLA threshold."""
        return latency_ms > self.config.sla_latency_threshold_ms

    def reset(self, initial_servers: int = 1) -> None:
        """Reset fleet to initial state."""
        self.active_servers = max(initial_servers, self.config.min_servers)
        self._warming_queue = []

    def get_state_dict(self) -> dict:
        """Serialize fleet state."""
        return {
            "active_servers": self.active_servers,
            "warming_queue": list(self._warming_queue),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore fleet state."""
        self.active_servers = state["active_servers"]
        self._warming_queue = state["warming_queue"]
