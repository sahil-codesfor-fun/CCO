"""
Traffic Generator — Realistic workload patterns for the cloud simulator.

Models three real-world traffic phenomena:
1. Daily seasonality (sinusoidal with peak at business hours)
2. Flash-sale / viral spikes (sudden bursts)
3. Random noise (natural variance)

Each task profile configures different traffic characteristics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SpikeEvent:
    """A scheduled traffic spike (e.g., flash sale, viral moment)."""

    start_step: int
    duration: int
    magnitude: float  # multiplier on base traffic

    def is_active(self, t: int) -> bool:
        return self.start_step <= t < self.start_step + self.duration

    def get_contribution(self, t: int) -> float:
        if not self.is_active(t):
            return 0.0
        # Bell-curve shape within the spike
        mid = self.start_step + self.duration / 2
        sigma = self.duration / 4
        return self.magnitude * math.exp(-0.5 * ((t - mid) / max(sigma, 1)) ** 2)


@dataclass
class TrafficConfig:
    """Configuration for a traffic generation profile."""

    base_load: float = 100.0  # Average requests/sec
    amplitude: float = 50.0  # Seasonal swing amplitude
    period: float = 288.0  # Steps per full day cycle (e.g., 288 = 5-min intervals)
    noise_std: float = 10.0  # Standard deviation of random noise
    spike_events: List[SpikeEvent] = field(default_factory=list)
    trend: float = 0.0  # Linear trend (gradual growth/decline)
    min_traffic: float = 5.0  # Floor — never goes below this
    phase_shift: float = 0.0  # Phase for seasonality (0 = peak at step=period/4)
    burst_probability: float = 0.02  # Chance of random micro-burst per step
    burst_magnitude: float = 80.0  # Size of random micro-bursts
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            self._rng = random.Random(self.seed)
        else:
            self._rng = random.Random()


class TrafficGenerator:
    """
    Generates realistic traffic workloads for the cloud simulation.

    Combines multiple signal components:
    - Sinusoidal seasonality (models day/night patterns)
    - Scheduled spike events (models flash sales)
    - Random micro-bursts (models viral content)
    - Gaussian noise (models natural variance)
    - Linear trend (models organic growth)
    """

    def __init__(self, config: TrafficConfig):
        self.config = config
        self._rng = config._rng
        self._previous_traffic = config.base_load

    def generate(self, t: int) -> float:
        """Generate traffic volume at time step t."""
        cfg = self.config

        # 1) Seasonal component (daily pattern)
        seasonal = cfg.amplitude * math.sin(2 * math.pi * (t + cfg.phase_shift) / cfg.period)

        # 2) Scheduled spike events
        spike_total = sum(spike.get_contribution(t) for spike in cfg.spike_events)

        # 3) Random micro-bursts
        burst = 0.0
        if self._rng.random() < cfg.burst_probability:
            burst = self._rng.uniform(0.5, 1.0) * cfg.burst_magnitude

        # 4) Gaussian noise
        noise = self._rng.gauss(0, cfg.noise_std)

        # 5) Linear trend
        trend = cfg.trend * t

        # Combine all components
        traffic = cfg.base_load + seasonal + spike_total + burst + noise + trend
        traffic = max(traffic, cfg.min_traffic)

        self._previous_traffic = traffic
        return traffic

    def get_trend(self, t: int, window: int = 5) -> float:
        """Estimate traffic trend by comparing consecutive samples."""
        if t < window:
            return 0.0
        current = self.generate(t)
        past = self.generate(t - window)
        return (current - past) / window

    def generate_episode(self, total_steps: int) -> List[float]:
        """Pre-generate an entire episode of traffic data."""
        return [self.generate(t) for t in range(total_steps)]

    def reset(self, seed: Optional[int] = None):
        """Reset the generator state."""
        if seed is not None:
            self._rng = random.Random(seed)
            self.config.seed = seed
        self._previous_traffic = self.config.base_load


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built traffic profiles for the 3 tasks
# ─────────────────────────────────────────────────────────────────────────────


def steady_traffic_config(seed: int = 42) -> TrafficConfig:
    """Task 1: Steady, predictable traffic with mild seasonality."""
    return TrafficConfig(
        base_load=100.0,
        amplitude=30.0,
        period=288,
        noise_std=5.0,
        spike_events=[],
        trend=0.0,
        burst_probability=0.0,
        burst_magnitude=0.0,
        seed=seed,
    )


def spike_traffic_config(seed: int = 42) -> TrafficConfig:
    """Task 2: Traffic with scheduled spike events (flash sales)."""
    spikes = [
        SpikeEvent(start_step=50, duration=30, magnitude=200.0),
        SpikeEvent(start_step=150, duration=20, magnitude=300.0),
        SpikeEvent(start_step=250, duration=40, magnitude=250.0),
    ]
    return TrafficConfig(
        base_load=120.0,
        amplitude=40.0,
        period=288,
        noise_std=8.0,
        spike_events=spikes,
        trend=0.0,
        burst_probability=0.01,
        burst_magnitude=50.0,
        seed=seed,
    )


def chaos_traffic_config(seed: int = 42) -> TrafficConfig:
    """Task 3: Chaotic, non-stationary traffic — the hardest challenge."""
    spikes = [
        SpikeEvent(start_step=30, duration=15, magnitude=350.0),
        SpikeEvent(start_step=80, duration=25, magnitude=200.0),
        SpikeEvent(start_step=130, duration=10, magnitude=500.0),
        SpikeEvent(start_step=200, duration=35, magnitude=300.0),
        SpikeEvent(start_step=270, duration=20, magnitude=400.0),
        SpikeEvent(start_step=330, duration=15, magnitude=250.0),
    ]
    return TrafficConfig(
        base_load=150.0,
        amplitude=60.0,
        period=200,  # Shorter, more erratic cycle
        noise_std=20.0,  # High noise
        spike_events=spikes,
        trend=0.1,  # Gradual organic growth
        burst_probability=0.05,
        burst_magnitude=120.0,
        seed=seed,
    )
