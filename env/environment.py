"""
CloudCostOptimizerEnv — The core OpenEnv-compatible RL environment.

Implements the full environment interface:
- reset()  → returns initial observation
- step(action) → returns (observation, reward, done, info)
- state() → returns full internal state
- render() → optional visual output

This environment simulates cloud infrastructure auto-scaling where an agent
must balance cost efficiency against SLA compliance (no dropped requests).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.models import (
    ACTION_INDEX_MAP,
    ACTION_MAP,
    Action,
    ActionType,
    EnvironmentState,
    EpisodeInfo,
    Observation,
    Reward,
)
from env.server_model import ServerConfig, ServerFleet
from env.traffic_generator import (
    TrafficConfig,
    TrafficGenerator,
    chaos_traffic_config,
    spike_traffic_config,
    steady_traffic_config,
)
from utils.reward import RewardConfig, compute_reward


class CloudCostOptimizerEnv(gym.Env):
    """
    Cloud Cost Optimizer — Gymnasium-compatible RL Environment.

    An agent manages a fleet of cloud servers, making real-time scaling
    decisions (scale up, scale down, or hold) to balance:
    - Performance: minimize dropped requests and latency
    - Cost: minimize operational expenditure

    The environment models realistic traffic patterns, server warm-up delays,
    and exponential latency near saturation.

    OpenEnv Interface:
    - reset() → Observation
    - step(Action) → (Observation, Reward, bool, dict)
    - state() → EnvironmentState
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        traffic_config: Optional[TrafficConfig] = None,
        server_config: Optional[ServerConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 360,
        initial_servers: int = 2,
        task_name: str = "default",
        render_mode: Optional[str] = None,
        traffic_generator: Optional[TrafficGenerator] = None,
    ):
        """
        Initialize the Cloud Cost Optimizer environment.

        Args:
            traffic_config: Traffic generation settings (defaults to steady)
            server_config: Server fleet configuration
            reward_config: Reward function weights
            max_steps: Maximum steps per episode
            initial_servers: Starting number of servers
            task_name: Name of the current task
            render_mode: "human" or "ansi"
        """
        super().__init__()

        self.traffic_config = traffic_config or steady_traffic_config()
        self.server_config = server_config or ServerConfig()
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps
        self.initial_servers = initial_servers
        self.task_name = task_name
        self.render_mode = render_mode

        # Internal components
        from env.traffic_generator import TrafficGenerator

        self._traffic_gen = traffic_generator or TrafficGenerator(self.traffic_config)
        self._fleet = ServerFleet(self.server_config)

        # Episode state
        self._timestep = 0
        self._total_cost = 0.0
        self._total_served = 0.0
        self._total_dropped = 0.0
        self._total_requests = 0.0
        self._total_reward = 0.0
        self._done = False
        self._previous_traffic = 0.0

        # History for analytics
        self._traffic_history: list[float] = []
        self._server_history: list[int] = []
        self._latency_history: list[float] = []
        self._cpu_history: list[float] = []
        self._reward_history: list[float] = []
        self._dropped_history: list[float] = []
        self._served_history: list[float] = []
        self._cost_history: list[float] = []

        # Pre-generate traffic for determinism
        self._traffic_schedule: list[float] = []

        # Gymnasium spaces
        # Observation: 11 float values
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -np.inf, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 50, 50, 1, 10000, np.inf, np.inf, np.inf, np.inf, 1], dtype=np.float32),
            dtype=np.float32,
        )
        # Action: 5 discrete actions
        self.action_space = spaces.Discrete(5)

    # ─────────────────────────────────────────────────────────────────────
    # OpenEnv Interface
    # ─────────────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to a clean initial state.

        Returns:
            Tuple of (observation_array, info_dict)
        """
        super().reset(seed=seed)

        # Reset traffic generator
        if seed is not None:
            self._traffic_gen.reset(seed=seed)
        else:
            self._traffic_gen.reset(seed=self.traffic_config.seed)

        # Pre-generate deterministic traffic schedule
        self._traffic_schedule = self._traffic_gen.generate_episode(self.max_steps)

        # Reset fleet
        self._fleet.reset(initial_servers=self.initial_servers)

        # Reset episode state
        self._timestep = 0
        self._total_cost = 0.0
        self._total_served = 0.0
        self._total_dropped = 0.0
        self._total_requests = 0.0
        self._total_reward = 0.0
        self._done = False
        self._previous_traffic = self._traffic_schedule[0] if self._traffic_schedule else 0.0

        # Clear history
        self._traffic_history = []
        self._server_history = []
        self._latency_history = []
        self._cpu_history = []
        self._reward_history = []
        self._dropped_history = []
        self._served_history = []
        self._cost_history = []

        obs = self._get_observation()
        info = self._get_info()

        return np.array(obs.to_flat_array(), dtype=np.float32), info

    def step(self, action: int | Action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.

        Args:
            action: Integer action index (0-4) or Action pydantic model

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # Parse action
        if isinstance(action, (int, np.integer)):
            action_type = ACTION_INDEX_MAP[int(action)]
            action_obj = Action(action_type=action_type)
        elif isinstance(action, Action):
            action_obj = action
        else:
            raise ValueError(f"Invalid action type: {type(action)}")

        # 1) Apply scaling decision
        delta = action_obj.delta_servers
        self._fleet.scale(delta)

        # 2) Advance warm-up queue
        self._fleet.tick()

        # 3) Get traffic for this step
        traffic = self._traffic_schedule[self._timestep] if self._timestep < len(self._traffic_schedule) else 0.0

        # 4) Process requests through the fleet
        served, dropped, cpu_load, latency = self._fleet.process_requests(traffic)

        # 5) Calculate cost
        step_cost = self._fleet.get_cost()

        # 6) Compute reward
        reward_obj = compute_reward(
            served=served,
            dropped=dropped,
            cost=step_cost,
            latency_ms=latency,
            cpu_load=cpu_load,
            config=self.reward_config,
        )

        # 7) Update cumulative state
        self._total_cost += step_cost
        self._total_served += served
        self._total_dropped += dropped
        self._total_requests += traffic
        self._total_reward += reward_obj.total

        # 8) Record history
        self._traffic_history.append(traffic)
        self._server_history.append(self._fleet.active_servers)
        self._latency_history.append(latency)
        self._cpu_history.append(cpu_load)
        self._reward_history.append(reward_obj.total)
        self._dropped_history.append(dropped)
        self._served_history.append(served)
        self._cost_history.append(step_cost)

        # Store previous traffic for trend calculation
        self._previous_traffic = traffic

        # 9) Advance timestep
        self._timestep += 1

        # 10) Check termination
        terminated = self._timestep >= self.max_steps
        truncated = False
        self._done = terminated

        # Build observation
        obs = self._get_observation()
        info = self._get_info()
        info["reward_breakdown"] = reward_obj.model_dump()

        return (
            np.array(obs.to_flat_array(), dtype=np.float32),
            reward_obj.total,
            terminated,
            truncated,
            info,
        )

    def state(self) -> EnvironmentState:
        """Return the full internal state for serialization/debugging."""
        return EnvironmentState(
            timestep=self._timestep,
            active_servers=self._fleet.active_servers,
            warming_servers=list(self._fleet._warming_queue),
            total_cost=self._total_cost,
            total_served=self._total_served,
            total_dropped=self._total_dropped,
            total_requests=self._total_requests,
            latency_history=list(self._latency_history),
            cpu_history=list(self._cpu_history),
            traffic_history=list(self._traffic_history),
            server_history=list(self._server_history),
            reward_history=list(self._reward_history),
            episode_done=self._done,
        )

    # ─────────────────────────────────────────────────────────────────────
    # OpenEnv: Pydantic-based interface (for LLM agents)
    # ─────────────────────────────────────────────────────────────────────

    def reset_openenv(self, seed: Optional[int] = None) -> Observation:
        """Reset and return typed Observation (OpenEnv interface)."""
        self.reset(seed=seed)
        return self._get_observation()

    def step_openenv(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Step with typed Action and return typed Observation/Reward.
        OpenEnv-compatible interface for LLM agents.
        """
        obs_arr, reward_float, terminated, truncated, info = self.step(action)
        obs = self._get_observation()
        reward_obj = Reward(**info["reward_breakdown"])
        done = terminated or truncated
        return obs, reward_obj, done, info

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> Observation:
        """Build the current observation."""
        # Current traffic (use scheduled or last known)
        if self._timestep < len(self._traffic_schedule):
            current_traffic = self._traffic_schedule[self._timestep]
        elif self._traffic_history:
            current_traffic = self._traffic_history[-1]
        else:
            current_traffic = 0.0

        # Traffic trend
        if len(self._traffic_history) >= 2:
            trend = self._traffic_history[-1] - self._traffic_history[-2]
        elif len(self._traffic_history) == 1:
            trend = 0.0
        else:
            trend = 0.0

        # Time of day (normalized)
        period = self.traffic_config.period if self.traffic_config.period > 0 else 288
        time_of_day = (self._timestep % period) / period

        # Latest step metrics
        latency = self._latency_history[-1] if self._latency_history else 10.0
        cpu_load = self._cpu_history[-1] if self._cpu_history else 0.0
        dropped = self._dropped_history[-1] if self._dropped_history else 0.0
        served = self._served_history[-1] if self._served_history else 0.0

        return Observation(
            timestep=self._timestep,
            incoming_requests=current_traffic,
            active_servers=self._fleet.active_servers,
            warming_up_servers=self._fleet.warming_up_count,
            cpu_load=cpu_load,
            latency_ms=latency,
            dropped_requests=dropped,
            served_requests=served,
            cost_so_far=self._total_cost,
            traffic_trend=trend,
            time_of_day=time_of_day,
        )

    def _get_info(self) -> Dict[str, Any]:
        """Build info dict with episode statistics."""
        return {
            "timestep": self._timestep,
            "total_cost": self._total_cost,
            "total_served": self._total_served,
            "total_dropped": self._total_dropped,
            "total_requests": self._total_requests,
            "total_reward": self._total_reward,
            "active_servers": self._fleet.active_servers,
            "warming_servers": self._fleet.warming_up_count,
            "task_name": self.task_name,
        }

    def get_episode_info(self) -> EpisodeInfo:
        """Get summary statistics for the episode."""
        return EpisodeInfo(
            total_steps=self._timestep,
            total_reward=self._total_reward,
            total_requests=self._total_requests,
            total_served=self._total_served,
            total_dropped=self._total_dropped,
            total_cost=self._total_cost,
            avg_latency=float(np.mean(self._latency_history)) if self._latency_history else 0.0,
            max_latency=float(np.max(self._latency_history)) if self._latency_history else 0.0,
            avg_cpu_load=float(np.mean(self._cpu_history)) if self._cpu_history else 0.0,
            sla_compliance_rate=self._compute_sla_compliance(),
            avg_active_servers=float(np.mean(self._server_history)) if self._server_history else 0.0,
            peak_servers=max(self._server_history) if self._server_history else 0,
        )

    def _compute_sla_compliance(self) -> float:
        """Compute the percentage of steps that met SLA."""
        if not self._latency_history:
            return 1.0
        threshold = self.server_config.sla_latency_threshold_ms
        met = sum(1 for lat in self._latency_history if lat <= threshold)
        return met / len(self._latency_history)

    def get_history(self) -> Dict[str, list]:
        """Get full history for plotting/analysis."""
        return {
            "traffic": list(self._traffic_history),
            "servers": list(self._server_history),
            "latency": list(self._latency_history),
            "cpu_load": list(self._cpu_history),
            "reward": list(self._reward_history),
            "dropped": list(self._dropped_history),
            "served": list(self._served_history),
            "cost": list(self._cost_history),
        }

    def render(self) -> Optional[str]:
        """Render current environment state."""
        if self.render_mode == "ansi":
            obs = self._get_observation()
            return obs.to_prompt()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions for the 3 tasks
# ─────────────────────────────────────────────────────────────────────────────


def make_steady_env(seed: int = 42, max_steps: int = 360) -> CloudCostOptimizerEnv:
    """Create environment for Task 1: Steady Traffic."""
    return CloudCostOptimizerEnv(
        traffic_config=steady_traffic_config(seed=seed),
        max_steps=max_steps,
        initial_servers=2,
        task_name="steady_traffic",
    )


def make_spike_env(seed: int = 42, max_steps: int = 360) -> CloudCostOptimizerEnv:
    """Create environment for Task 2: Spike Traffic."""
    return CloudCostOptimizerEnv(
        traffic_config=spike_traffic_config(seed=seed),
        max_steps=max_steps,
        initial_servers=3,
        task_name="spike_traffic",
    )


def make_chaos_env(seed: int = 42, max_steps: int = 400) -> CloudCostOptimizerEnv:
    """Create environment for Task 3: Chaos Traffic."""
    return CloudCostOptimizerEnv(
        traffic_config=chaos_traffic_config(seed=seed),
        max_steps=max_steps,
        initial_servers=3,
        task_name="chaos_traffic",
    )
