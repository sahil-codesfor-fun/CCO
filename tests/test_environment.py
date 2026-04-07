"""
Test Suite for Cloud Cost Optimizer — OpenEnv Validation.

Tests cover:
1. Environment interface compliance (reset, step, state)
2. Observation/Action/Reward Pydantic model validation
3. Traffic generator correctness
4. Server fleet behavior
5. Grader determinism and range
6. End-to-end episode execution
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import (
    CloudCostOptimizerEnv,
    make_chaos_env,
    make_spike_env,
    make_steady_env,
)
from env.models import (
    ACTION_INDEX_MAP,
    Action,
    ActionType,
    EnvironmentState,
    EpisodeInfo,
    Observation,
    Reward,
)
from env.server_model import ServerConfig, ServerFleet
from env.traffic_generator import (
    SpikeEvent,
    TrafficConfig,
    TrafficGenerator,
    chaos_traffic_config,
    spike_traffic_config,
    steady_traffic_config,
)
from tasks.graders import GradingConfig, grade_episode, load_grading_config
from utils.reward import RewardConfig, compute_reward

# ─────────────────────────────────────────────────────────────────────────────
# Test Environment Interface
# ─────────────────────────────────────────────────────────────────────────────


class TestEnvironmentInterface:
    """Test OpenEnv interface compliance."""

    def test_reset_returns_clean_state(self):
        env = make_steady_env(seed=42)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (11,)
        assert info["timestep"] == 0
        assert info["total_cost"] == 0.0
        assert info["total_dropped"] == 0.0

    def test_reset_is_deterministic(self):
        env1 = make_steady_env(seed=42)
        env2 = make_steady_env(seed=42)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_step_returns_correct_tuple(self):
        env = make_steady_env(seed=42)
        env.reset(seed=42)
        result = env.step(2)  # NO_OP
        assert len(result) == 5  # obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_with_action_object(self):
        env = make_steady_env(seed=42)
        env.reset(seed=42)
        action = Action(action_type=ActionType.SCALE_UP_1)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)

    def test_episode_terminates(self):
        env = make_steady_env(seed=42, max_steps=10)
        env.reset(seed=42)
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(2)
        assert terminated is True

    def test_state_returns_environment_state(self):
        env = make_steady_env(seed=42)
        env.reset(seed=42)
        env.step(2)
        state = env.state()
        assert isinstance(state, EnvironmentState)
        assert state.timestep == 1
        assert state.active_servers >= 1

    def test_episode_info(self):
        env = make_steady_env(seed=42, max_steps=20)
        env.reset(seed=42)
        for _ in range(20):
            env.step(2)
        info = env.get_episode_info()
        assert isinstance(info, EpisodeInfo)
        assert info.total_steps == 20
        assert info.total_requests > 0

    def test_openenv_interface(self):
        env = make_steady_env(seed=42)
        obs = env.reset_openenv(seed=42)
        assert isinstance(obs, Observation)
        assert obs.timestep == 0

        action = Action(action_type=ActionType.NO_OP)
        obs, reward, done, info = env.step_openenv(action)
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)


# ─────────────────────────────────────────────────────────────────────────────
# Test Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────


class TestPydanticModels:
    """Test typed model validation."""

    def test_observation_creation(self):
        obs = Observation(
            timestep=0,
            incoming_requests=100.0,
            active_servers=3,
            warming_up_servers=0,
            cpu_load=0.5,
            latency_ms=20.0,
            dropped_requests=0.0,
            served_requests=100.0,
            cost_so_far=0.3,
            traffic_trend=5.0,
            time_of_day=0.5,
        )
        assert obs.incoming_requests == 100.0
        assert obs.cpu_load == 0.5

    def test_observation_to_flat_array(self):
        obs = Observation(
            timestep=5,
            incoming_requests=100.0,
            active_servers=3,
            warming_up_servers=1,
            cpu_load=0.5,
            latency_ms=20.0,
            dropped_requests=0.0,
            served_requests=100.0,
            cost_so_far=1.5,
            traffic_trend=2.0,
            time_of_day=0.3,
        )
        arr = obs.to_flat_array()
        assert len(arr) == 11
        assert arr[0] == 5.0

    def test_observation_to_prompt(self):
        obs = Observation(
            timestep=10,
            incoming_requests=150.0,
            active_servers=4,
            warming_up_servers=1,
            cpu_load=0.72,
            latency_ms=45.3,
            dropped_requests=5.0,
            served_requests=145.0,
            cost_so_far=4.2,
            traffic_trend=-3.0,
            time_of_day=0.75,
        )
        prompt = obs.to_prompt()
        assert "Step 10" in prompt
        assert "150.0" in prompt

    def test_action_types(self):
        for action_type in ActionType:
            action = Action(action_type=action_type)
            assert isinstance(action.delta_servers, int)

    def test_action_delta_values(self):
        assert Action(action_type=ActionType.SCALE_UP_3).delta_servers == 3
        assert Action(action_type=ActionType.SCALE_UP_1).delta_servers == 1
        assert Action(action_type=ActionType.NO_OP).delta_servers == 0
        assert Action(action_type=ActionType.SCALE_DOWN_1).delta_servers == -1
        assert Action(action_type=ActionType.SCALE_DOWN_3).delta_servers == -3

    def test_reward_breakdown(self):
        reward = Reward(
            total=0.5,
            served_reward=1.0,
            cost_penalty=-0.3,
            drop_penalty=-0.1,
            latency_penalty=-0.05,
            efficiency_bonus=0.05,
        )
        assert reward.total == 0.5
        prompt = reward.to_prompt()
        assert "Reward Breakdown" in prompt

    def test_observation_validation_rejects_invalid(self):
        with pytest.raises(Exception):
            Observation(
                timestep=-1,  # Invalid: must be >= 0
                incoming_requests=100.0,
                active_servers=3,
                warming_up_servers=0,
                cpu_load=0.5,
                latency_ms=20.0,
                dropped_requests=0.0,
                served_requests=100.0,
                cost_so_far=0.3,
                traffic_trend=5.0,
                time_of_day=0.5,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test Traffic Generator
# ─────────────────────────────────────────────────────────────────────────────


class TestTrafficGenerator:
    """Test traffic generation."""

    def test_steady_traffic_is_positive(self):
        gen = TrafficGenerator(steady_traffic_config(seed=42))
        for t in range(100):
            traffic = gen.generate(t)
            assert traffic >= 0

    def test_real_time_traffic_fallback(self, tmp_path):
        import json

        local_json = tmp_path / "production_traffic.json"
        data = [{"timestamp": 0, "rps": 123.0}, {"timestamp": 1, "rps": 456.0}]
        with open(local_json, "w") as f:
            json.dump(data, f)

        import os

        from env.real_traffic_generator import RealTimeTrafficGenerator

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            config = steady_traffic_config()
            # API will fail (not running), should fall back to JSON
            gen = RealTimeTrafficGenerator(config, api_url="http://localhost:1/fail")
            assert len(gen.history_data) == 2
            assert gen.generate(0) == 123.0
            assert gen.generate(1) == 456.0
            # Wrap around
            assert gen.generate(2) > 0
        finally:
            os.chdir(old_cwd)

    def test_deterministic_with_seed(self):
        gen1 = TrafficGenerator(steady_traffic_config(seed=42))
        gen2 = TrafficGenerator(steady_traffic_config(seed=42))
        for t in range(50):
            assert gen1.generate(t) == gen2.generate(t)

    def test_spike_traffic_has_higher_peaks(self):
        steady = TrafficGenerator(steady_traffic_config(seed=42))
        spike = TrafficGenerator(spike_traffic_config(seed=42))
        steady_vals = [steady.generate(t) for t in range(300)]
        spike_vals = [spike.generate(t) for t in range(300)]
        assert max(spike_vals) > max(steady_vals)

    def test_chaos_traffic_is_highly_variable(self):
        chaos = TrafficGenerator(chaos_traffic_config(seed=42))
        vals = [chaos.generate(t) for t in range(300)]
        assert np.std(vals) > 30  # High variance

    def test_episode_generation(self):
        gen = TrafficGenerator(steady_traffic_config(seed=42))
        episode = gen.generate_episode(100)
        assert len(episode) == 100
        assert all(v >= 0 for v in episode)

    def test_spike_event(self):
        spike = SpikeEvent(start_step=10, duration=20, magnitude=100.0)
        assert spike.is_active(15)
        assert not spike.is_active(5)
        assert not spike.is_active(35)
        assert spike.get_contribution(15) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test Server Fleet
# ─────────────────────────────────────────────────────────────────────────────


class TestServerFleet:
    """Test server fleet simulation."""

    def test_initial_state(self):
        fleet = ServerFleet(ServerConfig())
        assert fleet.active_servers == 1
        assert fleet.warming_up_count == 0

    def test_scale_up_queues_warmup(self):
        fleet = ServerFleet(ServerConfig())
        fleet.scale(2)
        assert fleet.warming_up_count == 2
        assert fleet.active_servers == 1  # Not yet active

    def test_warmup_completes(self):
        config = ServerConfig(warmup_steps=2)
        fleet = ServerFleet(config)
        fleet.scale(1)
        fleet.tick()  # 1 step remaining
        assert fleet.active_servers == 1
        fleet.tick()  # 0 steps remaining → active
        assert fleet.active_servers == 2

    def test_scale_down_instant(self):
        fleet = ServerFleet(ServerConfig())
        fleet.active_servers = 5
        actual = fleet.scale(-2)
        assert actual == -2
        assert fleet.active_servers == 3

    def test_min_servers_enforced(self):
        fleet = ServerFleet(ServerConfig(min_servers=1))
        fleet.active_servers = 2
        fleet.scale(-5)  # Try to remove too many
        assert fleet.active_servers == 1

    def test_max_servers_enforced(self):
        config = ServerConfig(max_servers=5)
        fleet = ServerFleet(config)
        fleet.active_servers = 4
        fleet.scale(10)  # Try to add too many
        assert fleet.active_servers + fleet.warming_up_count <= 5

    def test_process_requests_no_drop(self):
        fleet = ServerFleet(ServerConfig(capacity_per_server=100))
        fleet.active_servers = 2
        served, dropped, cpu, latency = fleet.process_requests(150)
        assert served == 150
        assert dropped == 0

    def test_process_requests_with_drop(self):
        fleet = ServerFleet(ServerConfig(capacity_per_server=100))
        fleet.active_servers = 2
        served, dropped, cpu, latency = fleet.process_requests(250)
        assert served == 200
        assert dropped == 50

    def test_latency_increases_with_load(self):
        fleet = ServerFleet(ServerConfig(capacity_per_server=100))
        fleet.active_servers = 2
        _, _, _, lat_low = fleet.process_requests(50)  # 25% load
        _, _, _, lat_high = fleet.process_requests(180)  # 90% load
        assert lat_high > lat_low

    def test_cost_calculation(self):
        fleet = ServerFleet(ServerConfig(cost_per_server_per_step=0.1))
        fleet.active_servers = 5
        cost = fleet.get_cost()
        assert cost == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Test Reward Function
# ─────────────────────────────────────────────────────────────────────────────


class TestRewardFunction:
    """Test reward computation."""

    def test_reward_is_structured(self):
        reward = compute_reward(100, 0, 0.5, 20, 0.5)
        assert isinstance(reward, Reward)
        assert reward.total is not None

    def test_no_drops_positive_reward(self):
        reward = compute_reward(100, 0, 0.3, 10, 0.5)
        assert reward.drop_penalty == 0.0

    def test_drops_cause_heavy_penalty(self):
        reward_no_drop = compute_reward(100, 0, 0.5, 20, 0.5)
        reward_drop = compute_reward(100, 50, 0.5, 20, 0.5)
        assert reward_drop.total < reward_no_drop.total

    def test_high_latency_penalized(self):
        reward_low = compute_reward(100, 0, 0.5, 20, 0.5)
        reward_high = compute_reward(100, 0, 0.5, 500, 0.5)
        assert reward_high.total < reward_low.total

    def test_efficiency_bonus_in_sweet_spot(self):
        reward = compute_reward(100, 0, 0.5, 20, 0.6)  # In 0.4-0.75 range
        assert reward.efficiency_bonus > 0

    def test_no_efficiency_bonus_outside_range(self):
        reward = compute_reward(100, 0, 0.5, 20, 0.1)  # Below range
        assert reward.efficiency_bonus == 0.0

    def test_reward_provides_dense_signal(self):
        """Reward should vary across different conditions."""
        rewards = []
        for cpu in [0.1, 0.3, 0.5, 0.7, 0.9]:
            r = compute_reward(100, 0, 0.5, 20, cpu)
            rewards.append(r.total)
        assert len(set([round(r, 4) for r in rewards])) > 1  # Not all the same


# ─────────────────────────────────────────────────────────────────────────────
# Test Graders
# ─────────────────────────────────────────────────────────────────────────────


class TestGraders:
    """Test grading functions."""

    def _make_episode_info(self, **kwargs) -> EpisodeInfo:
        defaults = dict(
            total_steps=100,
            total_reward=50.0,
            total_requests=10000.0,
            total_served=9900.0,
            total_dropped=100.0,
            total_cost=50.0,
            avg_latency=30.0,
            max_latency=150.0,
            avg_cpu_load=0.6,
            sla_compliance_rate=0.95,
            avg_active_servers=5.0,
            peak_servers=8,
        )
        defaults.update(kwargs)
        return EpisodeInfo(**defaults)

    def test_grade_returns_0_to_1(self):
        info = self._make_episode_info()
        history = {"servers": list(range(100))}
        config = GradingConfig()
        result = grade_episode(info, history, config)
        assert 0.0 <= result["total_score"] <= 1.0

    def test_perfect_episode_scores_high(self):
        info = self._make_episode_info(
            total_served=10000.0,
            total_dropped=0.0,
            total_cost=40.0,
            avg_latency=15.0,
            max_latency=50.0,
            sla_compliance_rate=1.0,
        )
        history = {"servers": [5] * 100}  # Stable server count
        config = GradingConfig()
        result = grade_episode(info, history, config)
        assert result["total_score"] >= 0.7

    def test_terrible_episode_scores_low(self):
        info = self._make_episode_info(
            total_served=5000.0,
            total_dropped=5000.0,
            total_cost=200.0,
            avg_latency=500.0,
            max_latency=2000.0,
            sla_compliance_rate=0.1,
        )
        history = {"servers": [i % 10 + 1 for i in range(100)]}  # Wild oscillation
        config = GradingConfig()
        result = grade_episode(info, history, config)
        assert result["total_score"] <= 0.3

    def test_grading_is_deterministic(self):
        info = self._make_episode_info()
        history = {"servers": [5] * 100}
        config = GradingConfig()
        r1 = grade_episode(info, history, config)
        r2 = grade_episode(info, history, config)
        assert r1["total_score"] == r2["total_score"]

    def test_grade_has_component_scores(self):
        info = self._make_episode_info()
        history = {"servers": [5] * 100}
        config = GradingConfig()
        result = grade_episode(info, history, config)
        assert "sla_score" in result
        assert "cost_score" in result
        assert "latency_score" in result
        assert "stability_score" in result


# ─────────────────────────────────────────────────────────────────────────────
# Test End-to-End
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    """Test full episode execution."""

    def test_full_episode_steady(self):
        env = make_steady_env(seed=42, max_steps=50)
        obs, info = env.reset(seed=42)
        total_reward = 0
        for _ in range(50):
            obs, reward, done, _, info = env.step(2)  # NO_OP
            total_reward += reward
            if done:
                break
        assert done
        assert env.get_episode_info().total_steps == 50

    def test_full_episode_spike(self):
        env = make_spike_env(seed=42, max_steps=50)
        obs, info = env.reset(seed=42)
        for _ in range(50):
            obs, reward, done, _, info = env.step(1)  # SCALE_UP_1
            if done:
                break
        assert done

    def test_full_episode_chaos(self):
        env = make_chaos_env(seed=42, max_steps=50)
        obs, info = env.reset(seed=42)
        for _ in range(50):
            obs, reward, done, _, info = env.step(2)
            if done:
                break
        assert done

    def test_reproducibility(self):
        """Two runs with same seed should produce identical results."""
        env1 = make_steady_env(seed=42, max_steps=20)
        env2 = make_steady_env(seed=42, max_steps=20)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        for _ in range(20):
            o1, r1, d1, _, _ = env1.step(2)
            o2, r2, d2, _, _ = env2.step(2)
            assert r1 == r2
            np.testing.assert_array_almost_equal(o1, o2)

    def test_all_actions_valid(self):
        """All 5 action types should be accepted."""
        env = make_steady_env(seed=42, max_steps=10)
        env.reset(seed=42)
        for i in range(5):
            obs, reward, done, _, info = env.step(i)
            assert isinstance(reward, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
