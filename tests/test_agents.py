import numpy as np
import pytest

from agent.baselines import PredictiveHeuristicAgent, StaticAgent, ThresholdAgent, run_baseline
from env.environment import make_steady_env
from env.models import Action, ActionType, Observation


def test_static_agent():
    agent = StaticAgent(target_servers=5)
    assert agent.name() == "Static(5)"

    # Common fields for all observations
    base_obs = {
        "incoming_requests": 100.0,
        "warming_up_servers": 0,
        "latency_ms": 10.0,
        "dropped_requests": 0.0,
        "served_requests": 100.0,
        "cost_so_far": 50.0,
        "traffic_trend": 0.0,
        "time_of_day": 0.5,
    }

    # Below target
    obs = Observation(active_servers=3, cpu_load=0.5, timestep=1, **base_obs)
    action = agent.act(obs)
    assert action.action_type == ActionType.SCALE_UP_1

    # Above target
    obs = Observation(active_servers=7, cpu_load=0.5, timestep=1, **base_obs)
    action = agent.act(obs)
    assert action.action_type == ActionType.SCALE_DOWN_1

    # At target
    obs = Observation(active_servers=5, cpu_load=0.5, timestep=1, **base_obs)
    action = agent.act(obs)
    assert action.action_type == ActionType.NO_OP


def test_threshold_agent():
    agent = ThresholdAgent(high_threshold=0.8, low_threshold=0.2)
    assert "Threshold(80%/20%)" in agent.name()

    base_obs = {
        "incoming_requests": 100.0,
        "warming_up_servers": 0,
        "latency_ms": 10.0,
        "dropped_requests": 0.0,
        "served_requests": 100.0,
        "cost_so_far": 50.0,
        "traffic_trend": 0.0,
        "time_of_day": 0.5,
    }

    # High CPU -> Scale Up
    obs = Observation(active_servers=5, cpu_load=0.9, timestep=10, **base_obs)
    action = agent.act(obs)
    assert action.action_type == ActionType.SCALE_UP_1

    # Low CPU -> Scale Down
    obs = Observation(active_servers=5, cpu_load=0.1, timestep=20, **base_obs)
    action = agent.act(obs)
    assert action.action_type == ActionType.SCALE_DOWN_1

    # Cooldown period
    obs = Observation(active_servers=5, cpu_load=0.9, timestep=21, **base_obs)
    action = agent.act(obs)
    assert action.action_type == ActionType.NO_OP
    assert "Cooldown" in action.reasoning


def test_predictive_agent():
    agent = PredictiveHeuristicAgent()
    assert agent.name() == "PredictiveHeuristic"

    base_obs = {
        "latency_ms": 10.0,
        "dropped_requests": 0.0,
        "served_requests": 100.0,
        "cost_so_far": 50.0,
        "time_of_day": 0.5,
    }

    # Rising traffic + high CPU
    obs = Observation(
        active_servers=5,
        cpu_load=0.7,
        traffic_trend=10.0,
        incoming_requests=300.0,
        warming_up_servers=0,
        timestep=1,
        **base_obs,
    )
    action = agent.act(obs)
    assert action.action_type == ActionType.SCALE_UP_3

    # Critical load
    obs = Observation(
        active_servers=5,
        cpu_load=0.9,
        timestep=1,
        traffic_trend=0.0,
        incoming_requests=100.0,
        warming_up_servers=0,
        **base_obs,
    )
    action = agent.act(obs)
    assert action.action_type == ActionType.SCALE_UP_3


def test_run_baseline():
    env = make_steady_env(max_steps=10)
    agent = StaticAgent(target_servers=3)
    result = run_baseline(agent, env, seed=42)

    assert "agent_name" in result
    assert "total_reward" in result
    assert "episode_info" in result
    assert "history" in result
    assert result["steps"] == 10
