import json
import os
import time

import pytest

from utils.ingress import TRAFFIC_DATA_FILE, TrafficState
from utils.reward import compute_reward


def test_traffic_state_record_hit(tmp_path):
    # Change CWD for the test to avoid junking the project root
    os.chdir(tmp_path)

    state = TrafficState()

    # 1. Record hit in current window
    state.record_hit()
    assert state.current_window_hits == 1

    # 2. Force window rollover
    state.window_start_time = time.time() - 20  # 20 seconds ago
    state.record_hit()

    assert len(state.history) == 1
    assert state.current_window_hits == 1
    assert os.path.exists(TRAFFIC_DATA_FILE)


def test_traffic_state_record_batch(tmp_path):
    os.chdir(tmp_path)
    state = TrafficState()

    state.record_batch(rps=50.5, timestamp=123456789.0)
    assert len(state.history) == 1
    assert state.history[0]["rps"] == 50.5
    assert state.history[0]["timestamp"] == 123456789.0


def test_reward_calculation():
    # Test reward function logic
    # (High served, low cost, no drops)
    reward = compute_reward(served=1000, dropped=0, cost=0.1, latency_ms=10.0, cpu_load=0.7)  # Low cost
    assert reward.served_reward > 0
    assert reward.total > 0
    assert reward.drop_penalty == 0

    # (Many drops -> negative reward)
    reward_bad = compute_reward(served=1000, dropped=500, cost=2, latency_ms=500.0, cpu_load=0.1)
    assert reward_bad.total < reward.total
    assert reward_bad.drop_penalty < 0
