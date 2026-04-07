import os

import pytest

from agent.evaluate import evaluate_baseline_agents


def test_evaluate_baseline_agents():
    # Evaluate only on "steady" task to keep it fast
    results = evaluate_baseline_agents(tasks=["steady"], seed=42, verbose=False)

    assert "steady" in results
    assert len(results["steady"]) > 0
    # Check that at least one agent (e.g., Static(3)) is in results
    assert any("Static(3)" in name for name in results["steady"])

    # Check structure of results
    first_agent = list(results["steady"].keys())[0]
    data = results["steady"][first_agent]
    assert "total_score" in data
    assert "sla_score" in data
    assert "cost_score" in data
    assert "latency_score" in data
    assert "stability_score" in data
