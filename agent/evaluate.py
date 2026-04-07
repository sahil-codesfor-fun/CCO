"""
Evaluation Script — Runs agents against all tasks and produces graded scores.

Evaluates:
1. Baseline agents (Static, Threshold, Predictive)
2. Trained RL agents (DQN, PPO)
3. Produces comparison table with 0.0–1.0 scores
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent.baselines import (
    PredictiveHeuristicAgent,
    StaticAgent,
    ThresholdAgent,
    run_baseline,
)
from env.environment import make_chaos_env, make_spike_env, make_steady_env
from tasks.graders import grade_task

try:
    from stable_baselines3 import DQN, PPO

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


TASK_CONFIGS = {
    "steady": {
        "make_env": make_steady_env,
        "yaml_path": os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks", "task_steady.yaml"
        ),
    },
    "spike": {
        "make_env": make_spike_env,
        "yaml_path": os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks", "task_spike.yaml"
        ),
    },
    "chaos": {
        "make_env": make_chaos_env,
        "yaml_path": os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks", "task_chaos.yaml"
        ),
    },
}


def evaluate_baseline_agents(
    tasks: List[str] = ["steady", "spike", "chaos"],
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all baseline agents on all tasks."""
    agents = [
        StaticAgent(target_servers=3),
        StaticAgent(target_servers=5),
        StaticAgent(target_servers=10),
        ThresholdAgent(high_threshold=0.75, low_threshold=0.30),
        ThresholdAgent(high_threshold=0.80, low_threshold=0.20, cooldown_steps=5),
        PredictiveHeuristicAgent(),
    ]

    results = {}

    for task_name in tasks:
        task_cfg = TASK_CONFIGS[task_name]
        env = task_cfg["make_env"](seed=seed)
        yaml_path = task_cfg["yaml_path"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task_name.upper()}")
            print(f"{'='*60}")

        task_results = {}

        for agent in agents:
            env_fresh = task_cfg["make_env"](seed=seed)
            result = run_baseline(agent, env_fresh, seed=seed, verbose=False)
            grade = grade_task(yaml_path, result["episode_info"], result["history"])

            task_results[agent.name()] = {
                "total_score": grade["total_score"],
                "sla_score": grade["sla_score"],
                "cost_score": grade["cost_score"],
                "latency_score": grade["latency_score"],
                "stability_score": grade["stability_score"],
                "total_reward": result["total_reward"],
                "details": grade["details"],
            }

            if verbose:
                ep = result["episode_info"]
                print(f"\n  {agent.name():30s}  Score: {grade['total_score']:.4f}")
                print(
                    f"    SLA: {grade['sla_score']:.3f}  Cost: {grade['cost_score']:.3f}  "
                    f"Latency: {grade['latency_score']:.3f}  Stability: {grade['stability_score']:.3f}"
                )
                print(
                    f"    Served: {ep.total_served:.0f}  Dropped: {ep.total_dropped:.0f}  "
                    f"Cost: ${ep.total_cost:.2f}  Avg Latency: {ep.avg_latency:.1f}ms"
                )

        results[task_name] = task_results

    return results


def evaluate_trained_agent(
    model_path: str,
    algo: str = "dqn",
    tasks: List[str] = ["steady", "spike", "chaos"],
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a trained RL agent on all tasks."""
    if not HAS_SB3:
        raise ImportError("stable-baselines3 required for trained agent evaluation")

    if algo == "dqn":
        model = DQN.load(model_path)
    elif algo == "ppo":
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    results = {}

    for task_name in tasks:
        task_cfg = TASK_CONFIGS[task_name]
        env = task_cfg["make_env"](seed=seed)
        yaml_path = task_cfg["yaml_path"]

        obs, info = env.reset(seed=seed)
        total_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            if terminated or truncated:
                break

        episode_info = env.get_episode_info()
        history = env.get_history()
        grade = grade_task(yaml_path, episode_info, history)

        results[task_name] = {
            "total_score": grade["total_score"],
            "total_reward": total_reward,
            "details": grade["details"],
            "sla_score": grade["sla_score"],
            "cost_score": grade["cost_score"],
            "latency_score": grade["latency_score"],
            "stability_score": grade["stability_score"],
        }

        if verbose:
            print(f"\n  {task_name.upper():15s}  Score: {grade['total_score']:.4f}")
            print(
                f"    SLA: {grade['sla_score']:.3f}  Cost: {grade['cost_score']:.3f}  "
                f"Latency: {grade['latency_score']:.3f}  Stability: {grade['stability_score']:.3f}"
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agents on Cloud Cost Optimizer")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (optional)")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="evaluation_results.json")

    args = parser.parse_args()

    print("=" * 60)
    print("Cloud Cost Optimizer — Agent Evaluation")
    print("=" * 60)

    # Always run baseline evaluation
    print("\n📊 BASELINE AGENTS")
    baseline_results = evaluate_baseline_agents(seed=args.seed)

    all_results = {"baselines": baseline_results}

    # Optionally evaluate trained model
    if args.model:
        print(f"\n🤖 TRAINED {args.algo.upper()} AGENT")
        trained_results = evaluate_trained_agent(args.model, algo=args.algo, seed=args.seed)
        all_results["trained"] = trained_results

    # Save results
    output_path = args.output
    with open(output_path, "w") as f:
        # Convert EpisodeInfo objects to dicts for JSON serialization
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {output_path}")
