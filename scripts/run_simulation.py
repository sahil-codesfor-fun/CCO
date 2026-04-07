"""
Simulation Runner Script — CLI for running the Cloud Cost Optimizer.

Allows running specific tasks with specific agents and exporting results.
Used for bulk data generation and manual auditing of environment behavior.
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.baselines import PredictiveHeuristicAgent, StaticAgent, ThresholdAgent, run_baseline
from env.environment import make_chaos_env, make_spike_env, make_steady_env
from tasks.graders import grade_task

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TASK_MAP = {"steady": make_steady_env, "spike": make_spike_env, "chaos": make_chaos_env}

AGENT_MAP = {
    "threshold": lambda: ThresholdAgent(),
    "predictive": lambda: PredictiveHeuristicAgent(),
    "static3": lambda: StaticAgent(target_servers=3),
    "static5": lambda: StaticAgent(target_servers=5),
}


def run_sim(task_name: str, agent_name: str, seed: int, output: str):
    logger.info(f"🚀 Starting simulation: Task={task_name}, Agent={agent_name}, Seed={seed}")

    if task_name not in TASK_MAP:
        logger.error(f"Invalid task: {task_name}. Choose from: {list(TASK_MAP.keys())}")
        return

    if agent_name not in AGENT_MAP:
        logger.error(f"Invalid agent: {agent_name}. Choose from: {list(AGENT_MAP.keys())}")
        return

    # Create environment and agent
    env_func = TASK_MAP[task_name]
    env = env_func(seed=seed)
    agent = AGENT_MAP[agent_name]()

    # Run episode
    result = run_baseline(agent, env, seed=seed, verbose=True)

    # Grading (optional, depends on task config existence)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(project_root, "tasks", f"task_{task_name}.yaml")

    if os.path.exists(yaml_path):
        grade = grade_task(yaml_path, result["episode_info"], result["history"])
        result["grade"] = grade
        logger.info(f"🏆 Score: {grade['total_score']:.4f}")
    else:
        logger.warning(f"Grading config not found at {yaml_path}. Skipping score calculation.")

    # Save to file
    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"💾 Results saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Cost Optimizer - Simulation CLI")
    parser.add_argument("--task", type=str, default="steady", choices=["steady", "spike", "chaos"])
    parser.add_argument(
        "--agent", type=str, default="threshold", choices=["threshold", "predictive", "static3", "static5"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="sim_result.json")

    args = parser.parse_args()
    run_sim(args.task, args.agent, args.seed, args.output)
