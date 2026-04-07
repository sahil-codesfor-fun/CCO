"""
Baseline Inference Script — Uses the OpenAI API to run an LLM agent against the environment.

Reads API credentials from environment variables (OPENAI_API_KEY).
Produces a reproducible baseline score on all 3 tasks.

This demonstrates the OpenEnv interface working with LLM-based agents.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

import numpy as np

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Attempt to load environment variables from .env

# Attempt to load environment variables from .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import make_chaos_env, make_spike_env, make_steady_env
from env.models import Action, ActionType, Observation
from tasks.graders import grade_task

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


SYSTEM_PROMPT = """You are an AI cloud infrastructure auto-scaler managing a fleet of servers.

Your job is to make real-time scaling decisions to balance:
1. Performance: Minimize dropped requests and latency
2. Cost: Minimize server operational costs

At each step, you receive telemetry about the server fleet and must choose ONE action:
- "scale_up_3": Add 3 servers (aggressive scaling for spike events)
- "scale_up_1": Add 1 server (cautious scaling)
- "no_op": Keep current fleet size (steady state)
- "scale_down_1": Remove 1 server (cost optimization)
- "scale_down_3": Remove 3 servers (aggressive cost optimization)

Key constraints:
- New servers take 3 steps to warm up before they can serve traffic
- Latency grows EXPONENTIALLY as CPU load approaches 100%
- Dropped requests incur HEAVY penalties
- Each server costs money per time step even when idle
- You must maintain at least 1 active server

Strategy tips:
- If CPU > 75%, consider scaling up
- If CPU < 30% and you have many servers, consider scaling down
- Watch the traffic trend: positive = traffic increasing, negative = decreasing
- Be PROACTIVE: scale up BEFORE spikes hit (server warm-up takes time)

Respond with ONLY a JSON object: {"action": "<action_type>", "reasoning": "<brief reasoning>"}
"""


def run_llm_episode(
    env,
    task_name: str,
    seed: int = 42,
    model: str = "meta/llama-3.1-70b-instruct",
    max_retries: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single episode using an LLM agent.

    Args:
        env: The CloudCostOptimizer environment
        task_name: Name of the task
        seed: Random seed
        model: OpenAI model name
        max_retries: Max retries for API failures
        verbose: Print step-by-step output

    Returns:
        Dict with results and grading
    """
    if not HAS_OPENAI:
        raise ImportError(
            "openai package required. Install with: pip install openai\n"
            "Then set your API key: export OPENAI_API_KEY=nvapi-YOUR_KEY\n"
            "And base URL:         export OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1"
        )

    # Check for API credentials (prioritize hackathon env vars)
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API_KEY / OPENAI_API_KEY environment variable not set.\n"
            "Submit with the provided API_KEY or get an NVIDIA NIM key."
        )

    base_url = os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        logger.info(f"Using API endpoint: {base_url}")
    else:
        logger.info("Using default OpenAI endpoint (may fail if behind proxy)")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Reset environment
    obs = env.reset_openenv(seed=seed)
    total_reward = 0.0
    steps = 0

    if verbose:
        logger.info(f"{'='*60}")
        logger.info(f"Running LLM Agent ({model}) on task: {task_name}")
        logger.info(f"{'='*60}")

    while True:
        # Build prompt with current observation
        obs_text = obs.to_prompt()

        # Call LLM
        action = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": obs_text + "\nChoose your action:"},
                    ],
                    temperature=0.0,
                    max_tokens=150,
                    response_format={"type": "json_object"},
                )

                result_text = response.choices[0].message.content.strip()
                result_json = json.loads(result_text)
                action_str = result_json.get("action", "no_op")
                reasoning = result_json.get("reasoning", "")

                # Parse action
                try:
                    action_type = ActionType(action_str)
                except ValueError:
                    action_type = ActionType.NO_OP
                    reasoning = f"Invalid action '{action_str}', defaulting to NO_OP"

                action = Action(action_type=action_type, reasoning=reasoning)
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    if verbose:
                        logger.warning(f"  Retry {attempt + 1}: {e}")
                    continue
                else:
                    # Fallback to NO_OP
                    action = Action(action_type=ActionType.NO_OP, reasoning=f"API error: {e}")

        # Step environment
        obs, reward, done, info = env.step_openenv(action)
        total_reward += reward.total
        steps += 1

        if verbose and steps % 20 == 0:
            logger.info(
                f"  Step {steps:4d}: servers={obs.active_servers}, "
                f"cpu={obs.cpu_load:.1%}, dropped={obs.dropped_requests:.0f}, "
                f"action={action.action_type.value} | {action.reasoning}"
            )

        if done:
            break

    # Grade the episode
    episode_info = env.get_episode_info()
    history = env.get_history()

    return {
        "task_name": task_name,
        "model": model,
        "total_reward": total_reward,
        "steps": steps,
        "episode_info": episode_info,
        "history": history,
    }


def run_all_tasks(
    model: str = "meta/llama-3.1-70b-instruct",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run LLM agent on all 3 tasks and return graded results."""

    project_root = os.path.dirname(os.path.abspath(__file__))

    tasks = {
        "steady": {
            "make_env": make_steady_env,
            "yaml": os.path.join(project_root, "tasks", "task_steady.yaml"),
        },
        "spike": {
            "make_env": make_spike_env,
            "yaml": os.path.join(project_root, "tasks", "task_spike.yaml"),
        },
        "chaos": {
            "make_env": make_chaos_env,
            "yaml": os.path.join(project_root, "tasks", "task_chaos.yaml"),
        },
    }

    results = {}

    for task_name, task_cfg in tasks.items():
        env = task_cfg["make_env"](seed=seed)
        result = run_llm_episode(env, task_name, seed=seed, model=model, verbose=verbose)

        grade = grade_task(task_cfg["yaml"], result["episode_info"], result["history"])

        results[task_name] = {
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
            logger.info(f"{'─'*50}")
            logger.info(f"  Task: {task_name.upper()}")
            logger.info(f"  Score:     {grade['total_score']:.4f}")
            logger.info(f"  SLA:       {grade['sla_score']:.3f}")
            logger.info(f"  Cost:      {grade['cost_score']:.3f}")
            logger.info(f"  Latency:   {grade['latency_score']:.3f}")
            logger.info(f"  Stability: {grade['stability_score']:.3f}")
            logger.info(f"  Served:    {ep.total_served:.0f}")
            logger.info(f"  Dropped:   {ep.total_dropped:.0f}")
            logger.info(f"  Cost:      ${ep.total_cost:.2f}")

    # Overall score
    avg_score = np.mean([r["total_score"] for r in results.values()])
    if verbose:
        logger.info(f"{'='*60}")
        logger.info(f"  OVERALL SCORE: {avg_score:.4f}")
        logger.info(f"{'='*60}")

    results["overall"] = {"average_score": round(float(avg_score), 4)}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM baseline agent")
    parser.add_argument("--model", type=str, default="meta/llama-3.1-70b-instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="llm_baseline_results.json")

    args = parser.parse_args()

    results = run_all_tasks(model=args.model, seed=args.seed)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"💾 Results saved to: {args.output}")
