"""
Agent Deployment Script — Runs a trained agent in production mode.

This connects a trained RL agent to the Ingress API, creating a live
auto-scaling loop:

1. Pull latest telemetry from /telemetry/latest
2. Run the agent's policy to decide scaling action
3. Post the decision to /decision
4. Infrastructure (K8s/Terraform) polls /decision and applies

Usage:
    # Deploy with a registered agent
    python agent/deploy.py --company "Acme Corp" --agent "BlackFriday Scaler"

    # Deploy with a model file directly
    python agent/deploy.py --model-path models/best/best_model.zip --algo dqn

    # Deploy to a remote ingress
    python agent/deploy.py --company "Acme" --agent "Scaler" --ingress-url http://prod:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from stable_baselines3 import DQN, PPO

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

import numpy as np

from agent.baselines import PredictiveHeuristicAgent, ThresholdAgent
from agent.registry import AgentRegistry
from env.models import Action, ActionType, Observation

# Setup logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("AgentDeployer")


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry → Observation Adapter
# ─────────────────────────────────────────────────────────────────────────────


def telemetry_to_observation(telemetry: Dict[str, Any], step: int = 0) -> Observation:
    """Convert raw telemetry payload to an Observation object."""
    cpu = telemetry.get("cpu_load", 0.0)
    rps = telemetry.get("request_count", 0)
    latency = telemetry.get("latency_ms", 10.0)
    servers = telemetry.get("active_servers", 1)

    return Observation(
        timestep=step,
        incoming_requests=float(rps),
        active_servers=servers,
        warming_up_servers=0,
        cpu_load=cpu,
        latency_ms=latency,
        dropped_requests=0.0,
        served_requests=float(rps),
        cost_so_far=0.0,
        traffic_trend=0.0,
        time_of_day=(step % 288) / 288.0,
    )


def telemetry_to_array(telemetry: Dict[str, Any], step: int = 0) -> np.ndarray:
    """Convert telemetry to the numpy observation array for SB3 models."""
    obs = telemetry_to_observation(telemetry, step)
    return np.array(
        [
            obs.timestep / 400.0,
            obs.incoming_requests / 500.0,
            obs.active_servers / 50.0,
            obs.warming_up_servers / 10.0,
            obs.cpu_load,
            min(obs.latency_ms / 1000.0, 1.0),
            obs.dropped_requests / 500.0,
            obs.served_requests / 500.0,
            obs.cost_so_far / 200.0,
            obs.traffic_trend / 100.0,
            obs.time_of_day,
        ],
        dtype=np.float32,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Action → Decision Adapter
# ─────────────────────────────────────────────────────────────────────────────

ACTION_MAP = {
    0: ("SCALE_UP_3", "+3 servers (emergency)"),
    1: ("SCALE_UP_1", "+1 server (proactive)"),
    2: ("NO_OP", "Hold current fleet"),
    3: ("SCALE_DOWN_1", "-1 server (cost save)"),
    4: ("SCALE_DOWN_3", "-3 servers (aggressive save)"),
}


def action_to_decision(action_idx: int, reasoning_extra: str = "") -> Dict[str, Any]:
    """Convert integer action to a decision payload."""
    action_name, default_reason = ACTION_MAP.get(action_idx, ("NO_OP", "Unknown"))
    return {
        "action": action_name,
        "reasoning": reasoning_extra or default_reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent Deployer
# ─────────────────────────────────────────────────────────────────────────────


class AgentDeployer:
    """
    Deploys a trained agent to the production Ingress API.

    Continuously loops:
    1. GET /telemetry/latest
    2. Run agent policy
    3. POST /decision
    """

    def __init__(
        self,
        ingress_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        poll_interval: int = 10,
    ):
        if not HAS_REQUESTS:
            raise ImportError("requests library required. Install: pip install requests")

        self.ingress_url = ingress_url.rstrip("/")
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.step_count = 0
        self.model = None
        self.heuristic_agent = None
        self.algo = None
        self.agent_name = "unnamed"
        self.company_name = "unknown"

    def load_rl_model(self, model_path: str, algo: str = "dqn"):
        """Load a trained SB3 model."""
        if not HAS_SB3:
            raise ImportError("stable-baselines3 required for RL models.")

        if algo == "dqn":
            self.model = DQN.load(model_path)
        elif algo == "ppo":
            self.model = PPO.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        self.algo = algo
        logger.info(f"✅ Loaded {algo.upper()} model from {model_path}")

    def load_heuristic_agent(self, agent_type: str = "threshold"):
        """Load a heuristic baseline agent."""
        if agent_type == "threshold":
            self.heuristic_agent = ThresholdAgent()
        elif agent_type == "predictive":
            self.heuristic_agent = PredictiveHeuristicAgent()
        else:
            raise ValueError(f"Unknown heuristic: {agent_type}")

        self.algo = agent_type
        logger.info(f"✅ Loaded heuristic agent: {agent_type}")

    def load_from_registry(self, company_name: str, agent_name: str):
        """Load an agent from the registry."""
        registry = AgentRegistry()
        metadata = registry.get_agent(company_name, agent_name)

        if not metadata:
            raise FileNotFoundError(f"Agent '{agent_name}' not found for company '{company_name}'")

        self.agent_name = metadata.agent_name
        self.company_name = metadata.company_name

        if metadata.algorithm in ("dqn", "ppo"):
            if metadata.model_path and os.path.exists(metadata.model_path + ".zip"):
                self.load_rl_model(metadata.model_path, metadata.algorithm)
            elif metadata.model_path and os.path.exists(metadata.model_path):
                self.load_rl_model(metadata.model_path, metadata.algorithm)
            else:
                logger.warning(
                    f"No trained model found at '{metadata.model_path}'. " f"Falling back to threshold heuristic."
                )
                self.load_heuristic_agent("threshold")
        elif metadata.algorithm in ("threshold", "predictive"):
            self.load_heuristic_agent(metadata.algorithm)
        else:
            logger.warning(f"Unknown algorithm '{metadata.algorithm}', using threshold.")
            self.load_heuristic_agent("threshold")

        # Update status
        registry.update_agent_status(company_name, agent_name, "deployed", deployment_endpoint=self.ingress_url)

    def predict(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent's policy on telemetry data."""
        if self.model is not None:
            # RL agent
            obs_array = telemetry_to_array(telemetry, self.step_count)
            action, _ = self.model.predict(obs_array, deterministic=True)
            return action_to_decision(int(action), f"[{self.algo.upper()}] Step {self.step_count}")
        elif self.heuristic_agent is not None:
            # Heuristic agent
            obs = telemetry_to_observation(telemetry, self.step_count)
            action = self.heuristic_agent.act(obs)
            return {
                "action": action.action_type.name,
                "reasoning": action.reasoning,
            }
        else:
            return {"action": "NO_OP", "reasoning": "No agent loaded"}

    def get_telemetry(self) -> Dict[str, Any]:
        """Pull latest telemetry from the Ingress API."""
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        resp = requests.get(f"{self.ingress_url}/telemetry/latest", headers=headers)
        resp.raise_for_status()
        return resp.json()

    def post_decision(self, decision: Dict[str, Any]):
        """Post scaling decision to the Ingress API."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        resp = requests.post(f"{self.ingress_url}/decision", json=decision, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def run(self, max_steps: Optional[int] = None):
        """
        Main deployment loop.

        Continuously polls telemetry, makes decisions, and posts them.
        """
        logger.info("=" * 60)
        logger.info(f"🚀 DEPLOYING AGENT")
        logger.info(f"   Agent:    {self.agent_name}")
        logger.info(f"   Company:  {self.company_name}")
        logger.info(f"   Algo:     {self.algo}")
        logger.info(f"   Ingress:  {self.ingress_url}")
        logger.info(f"   Interval: {self.poll_interval}s")
        logger.info("=" * 60)

        try:
            while True:
                if max_steps and self.step_count >= max_steps:
                    logger.info(f"Reached max steps ({max_steps}). Stopping.")
                    break

                try:
                    # 1. Get telemetry
                    telemetry = self.get_telemetry()

                    # 2. Make decision
                    decision = self.predict(telemetry)
                    decision["timestamp"] = time.time()

                    # 3. Post decision
                    self.post_decision(decision)

                    cpu = telemetry.get("cpu_load", 0)
                    rps = telemetry.get("request_count", 0)
                    logger.info(
                        f"Step {self.step_count:04d} | "
                        f"CPU={cpu:.0%} RPS={rps} → "
                        f"{decision['action']} | {decision['reasoning']}"
                    )

                except requests.exceptions.ConnectionError:
                    logger.error(
                        f"Cannot reach ingress at {self.ingress_url}. " f"Retrying in {self.poll_interval}s..."
                    )
                except Exception as e:
                    logger.error(f"Error: {e}")

                self.step_count += 1
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            logger.info("\n🛑 Agent deployment stopped by user.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy a trained Cloud Cost Optimizer agent to production")
    parser.add_argument("--company", type=str, help="Company name (from registry)")
    parser.add_argument("--agent", type=str, help="Agent name (from registry)")
    parser.add_argument("--model-path", type=str, help="Direct path to model file")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo", "threshold", "predictive"])
    parser.add_argument("--ingress-url", type=str, default="http://localhost:8000")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--interval", type=int, default=10, help="Poll interval (seconds)")
    parser.add_argument("--max-steps", type=int, default=None)

    args = parser.parse_args()

    deployer = AgentDeployer(
        ingress_url=args.ingress_url,
        api_key=args.api_key,
        poll_interval=args.interval,
    )

    if args.company and args.agent:
        # Load from registry
        deployer.load_from_registry(args.company, args.agent)
    elif args.model_path:
        # Load model directly
        if args.algo in ("dqn", "ppo"):
            deployer.load_rl_model(args.model_path, args.algo)
        else:
            deployer.load_heuristic_agent(args.algo)
        deployer.agent_name = os.path.basename(args.model_path)
    else:
        # Default: threshold heuristic
        deployer.load_heuristic_agent("threshold")
        deployer.agent_name = "DefaultThreshold"

    deployer.run(max_steps=args.max_steps)
