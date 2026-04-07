"""
Agent Registry — Save, load, list, and manage trained AI agents.

This is the core system that allows companies to:
1. Register their company and create named agents
2. Train agents on custom or preset traffic data
3. Save trained agents with full metadata
4. Export agents for deployment
5. Load agents for inference/production use
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Agent Metadata
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentMetadata:
    """Full metadata for a registered agent."""

    agent_id: str  # Unique ID (company_name + agent_name)
    agent_name: str  # Custom name given by the company
    company_name: str  # Company that owns this agent
    algorithm: str  # "dqn", "ppo", "threshold", "predictive", "custom"
    created_at: str  # ISO timestamp
    updated_at: str  # ISO timestamp
    training_config: Dict[str, Any]  # Full training hyperparameters
    performance: Dict[str, Any]  # Last evaluation scores
    traffic_profile: str  # "steady", "spike", "chaos", "custom"
    total_training_steps: int  # How many steps trained
    model_path: Optional[str] = None  # Path to saved model file
    custom_traffic_data: Optional[str] = None  # Path to custom traffic JSON
    description: str = ""  # Agent description
    version: int = 1  # Version number (increments on retrain)
    status: str = "created"  # "created", "training", "trained", "deployed", "archived"
    deployment_endpoint: Optional[str] = None  # Where it's deployed
    tags: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent Registry
# ─────────────────────────────────────────────────────────────────────────────


class AgentRegistry:
    """
    Central registry for all company agents.

    Directory structure:
        agents/
        ├── registry.json          ← Master index of all agents
        ├── acme_corp/
        │   ├── autoscaler_v1/
        │   │   ├── metadata.json  ← Agent metadata
        │   │   ├── model.zip      ← Trained model weights
        │   │   └── traffic.json   ← Custom traffic data (if any)
        │   └── smart_scaler/
        │       ├── metadata.json
        │       └── model.zip
        └── netflix/
            └── titus_optimizer/
                ├── metadata.json
                └── model.zip
    """

    def __init__(self, base_dir: str = "agents"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_dir / "registry.json"
        self._load_registry()

    def _load_registry(self):
        """Load or create the master registry index."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                self._registry = json.load(f)
        else:
            self._registry = {"agents": {}, "companies": {}}
            self._save_registry()

    def _save_registry(self):
        """Persist the registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def _make_agent_id(self, company_name: str, agent_name: str) -> str:
        """Create a unique agent ID from company and agent name."""
        company_slug = company_name.lower().replace(" ", "_").replace("-", "_")
        agent_slug = agent_name.lower().replace(" ", "_").replace("-", "_")
        return f"{company_slug}/{agent_slug}"

    def _get_agent_dir(self, agent_id: str) -> Path:
        """Get the directory for an agent."""
        return self.base_dir / agent_id

    # ─── Company Management ─────────────────────────────────────────────

    def register_company(
        self,
        company_name: str,
        description: str = "",
        contact_email: str = "",
        industry: str = "",
    ) -> Dict[str, Any]:
        """Register a new company in the system."""
        company_slug = company_name.lower().replace(" ", "_").replace("-", "_")

        if company_slug in self._registry["companies"]:
            return {
                "status": "exists",
                "message": f"Company '{company_name}' is already registered.",
                "company": self._registry["companies"][company_slug],
            }

        company_data = {
            "name": company_name,
            "slug": company_slug,
            "description": description,
            "contact_email": contact_email,
            "industry": industry,
            "registered_at": datetime.now().isoformat(),
            "agents": [],
        }

        self._registry["companies"][company_slug] = company_data
        company_dir = self.base_dir / company_slug
        company_dir.mkdir(parents=True, exist_ok=True)
        self._save_registry()

        return {
            "status": "registered",
            "message": f"✅ Company '{company_name}' registered successfully!",
            "company": company_data,
        }

    def list_companies(self) -> List[Dict[str, Any]]:
        """List all registered companies."""
        return list(self._registry["companies"].values())

    # ─── Agent Management ────────────────────────────────────────────────

    def create_agent(
        self,
        company_name: str,
        agent_name: str,
        algorithm: str = "dqn",
        traffic_profile: str = "steady",
        description: str = "",
        training_config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> AgentMetadata:
        """
        Create a new agent registration for a company.

        Args:
            company_name: The company that owns this agent
            agent_name: Custom name for the agent (e.g., "BlackFriday Scaler")
            algorithm: Training algorithm ("dqn", "ppo", "threshold", "predictive")
            traffic_profile: Which traffic to train on
            description: Human-readable description
            training_config: Hyperparameters for training
            tags: Labels for organization

        Returns:
            AgentMetadata with the full agent record
        """
        agent_id = self._make_agent_id(company_name, agent_name)

        # Ensure company exists
        company_slug = company_name.lower().replace(" ", "_").replace("-", "_")
        if company_slug not in self._registry["companies"]:
            self.register_company(company_name)

        # Check for duplicates
        if agent_id in self._registry["agents"]:
            existing = self._registry["agents"][agent_id]
            raise ValueError(
                f"Agent '{agent_name}' already exists for company '{company_name}'. "
                f"Use update_agent() to modify it, or choose a different name."
            )

        now = datetime.now().isoformat()

        default_config = {
            "total_timesteps": 200_000,
            "learning_rate": 1e-3,
            "buffer_size": 50_000,
            "batch_size": 64,
            "gamma": 0.99,
            "exploration_fraction": 0.3,
            "seed": 42,
        }

        if training_config:
            default_config.update(training_config)

        metadata = AgentMetadata(
            agent_id=agent_id,
            agent_name=agent_name,
            company_name=company_name,
            algorithm=algorithm,
            created_at=now,
            updated_at=now,
            training_config=default_config,
            performance={},
            traffic_profile=traffic_profile,
            total_training_steps=0,
            description=description,
            tags=tags or [],
            status="created",
        )

        # Create directory
        agent_dir = self._get_agent_dir(agent_id)
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(agent_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        # Update registry
        self._registry["agents"][agent_id] = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "company": company_name,
            "algorithm": algorithm,
            "status": "created",
            "created_at": now,
        }
        self._registry["companies"][company_slug]["agents"].append(agent_id)
        self._save_registry()

        return metadata

    def get_agent(self, company_name: str, agent_name: str) -> Optional[AgentMetadata]:
        """Load agent metadata by company and agent name."""
        agent_id = self._make_agent_id(company_name, agent_name)
        agent_dir = self._get_agent_dir(agent_id)
        meta_path = agent_dir / "metadata.json"

        if not meta_path.exists():
            return None

        with open(meta_path, "r") as f:
            data = json.load(f)
        return AgentMetadata(**data)

    def list_agents(self, company_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all agents, optionally filtered by company."""
        agents = list(self._registry["agents"].values())
        if company_name:
            company_slug = company_name.lower().replace(" ", "_").replace("-", "_")
            agents = [a for a in agents if a.get("company", "").lower().replace(" ", "_") == company_slug]
        return agents

    def update_agent_status(self, company_name: str, agent_name: str, status: str, **extras):
        """Update an agent's status and optional extra fields."""
        agent_id = self._make_agent_id(company_name, agent_name)
        agent_dir = self._get_agent_dir(agent_id)
        meta_path = agent_dir / "metadata.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"Agent '{agent_name}' not found for '{company_name}'")

        with open(meta_path, "r") as f:
            data = json.load(f)

        data["status"] = status
        data["updated_at"] = datetime.now().isoformat()
        data.update(extras)

        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)

        # Update registry index
        if agent_id in self._registry["agents"]:
            self._registry["agents"][agent_id]["status"] = status
            self._save_registry()

    def update_agent_performance(self, company_name: str, agent_name: str, performance: Dict[str, Any]):
        """Update the agent's performance metrics after evaluation."""
        agent_id = self._make_agent_id(company_name, agent_name)
        agent_dir = self._get_agent_dir(agent_id)
        meta_path = agent_dir / "metadata.json"

        with open(meta_path, "r") as f:
            data = json.load(f)

        data["performance"] = performance
        data["updated_at"] = datetime.now().isoformat()

        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)

    def save_custom_traffic(self, company_name: str, agent_name: str, traffic_data: List[float]) -> str:
        """Save custom traffic data for an agent."""
        agent_id = self._make_agent_id(company_name, agent_name)
        agent_dir = self._get_agent_dir(agent_id)
        traffic_path = agent_dir / "traffic.json"

        with open(traffic_path, "w") as f:
            json.dump(traffic_data, f)

        # Update metadata
        meta_path = agent_dir / "metadata.json"
        with open(meta_path, "r") as f:
            data = json.load(f)
        data["custom_traffic_data"] = str(traffic_path)
        data["traffic_profile"] = "custom"
        data["updated_at"] = datetime.now().isoformat()
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)

        return str(traffic_path)

    def delete_agent(self, company_name: str, agent_name: str) -> bool:
        """Delete an agent and all its files."""
        agent_id = self._make_agent_id(company_name, agent_name)
        agent_dir = self._get_agent_dir(agent_id)

        if agent_dir.exists():
            shutil.rmtree(agent_dir)

        if agent_id in self._registry["agents"]:
            del self._registry["agents"][agent_id]

        company_slug = company_name.lower().replace(" ", "_").replace("-", "_")
        if company_slug in self._registry["companies"]:
            agents = self._registry["companies"][company_slug].get("agents", [])
            if agent_id in agents:
                agents.remove(agent_id)

        self._save_registry()
        return True

    # ─── Export / Deploy ─────────────────────────────────────────────────

    def export_agent(self, company_name: str, agent_name: str, export_dir: str = "exports") -> str:
        """
        Export a trained agent as a standalone package.

        The export includes:
        - model.zip (trained weights)
        - metadata.json (full config & performance)
        - deploy_config.json (deployment instructions)
        - inference.py (standalone inference script)
        """
        agent_id = self._make_agent_id(company_name, agent_name)
        agent_dir = self._get_agent_dir(agent_id)

        if not agent_dir.exists():
            raise FileNotFoundError(f"Agent '{agent_name}' not found for '{company_name}'")

        export_path = Path(export_dir)
        export_name = f"{agent_id.replace('/', '_')}_export"
        export_agent_dir = export_path / export_name
        export_agent_dir.mkdir(parents=True, exist_ok=True)

        # Copy metadata
        if (agent_dir / "metadata.json").exists():
            shutil.copy(agent_dir / "metadata.json", export_agent_dir / "metadata.json")

        # Copy model
        model_files = list(agent_dir.glob("*.zip"))
        for mf in model_files:
            shutil.copy(mf, export_agent_dir / mf.name)

        # Copy custom traffic if present
        if (agent_dir / "traffic.json").exists():
            shutil.copy(agent_dir / "traffic.json", export_agent_dir / "traffic.json")

        # Create deployment config
        deploy_config = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "company": company_name,
            "ingress_endpoint": "http://localhost:8000",
            "poll_interval_seconds": 10,
            "auto_deploy": False,
            "instructions": {
                "1_start_ingress": "python scripts/ingress_server.py",
                "2_run_agent": f"python agent/deploy.py --agent-dir {export_agent_dir}",
                "3_poll_decision": "curl http://localhost:8000/decision",
            },
        }
        with open(export_agent_dir / "deploy_config.json", "w") as f:
            json.dump(deploy_config, f, indent=2)

        # Create standalone inference script
        inference_script = f'''#!/usr/bin/env python3
"""
{agent_name} — Standalone Inference Agent
Company: {company_name}
Generated by Cloud Cost Optimizer

Usage:
    python inference.py --ingress-url http://localhost:8000
"""

import json
import time
import argparse
import requests

def load_metadata():
    with open("metadata.json", "r") as f:
        return json.load(f)

def get_telemetry(ingress_url):
    """Get latest telemetry from the ingress API."""
    resp = requests.get(f"{{ingress_url}}/telemetry/latest")
    return resp.json()

def make_decision(telemetry, metadata):
    """
    Simple threshold-based decision engine.
    For DQN/PPO agents, load the model and use model.predict().
    """
    cpu = telemetry.get("cpu_load", 0)
    rps = telemetry.get("request_count", 0)

    if cpu > 0.85:
        return {{"action": "SCALE_UP_3", "reasoning": f"Critical CPU: {{cpu:.0%}}"}}
    elif cpu > 0.70:
        return {{"action": "SCALE_UP_1", "reasoning": f"High CPU: {{cpu:.0%}}"}}
    elif cpu < 0.20:
        return {{"action": "SCALE_DOWN_1", "reasoning": f"Low CPU: {{cpu:.0%}}"}}
    else:
        return {{"action": "NO_OP", "reasoning": f"Stable at {{cpu:.0%}}"}}

def post_decision(ingress_url, decision, api_key=None):
    """Post the scaling decision to the ingress API."""
    headers = {{"Content-Type": "application/json"}}
    if api_key:
        headers["X-API-Key"] = api_key
    resp = requests.post(f"{{ingress_url}}/decision", json=decision, headers=headers)
    return resp.json()

def main():
    parser = argparse.ArgumentParser(description="{agent_name} Inference Agent")
    parser.add_argument("--ingress-url", default="http://localhost:8000")
    parser.add_argument("--interval", type=int, default=10, help="Poll interval in seconds")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    metadata = load_metadata()
    print(f"🤖 Agent: {{metadata['agent_name']}}")
    print(f"🏢 Company: {{metadata['company_name']}}")
    print(f"📡 Ingress: {{args.ingress_url}}")
    print(f"⏱️  Interval: {{args.interval}}s")
    print("=" * 50)

    while True:
        try:
            telemetry = get_telemetry(args.ingress_url)
            decision = make_decision(telemetry, metadata)
            result = post_decision(args.ingress_url, decision, args.api_key)
            print(f"[{{time.strftime('%H:%M:%S')}}] CPU={{telemetry.get('cpu_load', 0):.0%}} "
                  f"→ {{decision['action']}} | {{decision['reasoning']}}")
        except Exception as e:
            print(f"[{{time.strftime('%H:%M:%S')}}] ERROR: {{e}}")

        time.sleep(args.interval)

if __name__ == "__main__":
    main()
'''
        with open(export_agent_dir / "inference.py", "w") as f:
            f.write(inference_script)

        return str(export_agent_dir)

    def get_agent_summary(self, company_name: str, agent_name: str) -> Dict[str, Any]:
        """Get a comprehensive summary of an agent."""
        metadata = self.get_agent(company_name, agent_name)
        if not metadata:
            return {"error": f"Agent '{agent_name}' not found for '{company_name}'"}

        agent_dir = self._get_agent_dir(metadata.agent_id)
        has_model = any(agent_dir.glob("*.zip"))
        has_traffic = (agent_dir / "traffic.json").exists()

        return {
            "agent_id": metadata.agent_id,
            "agent_name": metadata.agent_name,
            "company": metadata.company_name,
            "algorithm": metadata.algorithm,
            "status": metadata.status,
            "traffic_profile": metadata.traffic_profile,
            "total_training_steps": metadata.total_training_steps,
            "performance": metadata.performance,
            "has_model": has_model,
            "has_custom_traffic": has_traffic,
            "description": metadata.description,
            "version": metadata.version,
            "created_at": metadata.created_at,
            "updated_at": metadata.updated_at,
            "tags": metadata.tags,
        }
