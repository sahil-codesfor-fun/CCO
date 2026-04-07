#!/usr/bin/env python3
"""
Company Agent Builder CLI — The main entry point for companies.

This is the command-line tool that allows any company to:
1. Register their company
2. Create and name a custom AI agent
3. Upload their own traffic data
4. Train the agent on their data
5. Evaluate performance
6. Export for deployment
7. Deploy to production via the Ingress API

Usage Examples:
    # Register your company
    python build_agent.py register --company "Acme Corp" --industry "E-commerce"

    # Create a named agent
    python build_agent.py create --company "Acme Corp" --agent "BlackFriday Scaler" \\
        --algo dqn --traffic spike --description "Handles Black Friday traffic"

    # Upload custom traffic data
    python build_agent.py upload-traffic --company "Acme Corp" --agent "BlackFriday Scaler" \\
        --file my_traffic.json

    # Train the agent
    python build_agent.py train --company "Acme Corp" --agent "BlackFriday Scaler" --steps 200000

    # Evaluate performance
    python build_agent.py evaluate --company "Acme Corp" --agent "BlackFriday Scaler"

    # Export for deployment
    python build_agent.py export --company "Acme Corp" --agent "BlackFriday Scaler"

    # Deploy to production
    python build_agent.py deploy --company "Acme Corp" --agent "BlackFriday Scaler" \\
        --ingress-url http://localhost:8000

    # List all agents
    python build_agent.py list

    # View agent details
    python build_agent.py info --company "Acme Corp" --agent "BlackFriday Scaler"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.registry import AgentMetadata, AgentRegistry


def cmd_register(args):
    """Register a new company."""
    registry = AgentRegistry()
    result = registry.register_company(
        company_name=args.company,
        description=args.description or "",
        contact_email=args.email or "",
        industry=args.industry or "",
    )

    print(f"\n{'='*60}")
    print(f"🏢 Company Registration")
    print(f"{'='*60}")
    print(f"  Status:   {result['status']}")
    print(f"  Message:  {result['message']}")
    if result["status"] == "registered":
        print(f"  Company:  {result['company']['name']}")
        print(f"  Industry: {result['company'].get('industry', 'N/A')}")
    print()


def cmd_create(args):
    """Create a new named agent."""
    registry = AgentRegistry()

    training_config = {}
    if args.lr:
        training_config["learning_rate"] = args.lr
    if args.steps:
        training_config["total_timesteps"] = args.steps
    if args.seed:
        training_config["seed"] = args.seed

    tags = args.tags.split(",") if args.tags else []

    try:
        metadata = registry.create_agent(
            company_name=args.company,
            agent_name=args.agent,
            algorithm=args.algo,
            traffic_profile=args.traffic,
            description=args.description or "",
            training_config=training_config,
            tags=tags,
        )

        print(f"\n{'='*60}")
        print(f"🤖 Agent Created Successfully!")
        print(f"{'='*60}")
        print(f"  Agent ID:   {metadata.agent_id}")
        print(f"  Name:       {metadata.agent_name}")
        print(f"  Company:    {metadata.company_name}")
        print(f"  Algorithm:  {metadata.algorithm}")
        print(f"  Traffic:    {metadata.traffic_profile}")
        print(f"  Status:     {metadata.status}")
        print(f"\n  Next step: Train your agent with:")
        print(f'  python build_agent.py train --company "{args.company}" --agent "{args.agent}"')
        print()

    except ValueError as e:
        print(f"\n❌ Error: {e}")


def cmd_upload_traffic(args):
    """Upload custom traffic data for an agent."""
    registry = AgentRegistry()

    # Load traffic data
    if args.file:
        with open(args.file, "r") as f:
            traffic_data = json.load(f)
        if not isinstance(traffic_data, list):
            print("❌ Traffic file must be a JSON array of numbers (requests per step).")
            return
    else:
        print("❌ Please provide a traffic file with --file")
        return

    path = registry.save_custom_traffic(args.company, args.agent, traffic_data)

    print(f"\n{'='*60}")
    print(f"📡 Custom Traffic Data Uploaded")
    print(f"{'='*60}")
    print(f"  Agent:       {args.agent}")
    print(f"  Data Points: {len(traffic_data)}")
    print(f"  Min Traffic: {min(traffic_data):.1f} req/s")
    print(f"  Max Traffic: {max(traffic_data):.1f} req/s")
    print(f"  Avg Traffic: {sum(traffic_data)/len(traffic_data):.1f} req/s")
    print(f"  Saved to:    {path}")
    print()


def cmd_train(args):
    """Train an agent."""
    registry = AgentRegistry()
    metadata = registry.get_agent(args.company, args.agent)

    if not metadata:
        print(f"❌ Agent '{args.agent}' not found for company '{args.company}'")
        print(f'   Create one first: python build_agent.py create --company "{args.company}" --agent "{args.agent}"')
        return

    algo = metadata.algorithm
    traffic = metadata.traffic_profile
    steps = args.steps or metadata.training_config.get("total_timesteps", 200_000)
    lr = args.lr or metadata.training_config.get("learning_rate", 1e-3)
    seed = args.seed or metadata.training_config.get("seed", 42)

    print(f"\n{'='*60}")
    print(f"🏋️ Training Agent: {metadata.agent_name}")
    print(f"{'='*60}")
    print(f"  Company:    {metadata.company_name}")
    print(f"  Algorithm:  {algo}")
    print(f"  Traffic:    {traffic}")
    print(f"  Steps:      {steps:,}")
    print(f"  LR:         {lr}")
    print(f"  Seed:       {seed}")
    print(f"{'='*60}\n")

    # Update status
    registry.update_agent_status(args.company, args.agent, "training")

    # Determine save directory
    agent_dir = registry._get_agent_dir(metadata.agent_id)
    save_dir = str(agent_dir)

    if algo in ("dqn", "ppo"):
        # Use the training script
        from agent.train import train_dqn, train_ppo

        try:
            if algo == "dqn":
                model_path = train_dqn(
                    task=traffic if traffic != "custom" else "steady",
                    total_timesteps=steps,
                    learning_rate=lr,
                    seed=seed,
                    save_dir=save_dir,
                )
            else:
                model_path = train_ppo(
                    task=traffic if traffic != "custom" else "steady",
                    total_timesteps=steps,
                    learning_rate=lr,
                    seed=seed,
                    save_dir=save_dir,
                )

            # Update metadata
            registry.update_agent_status(
                args.company,
                args.agent,
                "trained",
                model_path=model_path,
                total_training_steps=steps,
                version=metadata.version + 1,
            )

            print(f"\n✅ Training Complete!")
            print(f"  Model saved: {model_path}")
            print(f"  Status:      trained")
            print(f"\n  Next: Evaluate with:")
            print(f'  python build_agent.py evaluate --company "{args.company}" --agent "{args.agent}"')

        except ImportError as e:
            print(f"\n❌ Training failed: {e}")
            print(f"   Install: pip install stable-baselines3")
            registry.update_agent_status(args.company, args.agent, "created")

    elif algo in ("threshold", "predictive"):
        # Heuristic agents don't need training
        registry.update_agent_status(
            args.company,
            args.agent,
            "trained",
            total_training_steps=0,
        )
        print(f"\n✅ Heuristic agent '{algo}' is ready (no training needed)")
        print(f"   It uses pre-defined rules that work immediately.")

    else:
        print(f"❌ Unknown algorithm: {algo}")


def cmd_evaluate(args):
    """Evaluate an agent's performance."""
    registry = AgentRegistry()
    metadata = registry.get_agent(args.company, args.agent)

    if not metadata:
        print(f"❌ Agent '{args.agent}' not found for company '{args.company}'")
        return

    from agent.baselines import PredictiveHeuristicAgent, ThresholdAgent, run_baseline
    from agent.evaluate import evaluate_baseline_agents, evaluate_trained_agent
    from env.environment import make_chaos_env, make_spike_env, make_steady_env
    from tasks.graders import grade_task

    print(f"\n{'='*60}")
    print(f"📊 Evaluating Agent: {metadata.agent_name}")
    print(f"{'='*60}")

    tasks = ["steady", "spike", "chaos"]
    seed = args.seed or 42

    if metadata.algorithm in ("dqn", "ppo") and metadata.model_path:
        results = evaluate_trained_agent(
            model_path=metadata.model_path,
            algo=metadata.algorithm,
            tasks=tasks,
            seed=seed,
        )
    elif metadata.algorithm == "threshold":
        agent = ThresholdAgent()
        results = {}
        task_envs = {"steady": make_steady_env, "spike": make_spike_env, "chaos": make_chaos_env}
        task_yamls = {
            "steady": "tasks/task_steady.yaml",
            "spike": "tasks/task_spike.yaml",
            "chaos": "tasks/task_chaos.yaml",
        }
        for task in tasks:
            env = task_envs[task](seed=seed)
            result = run_baseline(agent, env, seed=seed)
            grade = grade_task(task_yamls[task], result["episode_info"], result["history"])
            results[task] = {
                "total_score": grade["total_score"],
                "sla_score": grade["sla_score"],
                "cost_score": grade["cost_score"],
                "latency_score": grade["latency_score"],
                "stability_score": grade["stability_score"],
            }
            print(f"\n  {task.upper():15s}  Score: {grade['total_score']:.4f}")
    elif metadata.algorithm == "predictive":
        agent = PredictiveHeuristicAgent()
        results = {}
        task_envs = {"steady": make_steady_env, "spike": make_spike_env, "chaos": make_chaos_env}
        task_yamls = {
            "steady": "tasks/task_steady.yaml",
            "spike": "tasks/task_spike.yaml",
            "chaos": "tasks/task_chaos.yaml",
        }
        for task in tasks:
            env = task_envs[task](seed=seed)
            result = run_baseline(agent, env, seed=seed)
            grade = grade_task(task_yamls[task], result["episode_info"], result["history"])
            results[task] = {
                "total_score": grade["total_score"],
                "sla_score": grade["sla_score"],
                "cost_score": grade["cost_score"],
                "latency_score": grade["latency_score"],
                "stability_score": grade["stability_score"],
            }
            print(f"\n  {task.upper():15s}  Score: {grade['total_score']:.4f}")
    else:
        print(f"❌ Cannot evaluate: no model found for algorithm '{metadata.algorithm}'")
        return

    # Save performance
    registry.update_agent_performance(args.company, args.agent, results)

    print(f"\n✅ Evaluation complete! Results saved to agent metadata.")
    print(f'   View: python build_agent.py info --company "{args.company}" --agent "{args.agent}"')


def cmd_export(args):
    """Export an agent for standalone deployment."""
    registry = AgentRegistry()

    export_dir = args.output or "exports"
    path = registry.export_agent(args.company, args.agent, export_dir)

    print(f"\n{'='*60}")
    print(f"📦 Agent Exported Successfully!")
    print(f"{'='*60}")
    print(f"  Agent:      {args.agent}")
    print(f"  Company:    {args.company}")
    print(f"  Export Dir: {path}")
    print(f"\n  Contents:")

    export_path = Path(path)
    for f in sorted(export_path.iterdir()):
        size = f.stat().st_size
        print(f"    📄 {f.name:30s} ({size:,} bytes)")

    print(f"\n  Deploy instructions:")
    print(f"  1. Start ingress:  python scripts/ingress_server.py")
    print(f"  2. Run agent:      cd {path} && python inference.py")
    print()


def cmd_deploy(args):
    """Deploy an agent to the production Ingress API."""
    from agent.deploy import AgentDeployer

    deployer = AgentDeployer(
        ingress_url=args.ingress_url,
        api_key=args.api_key,
        poll_interval=args.interval,
    )

    deployer.load_from_registry(args.company, args.agent)
    deployer.run(max_steps=args.max_steps)


def cmd_list(args):
    """List all registered agents."""
    registry = AgentRegistry()

    if args.company:
        agents = registry.list_agents(args.company)
        title = f"Agents for {args.company}"
    else:
        agents = registry.list_agents()
        title = "All Registered Agents"

    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print(f"{'='*60}")

    if not agents:
        print("  No agents registered yet.")
        print(f'  Create one: python build_agent.py create --company "YourCompany" --agent "YourAgent"')
    else:
        print(f"  {'ID':<35s} {'Algorithm':<12s} {'Status':<12s} {'Created':<20s}")
        print(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*20}")
        for a in agents:
            print(
                f"  {a['agent_id']:<35s} {a.get('algorithm','?'):<12s} "
                f"{a.get('status','?'):<12s} {a.get('created_at','?')[:19]}"
            )

    # Also list companies
    companies = registry.list_companies()
    if companies:
        print(f"\n  🏢 Registered Companies: {', '.join(c['name'] for c in companies)}")

    print()


def cmd_info(args):
    """Show detailed agent information."""
    registry = AgentRegistry()
    summary = registry.get_agent_summary(args.company, args.agent)

    if "error" in summary:
        print(f"\n❌ {summary['error']}")
        return

    print(f"\n{'='*60}")
    print(f"🤖 Agent Details")
    print(f"{'='*60}")
    print(f"  Agent ID:      {summary['agent_id']}")
    print(f"  Name:          {summary['agent_name']}")
    print(f"  Company:       {summary['company']}")
    print(f"  Algorithm:     {summary['algorithm']}")
    print(f"  Status:        {summary['status']}")
    print(f"  Traffic:       {summary['traffic_profile']}")
    print(f"  Training Steps:{summary['total_training_steps']:,}")
    print(f"  Version:       {summary['version']}")
    print(f"  Has Model:     {'✅' if summary['has_model'] else '❌'}")
    print(f"  Custom Data:   {'✅' if summary['has_custom_traffic'] else '❌'}")
    print(f"  Created:       {summary['created_at']}")
    print(f"  Updated:       {summary['updated_at']}")

    if summary.get("description"):
        print(f"  Description:   {summary['description']}")
    if summary.get("tags"):
        print(f"  Tags:          {', '.join(summary['tags'])}")

    if summary.get("performance"):
        print(f"\n  📊 Performance Scores:")
        for task, scores in summary["performance"].items():
            if isinstance(scores, dict):
                total = scores.get("total_score", "N/A")
                print(f"    {task.upper():15s} → {total}")

    print()


def cmd_delete(args):
    """Delete an agent."""
    registry = AgentRegistry()

    if not args.force:
        confirm = input(f"⚠️  Delete agent '{args.agent}' from '{args.company}'? (y/N): ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    registry.delete_agent(args.company, args.agent)
    print(f"\n🗑️  Agent '{args.agent}' deleted from '{args.company}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="☁️ Cloud Cost Optimizer — Company Agent Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_agent.py register --company "Acme Corp" --industry "E-commerce"
  python build_agent.py create --company "Acme Corp" --agent "Scaler v1" --algo dqn
  python build_agent.py train --company "Acme Corp" --agent "Scaler v1" --steps 100000
  python build_agent.py evaluate --company "Acme Corp" --agent "Scaler v1"
  python build_agent.py export --company "Acme Corp" --agent "Scaler v1"
  python build_agent.py deploy --company "Acme Corp" --agent "Scaler v1"
  python build_agent.py list
  python build_agent.py info --company "Acme Corp" --agent "Scaler v1"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register
    p_register = subparsers.add_parser("register", help="Register a new company")
    p_register.add_argument("--company", required=True, help="Company name")
    p_register.add_argument("--description", help="Company description")
    p_register.add_argument("--email", help="Contact email")
    p_register.add_argument("--industry", help="Industry (e.g., E-commerce, SaaS, Gaming)")

    # Create
    p_create = subparsers.add_parser("create", help="Create a new named agent")
    p_create.add_argument("--company", required=True, help="Company name")
    p_create.add_argument("--agent", required=True, help="Agent name (e.g., 'BlackFriday Scaler')")
    p_create.add_argument("--algo", default="dqn", choices=["dqn", "ppo", "threshold", "predictive"])
    p_create.add_argument("--traffic", default="steady", choices=["steady", "spike", "chaos", "custom"])
    p_create.add_argument("--description", help="Agent description")
    p_create.add_argument("--lr", type=float, help="Learning rate")
    p_create.add_argument("--steps", type=int, help="Training timesteps")
    p_create.add_argument("--seed", type=int, help="Random seed")
    p_create.add_argument("--tags", help="Comma-separated tags")

    # Upload custom traffic
    p_upload = subparsers.add_parser("upload-traffic", help="Upload custom traffic data")
    p_upload.add_argument("--company", required=True)
    p_upload.add_argument("--agent", required=True)
    p_upload.add_argument("--file", required=True, help="Path to JSON file with traffic data")

    # Train
    p_train = subparsers.add_parser("train", help="Train an agent")
    p_train.add_argument("--company", required=True)
    p_train.add_argument("--agent", required=True)
    p_train.add_argument("--steps", type=int, help="Training timesteps (overrides config)")
    p_train.add_argument("--lr", type=float, help="Learning rate (overrides config)")
    p_train.add_argument("--seed", type=int, help="Random seed (overrides config)")

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate agent performance")
    p_eval.add_argument("--company", required=True)
    p_eval.add_argument("--agent", required=True)
    p_eval.add_argument("--seed", type=int, default=42)

    # Export
    p_export = subparsers.add_parser("export", help="Export agent for deployment")
    p_export.add_argument("--company", required=True)
    p_export.add_argument("--agent", required=True)
    p_export.add_argument("--output", help="Export directory (default: exports/)")

    # Deploy
    p_deploy = subparsers.add_parser("deploy", help="Deploy agent to production")
    p_deploy.add_argument("--company", required=True)
    p_deploy.add_argument("--agent", required=True)
    p_deploy.add_argument("--ingress-url", default="http://localhost:8000")
    p_deploy.add_argument("--api-key", help="API key for ingress")
    p_deploy.add_argument("--interval", type=int, default=10, help="Poll interval seconds")
    p_deploy.add_argument("--max-steps", type=int, help="Max deployment steps")

    # List
    p_list = subparsers.add_parser("list", help="List registered agents")
    p_list.add_argument("--company", help="Filter by company")

    # Info
    p_info = subparsers.add_parser("info", help="Show agent details")
    p_info.add_argument("--company", required=True)
    p_info.add_argument("--agent", required=True)

    # Delete
    p_delete = subparsers.add_parser("delete", help="Delete an agent")
    p_delete.add_argument("--company", required=True)
    p_delete.add_argument("--agent", required=True)
    p_delete.add_argument("--force", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n💡 Quick start:")
        print('   python build_agent.py register --company "YourCompany"')
        print('   python build_agent.py create --company "YourCompany" --agent "MyScaler" --algo dqn')
        print('   python build_agent.py train --company "YourCompany" --agent "MyScaler"')
        return

    commands = {
        "register": cmd_register,
        "create": cmd_create,
        "upload-traffic": cmd_upload_traffic,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "deploy": cmd_deploy,
        "list": cmd_list,
        "info": cmd_info,
        "delete": cmd_delete,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
