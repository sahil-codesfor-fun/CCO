"""
DQN Training Script — Trains a Deep Q-Network agent for cloud auto-scaling.

Uses Stable-Baselines3 for the DQN implementation.
The agent learns to make optimal scaling decisions by interacting with
the CloudCostOptimizer environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("WARNING: stable-baselines3 not installed. Training disabled.")

from env.environment import (
    CloudCostOptimizerEnv,
    make_chaos_env,
    make_spike_env,
    make_steady_env,
)


def create_training_env(task: str = "steady", seed: int = 42) -> CloudCostOptimizerEnv:
    """Create the appropriate environment for the given task."""
    if task == "steady":
        return make_steady_env(seed=seed)
    elif task == "spike":
        return make_spike_env(seed=seed)
    elif task == "chaos":
        return make_chaos_env(seed=seed)
    else:
        raise ValueError(f"Unknown task: {task}")


def train_dqn(
    task: str = "steady",
    total_timesteps: int = 200_000,
    learning_rate: float = 1e-3,
    buffer_size: int = 50_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.05,
    target_update_interval: int = 1000,
    seed: int = 42,
    save_dir: str = "models",
    verbose: int = 1,
) -> str:
    """
    Train a DQN agent on the specified task.

    Args:
        task: Task name ("steady", "spike", "chaos")
        total_timesteps: Total training steps
        learning_rate: Learning rate for the optimizer
        buffer_size: Replay buffer size
        batch_size: Mini-batch size
        gamma: Discount factor
        exploration_fraction: Fraction of training for epsilon decay
        exploration_final_eps: Final epsilon value
        target_update_interval: Steps between target network updates
        seed: Random seed for reproducibility
        save_dir: Directory to save trained models
        verbose: Verbosity level

    Returns:
        Path to the saved model
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required for training. Install with: pip install stable-baselines3")

    # Create environments
    train_env = Monitor(create_training_env(task, seed=seed))
    eval_env = Monitor(create_training_env(task, seed=seed + 100))

    # Set up saving
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"dqn_{task}_{timestamp}"

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best"),
        log_path=str(save_path / "logs"),
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=verbose,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(save_path / "checkpoints"),
        name_prefix=model_name,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # Create DQN agent
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        seed=seed,
        verbose=verbose,
        tensorboard_log=str(save_path / "tb_logs"),
    )

    print(f"{'='*60}")
    print(f"Training DQN Agent")
    print(f"  Task:            {task}")
    print(f"  Total Steps:     {total_timesteps:,}")
    print(f"  Learning Rate:   {learning_rate}")
    print(f"  Buffer Size:     {buffer_size:,}")
    print(f"  Network:         [256, 256, 128]")
    print(f"  Save Directory:  {save_path}")
    print(f"{'='*60}")

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = str(save_path / model_name)
    model.save(final_path)
    print(f"\nModel saved to: {final_path}.zip")

    return final_path


def train_ppo(
    task: str = "steady",
    total_timesteps: int = 200_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    n_epochs: int = 10,
    clip_range: float = 0.2,
    seed: int = 42,
    save_dir: str = "models",
    verbose: int = 1,
) -> str:
    """Train a PPO agent (Phase 2 — industry favorite)."""
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required for training.")

    train_env = Monitor(create_training_env(task, seed=seed))
    eval_env = Monitor(create_training_env(task, seed=seed + 100))

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ppo_{task}_{timestamp}"

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best"),
        log_path=str(save_path / "logs"),
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=verbose,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_epochs=n_epochs,
        clip_range=clip_range,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        seed=seed,
        verbose=verbose,
        tensorboard_log=str(save_path / "tb_logs"),
    )

    print(f"{'='*60}")
    print(f"Training PPO Agent")
    print(f"  Task:            {task}")
    print(f"  Total Steps:     {total_timesteps:,}")
    print(f"{'='*60}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    final_path = str(save_path / model_name)
    model.save(final_path)
    print(f"\nModel saved to: {final_path}.zip")

    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for Cloud Cost Optimizer")
    parser.add_argument("--task", type=str, default="steady", choices=["steady", "spike", "chaos"])
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="models")

    args = parser.parse_args()

    if args.algo == "dqn":
        train_dqn(
            task=args.task,
            total_timesteps=args.steps,
            learning_rate=args.lr,
            seed=args.seed,
            save_dir=args.save_dir,
        )
    else:
        train_ppo(
            task=args.task,
            total_timesteps=args.steps,
            learning_rate=args.lr,
            seed=args.seed,
            save_dir=args.save_dir,
        )
