from unittest.mock import MagicMock, patch

import pytest

from agent.train import create_training_env, train_dqn, train_ppo


def test_create_training_env():
    env = create_training_env("steady")
    assert env is not None

    with pytest.raises(ValueError):
        create_training_env("invalid")


@patch("agent.train.DQN")
@patch("agent.train.Monitor")
@patch("agent.train.EvalCallback")
@patch("agent.train.CheckpointCallback")
def test_train_dqn(mock_checkpoint, mock_eval, mock_monitor, mock_dqn, tmp_path):
    mock_model = MagicMock()
    mock_dqn.return_value = mock_model

    save_dir = tmp_path / "models"
    result = train_dqn(task="steady", total_timesteps=100, save_dir=str(save_dir), verbose=0)

    assert "dqn_steady_" in result
    mock_model.learn.assert_called_once()
    mock_model.save.assert_called_once()


@patch("agent.train.PPO")
@patch("agent.train.Monitor")
@patch("agent.train.EvalCallback")
def test_train_ppo(mock_eval, mock_monitor, mock_ppo, tmp_path):
    mock_model = MagicMock()
    mock_ppo.return_value = mock_model

    save_dir = tmp_path / "models"
    result = train_ppo(task="steady", total_timesteps=100, save_dir=str(save_dir), verbose=0)

    assert "ppo_steady_" in result
    mock_model.learn.assert_called_once()
    mock_model.save.assert_called_once()
