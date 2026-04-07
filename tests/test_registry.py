import json
import os
import shutil
from pathlib import Path

import pytest

from agent.registry import AgentMetadata, AgentRegistry


@pytest.fixture
def registry(tmp_path):
    base_dir = tmp_path / "test_agents"
    return AgentRegistry(base_dir=str(base_dir))


def test_company_registration(registry):
    result = registry.register_company("Acme Corp", description="Testing", industry="SaaS")
    assert result["status"] == "registered"
    assert result["company"]["name"] == "Acme Corp"

    companies = registry.list_companies()
    assert len(companies) == 1
    assert companies[0]["name"] == "Acme Corp"

    # Duplicate registration
    result = registry.register_company("Acme Corp")
    assert result["status"] == "exists"


def test_agent_creation(registry):
    registry.register_company("Acme Corp")
    metadata = registry.create_agent(
        company_name="Acme Corp", agent_name="Test Scaler", algorithm="dqn", description="A test agent"
    )

    assert metadata.agent_name == "Test Scaler"
    assert metadata.company_name == "Acme Corp"
    assert metadata.status == "created"
    assert "acme_corp/test_scaler" in metadata.agent_id

    # Duplicate agent
    with pytest.raises(ValueError):
        registry.create_agent("Acme Corp", "Test Scaler")


def test_get_agent(registry):
    registry.register_company("Acme Corp")
    registry.create_agent("Acme Corp", "Test Scaler")

    metadata = registry.get_agent("Acme Corp", "Test Scaler")
    assert metadata is not None
    assert metadata.agent_name == "Test Scaler"

    non_existent = registry.get_agent("Acme Corp", "No Such Agent")
    assert non_existent is None


def test_update_agent_status(registry):
    registry.register_company("Acme Corp")
    registry.create_agent("Acme Corp", "Test Scaler")

    registry.update_agent_status("Acme Corp", "Test Scaler", "trained", version=2)

    metadata = registry.get_agent("Acme Corp", "Test Scaler")
    assert metadata.status == "trained"
    assert metadata.version == 2


def test_delete_agent(registry):
    registry.register_company("Acme Corp")
    registry.create_agent("Acme Corp", "Test Scaler")

    registry.delete_agent("Acme Corp", "Test Scaler")
    assert registry.get_agent("Acme Corp", "Test Scaler") is None

    agents = registry.list_agents("Acme Corp")
    assert len(agents) == 0


def test_custom_traffic(registry):
    registry.register_company("Acme Corp")
    registry.create_agent("Acme Corp", "Test Scaler")

    traffic = [10.0, 20.0, 30.0]
    path = registry.save_custom_traffic("Acme Corp", "Test Scaler", traffic)

    assert os.path.exists(path)
    metadata = registry.get_agent("Acme Corp", "Test Scaler")
    assert metadata.traffic_profile == "custom"
    assert str(path) in metadata.custom_traffic_data


def test_export_agent(registry, tmp_path):
    registry.register_company("Acme Corp")
    registry.create_agent("Acme Corp", "Test Scaler")

    export_dir = tmp_path / "exports"
    path = registry.export_agent("Acme Corp", "Test Scaler", export_dir=str(export_dir))

    assert os.path.exists(path)
    assert os.path.exists(os.path.join(path, "metadata.json"))
    assert os.path.exists(os.path.join(path, "deploy_config.json"))
    assert os.path.exists(os.path.join(path, "inference.py"))
