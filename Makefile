.PHONY: install install-dev test lint format validate run-baseline run-llm train-dqn train-ppo run-ingress clean ready docker-build docker-run help register create-agent train-agent evaluate-agent export-agent deploy-agent list-agents

# Variables
PYTHON       = python3
PIP          = $(PYTHON) -m pip
PYTEST       = $(PYTHON) -m pytest
VALIDATE     = validate.py
BASELINE     = baseline_inference.py
COVERAGE_MIN = 70

# Default company/agent (override with: make create-agent COMPANY="My Corp" AGENT="My Bot")
COMPANY      ?= MyCompany
AGENT        ?= AutoScaler
ALGO         ?= dqn
TRAFFIC      ?= steady
STEPS        ?= 200000

# ── Help ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  ☁️  Cloud Cost Optimizer — Makefile"
	@echo "  ───────────────────────────────────────────────────"
	@echo ""
	@echo "  Setup:"
	@echo "    install          Install core dependencies"
	@echo "    install-dev      Install all dependencies (dev + training + dashboard)"
	@echo ""
	@echo "  Quality:"
	@echo "    test             Run unit tests with coverage"
	@echo "    lint             Check code style (flake8, black, isort)"
	@echo "    format           Auto-format code (black + isort)"
	@echo "    validate         Validate OpenEnv interface compliance (10 checks)"
	@echo ""
	@echo "  Run:"
	@echo "    run-baseline     Run baseline heuristic agents"
	@echo "    run-llm          Run LLM agent via NVIDIA NIM (needs API key in .env)"
	@echo "    run-ingress      Start the production Ingress API server"
	@echo ""
	@echo "  🏢 Company Agent Builder:"
	@echo "    register         Register a company:     make register COMPANY=\"Acme Corp\""
	@echo "    create-agent     Create a named agent:   make create-agent COMPANY=\"Acme\" AGENT=\"Scaler v1\""
	@echo "    train-agent      Train your agent:       make train-agent COMPANY=\"Acme\" AGENT=\"Scaler v1\""
	@echo "    evaluate-agent   Evaluate performance:   make evaluate-agent COMPANY=\"Acme\" AGENT=\"Scaler v1\""
	@echo "    export-agent     Export for deployment:   make export-agent COMPANY=\"Acme\" AGENT=\"Scaler v1\""
	@echo "    deploy-agent     Deploy to production:    make deploy-agent COMPANY=\"Acme\" AGENT=\"Scaler v1\""
	@echo "    list-agents      List all agents:         make list-agents"
	@echo ""
	@echo "  Training (standalone):"
	@echo "    train-dqn        Train DQN agent on steady task"
	@echo "    train-ppo        Train PPO agent on steady task"
	@echo ""
	@echo "  Deploy:"
	@echo "    docker-build     Build the Docker image"
	@echo "    docker-run       Run the containerized dashboard"
	@echo ""
	@echo "  Misc:"
	@echo "    clean            Remove caches, coverage reports, and temp files"
	@echo "    ready            Full pipeline: lint → validate → test → baseline"
	@echo ""

# ── Setup ───────────────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install black isort flake8 mypy pytest-mock

# ── Quality ─────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --cov=env --cov=agent --cov=utils --cov-report=term-missing --cov-fail-under=$(COVERAGE_MIN)

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=frontend,node_modules,.venv,venv
	black . --check --exclude="/(frontend|node_modules|\.venv|venv)/"
	isort . --check-only --skip-glob="frontend/*" --skip-glob="node_modules/*"

format:
	black . --exclude="/(frontend|node_modules|\.venv|venv)/"
	isort . --skip-glob="frontend/*" --skip-glob="node_modules/*"

validate:
	$(PYTHON) $(VALIDATE)

# ── Run ─────────────────────────────────────────────────────────────────────
run-baseline:
	$(PYTHON) agent/evaluate.py

run-llm:
	$(PYTHON) $(BASELINE) --model meta/llama-3.1-70b-instruct

run-ingress:
	$(PYTHON) scripts/ingress_server.py

# ── Training ────────────────────────────────────────────────────────────────
train-dqn:
	$(PYTHON) agent/train.py --task steady --algo dqn --steps 200000

train-ppo:
	$(PYTHON) agent/train.py --task steady --algo ppo --steps 200000

# ── Company Agent Builder ───────────────────────────────────────────────────
register:
	$(PYTHON) build_agent.py register --company "$(COMPANY)"

create-agent:
	$(PYTHON) build_agent.py create --company "$(COMPANY)" --agent "$(AGENT)" --algo $(ALGO) --traffic $(TRAFFIC)

train-agent:
	$(PYTHON) build_agent.py train --company "$(COMPANY)" --agent "$(AGENT)" --steps $(STEPS)

evaluate-agent:
	$(PYTHON) build_agent.py evaluate --company "$(COMPANY)" --agent "$(AGENT)"

export-agent:
	$(PYTHON) build_agent.py export --company "$(COMPANY)" --agent "$(AGENT)"

deploy-agent:
	$(PYTHON) build_agent.py deploy --company "$(COMPANY)" --agent "$(AGENT)"

list-agents:
	$(PYTHON) build_agent.py list

# ── Docker ──────────────────────────────────────────────────────────────────
docker-build:
	docker build -t cloud-cost-optimizer .

docker-run:
	docker run -p 8501:8501 cloud-cost-optimizer

# ── Full Pipeline ───────────────────────────────────────────────────────────
ready: lint validate test run-baseline
	@echo ""
	@echo "  ✅ Project is industry-ready and fully validated!"
	@echo ""

# ── Clean ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov coverage.xml
	rm -rf models/checkpoints models/best models/logs models/tb_logs
	rm -rf exports/
	rm -f sim_result.json llm_baseline_results.json
	@echo "  🧹 Cleaned!"
