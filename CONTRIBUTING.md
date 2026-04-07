# Contributing to Cloud Cost Optimizer

Thank you for considering contributing! Here's how to get started.

## 🚀 Quick Setup

```bash
# 1. Clone the repo
git clone https://github.com/himanshugahalyan06/Cloud_Cost_Optimizer.git
cd Cloud_Cost_Optimizer

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install all dependencies
make install-dev

# 4. Copy environment template and add your API key
cp .env.example .env
# Edit .env with your NVIDIA NIM API key

# 5. Validate everything works
make ready
```

## 🧪 Running Tests

```bash
make test          # Run with coverage
make validate      # OpenEnv interface validation (10 checks)
make run-baseline  # Run all baseline agents
```

## 📐 Code Style

We use **Black** for formatting and **isort** for import sorting:

```bash
make format   # Auto-format code
make lint     # Check style (CI will enforce this)
```

## 📝 Pull Request Checklist

- [ ] All tests pass (`make test`)
- [ ] OpenEnv validation passes (`make validate`)
- [ ] Code is formatted (`make lint`)
- [ ] New features have tests
- [ ] README updated if needed

## 🏗️ Architecture

See the [README.md](README.md) for the full architecture diagram and component breakdown.

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.
