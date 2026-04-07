# ☁️ Cloud Cost Optimizer

> **AI-powered auto-scaling simulation for cloud infrastructure cost optimization**

An OpenEnv-compatible Reinforcement Learning environment that simulates real-world cloud infrastructure auto-scaling. An AI agent manages a fleet of servers, making real-time scaling decisions to balance **SLA compliance** (zero dropped requests) against **operational cost** — mirroring production systems like **Kubernetes HPA**, **AWS Auto Scaling**, and **Netflix Titus**.

---

## 🎯 What Problem Does This Project Solve?

Imagine you're the **CTO of Amazon** on **Black Friday**. Your website traffic looks like this over 24 hours:

```
     Traffic (requests/sec)
500 │            ╱╲  ← Flash Sale Spike!
400 │           ╱  ╲
300 │     ╱╲   ╱    ╲       ╱╲
200 │    ╱  ╲ ╱      ╲     ╱  ╲
100 │───╱    ╲        ╲───╱    ╲───
  0 └──────────────────────────────→ Time
    12am  6am  12pm  6pm  12am
```

You're paying for **cloud servers** (AWS/GCP/Azure) by the hour. You face a brutal tradeoff:

| Decision | Consequence |
|----------|-------------|
| 🟥 **Overprovision** (too many servers) | You waste **$100,000s/month** in idle server costs |
| 🟥 **Underprovision** (too few servers) | Users see **"503 Service Unavailable"** errors, you lose customers |
| 🟩 **Perfect scaling** | Just the right amount of servers at every moment — minimum cost, zero dropped requests |

**This project builds an AI agent that solves this problem automatically** — by learning to make perfect scaling decisions through reinforcement learning.

---

## 🧠 How Does It Work? (Step by Step)

The system has **5 core components** working together in a loop:

### Step 1: 📡 Traffic Generator Creates Realistic Load

The simulator generates traffic that mimics real-world patterns:

```
Total Traffic = Seasonality + Spikes + Micro-bursts + Noise + Growth Trend
```

| Component | What It Models | Example |
|-----------|---------------|---------|
| **Sinusoidal Seasonality** | Day/night cycles — traffic peaks at noon, drops at 3am | `30 × sin(2π × t/288)` |
| **Flash-Sale Spikes** | Sudden traffic surges (Black Friday, viral tweet) | Bell-curve shaped, 200-500 req/s |
| **Random Micro-bursts** | Unpredictable short spikes | 5% chance each step |
| **Gaussian Noise** | Natural variance in real traffic | Random fluctuation |
| **Linear Trend** | Organic user growth over time | `0.1 × t` |

### Step 2: 🌐 Server Fleet Processes Requests

Each server in the fleet:
- **Handles 50 requests/second** (capacity)
- **Costs $0.10 per time step** to run
- **Takes 3 steps to warm up** when newly added (can't serve traffic during warm-up!)
- **Latency grows exponentially** near saturation: `latency = 10ms × e^(cpu_load × 5)`

```
  CPU Load vs Latency:

  20% CPU → 27ms  ✅ Fast
  50% CPU → 122ms ✅ Acceptable
  75% CPU → 424ms ⚠️ Slow (SLA violation!)
  90% CPU → 903ms 🔴 Very slow
  100% CPU → Requests get DROPPED 💀
```

### Step 3: 🤖 AI Agent Makes Scaling Decisions

Every timestep, the agent **observes 11 metrics** about the fleet:

```python
Observation = {
    timestep: 42,               # Current time
    incoming_requests: 320.5,   # Traffic right now
    active_servers: 5,          # Running servers
    warming_up_servers: 2,      # Servers booting up
    cpu_load: 0.78,             # How loaded the fleet is (78%)
    latency_ms: 156.3,          # Response time
    dropped_requests: 12.0,     # Failed requests
    served_requests: 250.0,     # Successful requests
    cost_so_far: 23.50,         # Money spent so far
    traffic_trend: +15.2,       # Is traffic going UP or DOWN?
    time_of_day: 0.58,          # What time is it? (afternoon)
}
```

Then it chooses **1 of 5 actions**:

| Action | Effect | When to Use |
|--------|--------|-------------|
| `SCALE_UP_3` | Add 3 servers | Spike incoming — emergency ramp up |
| `SCALE_UP_1` | Add 1 server | Traffic gradually rising |
| `NO_OP` | Do nothing | Fleet is right-sized |
| `SCALE_DOWN_1` | Remove 1 server | Traffic dropping, save money |
| `SCALE_DOWN_3` | Remove 3 servers | Traffic plummeted, cut costs fast |

### Step 4: 📊 Reward Function Scores the Decision

After each action, the environment computes a **reward signal**:

```
Reward = +0.01 × ServedRequests     ← "Good job serving traffic"
       − 0.50 × ServerCost         ← "Each server costs money"
       − 5.00 × DroppedRequests    ← "CATASTROPHIC — never drop requests!"
       − 0.002 × LatencyPenalty    ← "Keep response times low 
       + 0.10 × EfficiencyBonus    ← "Bonus for optimal CPU 40-75%"
```

Notice: **dropping requests is penalized 100x more than idle servers.** The agent learns that SLA violations are catastrophic — just like in the real world where a downed website costs millions in lost revenue.

### Step 5: 🏆 Grader Produces Final Score (0.0 – 1.0)

After the entire episode, the grader evaluates the agent:

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **SLA Compliance** | 40% | Did you keep latency under 200ms? Any dropped requests? |
| **Cost Efficiency** | 30% | How much money did you spend vs. the theoretical best? |
| **Latency Quality** | 20% | Average and peak latency — lower is better |
| **Scaling Stability** | 10% | Did you oscillate wildly (scale up/down/up/down)? |

---

## 🎮 The 3 Difficulty Levels

| Level | Traffic Pattern | Challenge |
|-------|----------------|-----------|
| 🟢 **Steady (Easy)** | Smooth daily cycle, no spikes | Can you right-size and minimize cost? |
| 🟡 **Spike (Medium)** | 3 flash-sale spikes | Can you scale up BEFORE the spike, not after? |
| 🔴 **Chaos (Hard)** | 6 random spikes + high noise + bursts + growth | Genuinely challenges frontier AI models |

---

## 🤖 Agent Strategies Compared

The project includes **5 pre-built agents**:

| Agent | Strategy | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| **Static(3)** | Always maintain 3 servers | Zero scaling overhead | Can't handle spikes at all |
| **Static(5)** | Always maintain 5 servers | Handles moderate load | Wastes money during low traffic |
| **Threshold (HPA)** | Scale at CPU > 75%, shrink at < 30% | Industry standard — reactive | Always **late** to spikes |
| **Predictive** | Looks at traffic trend + anticipates | Proactive — scales before spikes | Over-provisions sometimes |
| **Random** | Random actions | None | Terrible at everything |

---

## 🏗️ Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLOUD COST OPTIMIZER                         │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│              │              │              │                   │
│  📡 Traffic   │  🖥️ Server    │  🤖 Agent     │  📊 Reward        │
│  Generator   │  Fleet Sim   │  (Decision)  │  Engine          │
│              │              │              │                   │
│  Seasonality │  Capacity    │  Observe     │  Served: +reward  │
│  + Spikes    │  + Warm-up   │  → Think     │  Cost: -penalty   │
│  + Bursts    │  + Latency   │  → Act       │  Drops: -5x       │
│  + Noise     │  + Dropping  │  → Learn     │  Latency: -pen    │
│  + Trend     │  + Cost      │              │  Efficiency: +    │
│              │              │              │                   │
├──────────────┴──────────────┴──────────────┴───────────────────┤
│                                                                │
│  🏆 Grader: SLA (40%) + Cost (30%) + Latency (20%) +           │
│              Stability (10%) = Score 0.0 – 1.0                 │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  🎨 React Frontend (Vite + Recharts)                           │
│  Runs simulation entirely in-browser —("no backend needed")    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
📡 Traffic Generator → 🌐 Server Fleet Sim → 🤖 AI Agent → 📊 Reward Engine → 🏆 Grader
```

1. A **Traffic Generator** produces realistic workload patterns (daily cycles, flash sales, random bursts).
2. The **Environment** simulates a fleet of servers with warm-up delays and exponential latency.
3. An **Agent** observes 11 telemetry dimensions and outputs scaling decisions.
4. A **Reward Function** provides dense feedback balancing cost vs. performance.
5. A **Grader** scores the agent on a 0.0–1.0 scale across 4 metrics.

---

## 💡 Real-World Equivalent

| This Project | Real Production System |
|-------------|----------------------|
| Environment | AWS EC2 / GCP Compute / Azure VM fleet |
| Agent | Kubernetes HPA / AWS Auto Scaling Group |
| Traffic Generator | Real user traffic from CDN logs |
| Reward Function | CloudWatch / Prometheus + Monthly AWS bill |
| Grader | SLA dashboard + Finance report |
| Dropped requests | "503 Service Unavailable" errors |
| Server warm-up | EC2 instance boot time (~30-60 seconds) |

In short: **this project is a training ground for AI to learn cloud operations** — the same systems that keep Netflix, Uber, and Amazon running 24/7. 🎯

---

## 🔬 Environment Design Details

### Traffic Modeling
Combines 5 signal components for realism:
1. **Sinusoidal seasonality** — day/night patterns
2. **Scheduled spike events** — bell-curve shaped flash sales
3. **Random micro-bursts** — viral content moments
4. **Gaussian noise** — natural variance
5. **Linear trend** — organic growth

### Server Model
- **Warm-up delay**: New servers take 3 steps to activate
- **Exponential latency**: `latency = base × exp(cpu_load × k)`
- **Hard dropping**: Requests beyond capacity are dropped
- **Cost model**: Active servers cost per step; warming servers cost half

### Reward Engineering
- **Dense signal**: Varies every step (not sparse)
- **Multi-objective**: Balances 5 competing objectives
- **Interpretable**: Structured breakdown for debugging

---

## 🔧 OpenEnv Interface

### Observation Space (11 dimensions)
```python
class Observation(BaseModel):
    timestep: int                 # Current time step
    incoming_requests: float      # Traffic volume (req/s)
    active_servers: int           # Provisioned server count
    warming_up_servers: int       # Servers in warm-up phase
    cpu_load: float               # Fleet CPU utilization (0-1)
    latency_ms: float             # Average response latency
    dropped_requests: float       # Requests dropped this step
    served_requests: float        # Requests served this step
    cost_so_far: float            # Cumulative cost
    traffic_trend: float          # Traffic rate of change
    time_of_day: float            # Normalized time (0=midnight)
```

### Action Space (5 discrete actions)
| Action | Delta | Effect |
|--------|-------|--------|
| `SCALE_UP_3` | +3 | Aggressive scaling for spike events |
| `SCALE_UP_1` | +1 | Cautious proactive scaling |
| `NO_OP` | 0 | Hold current fleet size |
| `SCALE_DOWN_1` | -1 | Conservative cost optimization |
| `SCALE_DOWN_3` | -3 | Aggressive cost reduction |

### Reward Function
```
R(t) = w₁·ServedRequests - w₂·ServerCost - w₃·DroppedRequests - w₄·LatencyPenalty + w₅·EfficiencyBonus

where:
  w₁ = 0.01   (served reward, normalized)
  w₂ = 0.50   (cost penalty — continuous bleed)
  w₃ = 5.00   (drop penalty — CATASTROPHIC)
  w₄ = 0.002  (latency penalty — soft, above 100ms)
  w₅ = 0.10   (efficiency bonus — CPU in 40-75% sweet spot)
```

Key design: **Dropped requests incur 100x the penalty of idle servers** → the agent learns that SLA violations are catastrophic.

---

## 📈 Baseline Results

| Agent | Steady Score | Spike Score | Chaos Score |
|-------|-------------|-------------|-------------|
| Static(3) | ~0.45 | ~0.30 | ~0.25 |
| Static(5) | ~0.55 | ~0.40 | ~0.35 |
| Threshold(HPA) | ~0.65 | ~0.55 | ~0.45 |
| Predictive | ~0.70 | ~0.60 | ~0.50 |
| DQN (trained) | ~0.80 | ~0.70 | ~0.60 |

*Scores vary by seed. Run `agent/evaluate.py` for exact numbers.*

---

## 🚀 Quick Start

### Installation
```bash
# Backend (Python environment)
pip install -r requirements.txt

# Frontend (React dashboard)
cd frontend && npm install
```

### Run the React Frontend
```bash
cd frontend && npm run dev 
# Opens at → http://localhost:5173
```

### Validate OpenEnv Interface
```bash
python validate.py
```

### Run Tests
```bash
python -m pytest tests/ -v
```

### Evaluate Baseline Agents
```bash
python agent/evaluate.py
```

### Run LLM Baseline (requires OpenAI API key)

```bash
# For OpenAI:
export API_KEY="sk-..."

# For NVIDIA NIM (OpenAI compatible):
export API_KEY="nvapi-..."
export API_BASE_URL="https://integrate.api.nvidia.com/v1"

python baseline_inference.py --model meta/llama-3.1-70b-instruct
```

### Train DQN Agent
```bash
python agent/train.py --task steady --algo dqn --steps 200000
```

### Launch Streamlit Dashboard (alternative)
```bash
streamlit run dashboard/app.py
```

### Docker
```bash
docker build -t cloud-cost-optimizer .
docker run -p 8501:8501 cloud-cost-optimizer
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|-----------|---------|
| 🐍 **Python** | Core environment & RL training |
| 🏋️ **Gymnasium** | Standard RL interface |
| 🧠 **Stable-Baselines3** | DQN & PPO training algorithms |
| 📋 **Pydantic** | Typed models & validation |
| ⚛️ **React + Vite** | Stunning interactive frontend |
| 📊 **Recharts** | Data visualization (6 chart types) |
| 🎨 **Framer Motion** | Smooth animations |
| 🔑 **OpenAI API** | LLM baseline agent |
| 🐳 **Docker** | Containerized deployment |

---

## 📁 Project Structure

```
cloud-cost-optimizer/
├── frontend/                     ← 🎨 React + Vite + Recharts (stunning UI)
│   └── src/
│       ├── App.jsx               ← 5 sections: Hero, Simulator, Compare, Architecture, About
│       ├── engine.js             ← Full JS simulation engine (runs in browser!)
│       └── index.css             ← Premium dark glassmorphism theme
│
├── env/                          ← 🧠 Core Python Environment
│   ├── environment.py            ← Gymnasium-compatible RL env (OpenEnv interface)
│   ├── models.py                 ← Typed Pydantic models (Observation, Action, Reward)
│   ├── traffic_generator.py      ← Realistic workload patterns
│   └── server_model.py           ← Server fleet with warm-up & latency
│
├── agent/                        ← 🤖 AI Agents
│   ├── baselines.py              ← Static, Threshold, Predictive heuristics
│   ├── train.py                  ← DQN/PPO training with Stable-Baselines3
│   └── evaluate.py               ← Full evaluation pipeline
│
├── tasks/                        ← 📋 Tasks + Deterministic Grading
│   ├── task_steady.yaml          ← Easy task config
│   ├── task_spike.yaml           ← Medium task config
│   ├── task_chaos.yaml           ← Hard task config
│   └── graders.py                ← Deterministic scoring (0.0-1.0)
│
├── dashboard/                    ← 📊 Streamlit Alternative Dashboard
│   └── app.py                    ← Interactive simulation dashboard
│
├── configs/
│   └── config.yaml               ← Central configuration
│
├── tests/
│   └── test_environment.py       ← 35+ test cases
│
├── baseline_inference.py         ← 🔑 OpenAI API-based LLM agent
├── validate.py                   ← ✅ OpenEnv interface validator (10 checks)
├── openenv.yaml                  ← OpenEnv metadata spec
├── Dockerfile                    ← 🐳 Docker deployment
├── requirements.txt              ← Python dependencies
└── README.md                     ← 📘 This file
```

---

## 🎨 Frontend Features

The React frontend has **5 premium sections** with a dark glassmorphism theme:

| Section | What It Does |
|---------|-------------|
| **🏠 Hero** | Animated landing with floating gradient orbs, stats bar, and CTAs |
| **🚀 Live Simulator** | Select traffic scenario + agent → watch 6 real-time charts animate |
| **📊 Agent Comparison** | Pit all 5 agents against each other — leaderboard table + radar chart |
| **🏗️ Architecture** | Pipeline diagram, observation/action space specs, reward formula |
| **📘 About** | Full explanation, real-world mapping table, difficulty tiers, tech stack grid |

**Design highlights:**
- 🌑 Dark theme with animated gradient orbs & glassmorphism cards
- 📊 6 interactive Recharts charts with custom tooltips & animated data loading
- 🎯 Radar chart for multi-axis agent comparison
- 🏆 Leaderboard with medal emojis & score badges
- 💎 Inter + JetBrains Mono typography, premium hover effects
- 📱 Fully responsive layout
- 🌐 **Runs entirely in-browser** — no backend server needed!

---

## 📊 Grading System (0.0 – 1.0)

Each task is graded on 4 dimensions:

| Metric | Weight | What It Measures |
|--------|--------|-------------------|
| **SLA Compliance** | 40% | Drop rate + latency compliance |
| **Cost Efficiency** | 25-30% | Cost vs theoretical best/worst |
| **Latency Quality** | 20% | Average and P99 latency |
| **Scaling Stability** | 10-15% | Oscillation and smoothness |

Graders are **deterministic and reproducible** given the same seed.

---

## 🔮 Future Work

- **LSTM Traffic Prediction** — Predictive scaling with learned forecasting
- **Multi-Region Scaling** — Simulate multiple data centers with cross-region routing
- **Spot vs On-Demand Pricing** — Different cost models for cost arbitrage
- **Kubernetes HPA Simulation** — Full HPA behavior with custom metrics
- **PPO + Curriculum Learning** — Train on progressive difficulty

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🏗️ Production-Grade Readiness

This project is built to **industry-standard specifications**, ensuring it's not just a prototype but a robust tool for cloud infrastructure simulation.

### ✅ Industry-Standard Features
- **Makefile Implementation**: Standardized workflow for installation, testing, linting, and running simulations.
- **CI/CD Integration**: Pre-configured GitHub Actions (`.github/workflows/ci.yml`) for automated linting, interface validation, and unit testing across multiple Python versions.
- **Structured Logging**: Replaced generic prints with Python's standard `logging` library for traceable execution and debugging.
- **Environment Management**: Fully supports `.env` for secrets (like API keys) and `config.yaml` for system parameters.
- **Interface Validation**: A 10-point `validate.py` script ensures every component adheres to the **OpenEnv Specification**.
- **Containerization**: Production-ready `Dockerfile` for easy deployment as a microservice.
- **Determinism**: Every simulation is seed-controlled, ensuring the same decisions always produce the same results—critical for comparative AI research.

---

## 🏗️ Real-World Integration (The "Production Bridge")

If a company wants to use this for **live infrastructure**, they use the **Ingress Bridge API**. This allows real systems to feed the AI real-time data.

### 📡 Step 1: Start the Ingress Server
This creates a REST API endpoint for your production servers to report their status.
```bash
make run-ingress
```

### 📤 Step 2: Feed Real Metrics (from your Production Servers)
Your real-world servers (Node.js, Go, Python) can call the API to update the AI on current load:

**Method: POST /telemetry**
```bash
curl -X POST http://localhost:8000/telemetry \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_load": 0.85,
    "request_count": 420,
    "latency_ms": 125.5,
    "active_servers": 5
  }'
```

### 📥 Step 3: Receive Scaling Decisions (Infrastructure Layer)
Your **Kubernetes Controller** or **Terraform Script** polls the AI for the next action:

**Method: GET /decision**
```bash
curl http://localhost:8000/decision
```
**Response:**
```json
{
  "action": "SCALE_UP_3",
  "reasoning": "CPU critical (85%) and traffic surging.",
  "timestamp": 1712000456
}
```

---

<p align="center">
  <strong>Built for the OpenEnv Hackathon 2026</strong><br>
  <em>Industry-Ready Cloud Engineering & Reinforcement Learning</em>
</p>
