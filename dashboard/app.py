"""
Cloud Cost Optimizer — Streamlit Dashboard

Real-time simulation dashboard with:
- Interactive traffic and fleet visualization
- Live agent comparison
- Parameter tuning controls
- Performance metrics
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from agent.baselines import (
    PredictiveHeuristicAgent,
    StaticAgent,
    ThresholdAgent,
    run_baseline,
)
from env.environment import (
    CloudCostOptimizerEnv,
    make_chaos_env,
    make_spike_env,
    make_steady_env,
)
from env.models import Action, ActionType, Observation
from tasks.graders import grade_task

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="☁️ Cloud Cost Optimizer",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    .score-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .score-high { background: linear-gradient(135deg, #00b09b, #96c93d); color: white; }
    .score-mid { background: linear-gradient(135deg, #f7971e, #ffd200); color: #1a1a2e; }
    .score-low { background: linear-gradient(135deg, #eb3349, #f45c43); color: white; }
    
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.08);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("## ⚙️ Simulation Controls")

task = st.sidebar.selectbox(
    "📋 Traffic Scenario",
    ["Steady (Easy)", "Spike (Medium)", "Chaos (Hard)", "🚀 Live Production Data"],
    help="Choose the traffic pattern difficulty. Live data requires the Ingress API to be running.",
)

agent_type = st.sidebar.selectbox(
    "🤖 Agent Type",
    ["Threshold (HPA)", "Static", "Predictive Heuristic", "Random"],
    help="Choose the scaling agent",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Agent Parameters")

if agent_type == "Static":
    target_servers = st.sidebar.slider("Target Servers", 1, 20, 5)
elif agent_type == "Threshold (HPA)":
    high_threshold = st.sidebar.slider("Scale-Up Threshold (CPU %)", 50, 95, 75) / 100
    low_threshold = st.sidebar.slider("Scale-Down Threshold (CPU %)", 5, 50, 30) / 100
    cooldown = st.sidebar.slider("Cooldown Steps", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Reward Weights")
w_cost = st.sidebar.slider("Cost Penalty Weight", 0.1, 2.0, 0.5, 0.1)
w_dropped = st.sidebar.slider("Drop Penalty Weight", 1.0, 20.0, 5.0, 0.5)

seed = st.sidebar.number_input("🎲 Random Seed", 1, 9999, 42)

run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)
compare_button = st.sidebar.button("📊 Compare All Agents", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="header-gradient">☁️ Cloud Cost Optimizer</h1>', unsafe_allow_html=True)
st.markdown("*AI-powered auto-scaling simulation for cloud infrastructure*")
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_env(task_str, seed_val):
    if "Steady" in task_str:
        return make_steady_env(seed=seed_val), "steady"
    elif "Spike" in task_str:
        return make_spike_env(seed=seed_val), "spike"
    elif "Live" in task_str:
        from env.real_traffic_generator import RealTimeTrafficGenerator
        from env.traffic_generator import TrafficConfig

        cfg = TrafficConfig(seed=seed_val)
        gen = RealTimeTrafficGenerator(cfg)
        env = CloudCostOptimizerEnv(
            traffic_generator=gen,
            max_steps=100,  # Show last 100 points
            task_name="production",
        )
        return env, "production"
    else:
        return make_chaos_env(seed=seed_val), "chaos"


def get_agent(agent_str):
    if agent_str == "Static":
        servers = target_servers if agent_type == "Static" else 5
        return StaticAgent(target_servers=servers)
    elif agent_str == "Threshold (HPA)":
        ht = high_threshold if agent_type == "Threshold (HPA)" else 0.75
        lt = low_threshold if agent_type == "Threshold (HPA)" else 0.30
        cd = cooldown if agent_type == "Threshold (HPA)" else 3
        return ThresholdAgent(
            high_threshold=ht,
            low_threshold=lt,
            cooldown_steps=cd,
        )
    elif agent_str == "Predictive Heuristic":
        return PredictiveHeuristicAgent()
    else:
        return None  # Random


def run_episode(env, agent, seed_val):
    obs_arr, info = env.reset(seed=seed_val)
    obs = env._get_observation()
    total_reward = 0

    while True:
        if agent:
            action = agent.act(obs)
        else:
            action_idx = np.random.randint(0, 5)
            from env.models import ACTION_INDEX_MAP

            action = Action(action_type=ACTION_INDEX_MAP[action_idx])

        obs_arr, reward, done, _, info = env.step(action)
        obs = env._get_observation()
        total_reward += reward
        if done:
            break

    return env.get_episode_info(), env.get_history(), total_reward


def create_dashboard_plots(history, episode_info, title=""):
    """Create the main dashboard visualization."""
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "📈 Traffic vs Served Requests",
            "🖥️ Active Servers",
            "📊 CPU Load",
            "⏱️ Latency (ms)",
            "❌ Dropped Requests",
            "💰 Cumulative Cost",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    steps = list(range(len(history["traffic"])))

    # Colors
    color_traffic = "#00d2ff"
    color_served = "#3a7bd5"
    color_servers = "#764ba2"
    color_cpu = "#f7971e"
    color_latency = "#eb3349"
    color_dropped = "#ff6b6b"
    color_cost = "#96c93d"

    # Traffic vs Served
    fig.add_trace(
        go.Scatter(x=steps, y=history["traffic"], name="Incoming", line=dict(color=color_traffic, width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=steps, y=history["served"], name="Served", line=dict(color=color_served, width=2, dash="dash")),
        row=1,
        col=1,
    )

    # Active Servers
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=history["servers"],
            name="Servers",
            fill="tozeroy",
            fillcolor="rgba(118,75,162,0.3)",
            line=dict(color=color_servers, width=2),
        ),
        row=1,
        col=2,
    )

    # CPU Load
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=history["cpu_load"],
            name="CPU",
            fill="tozeroy",
            fillcolor="rgba(247,151,30,0.2)",
            line=dict(color=color_cpu, width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.75, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=0.30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # Latency
    fig.add_trace(
        go.Scatter(x=steps, y=history["latency"], name="Latency", line=dict(color=color_latency, width=2)), row=2, col=2
    )
    fig.add_hline(y=200, line_dash="dash", line_color="red", opacity=0.7, annotation_text="SLA: 200ms", row=2, col=2)

    # Dropped
    fig.add_trace(
        go.Bar(x=steps, y=history["dropped"], name="Dropped", marker_color=color_dropped, opacity=0.7), row=3, col=1
    )

    # Cumulative Cost
    cum_cost = np.cumsum(history["cost"])
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=cum_cost,
            name="Cost",
            fill="tozeroy",
            fillcolor="rgba(150,201,61,0.2)",
            line=dict(color=color_cost, width=2),
        ),
        row=3,
        col=2,
    )

    fig.update_layout(
        height=900,
        title=dict(text=title, font=dict(size=20)),
        template="plotly_dark",
        showlegend=False,
        font=dict(family="Inter", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,12,41,0.8)",
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main Simulation
# ─────────────────────────────────────────────────────────────────────────────

if run_button:
    env, task_key = get_env(task, seed)
    agent = get_agent(agent_type)

    with st.spinner("🔄 Running simulation..."):
        episode_info, history, total_reward = run_episode(env, agent, seed)

    # Grade
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(project_root, "tasks", f"task_{task_key}.yaml")
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tasks", f"task_{task_key}.yaml")

    try:
        grade = grade_task(yaml_path, episode_info, history)
        score = grade["total_score"]
    except Exception as e:
        score = 0.0
        grade = {"sla_score": 0, "cost_score": 0, "latency_score": 0, "stability_score": 0}

    # ── Metrics Row ──
    col1, col2, col3, col4, col5 = st.columns(5)

    score_class = "score-high" if score >= 0.7 else "score-mid" if score >= 0.4 else "score-low"

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{score:.2f}</div>
            <div class="metric-label">Overall Score</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("💰 Total Cost", f"${episode_info.total_cost:.2f}")

    with col3:
        drop_pct = episode_info.drop_rate * 100
        st.metric("❌ Drop Rate", f"{drop_pct:.2f}%")

    with col4:
        st.metric("⏱️ Avg Latency", f"{episode_info.avg_latency:.1f} ms")

    with col5:
        st.metric("🖥️ Peak Servers", f"{episode_info.peak_servers}")

    st.markdown("---")

    # ── Score Breakdown ──
    st.markdown("### 📊 Score Breakdown")
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        st.metric("SLA Compliance", f"{grade.get('sla_score', 0):.3f}")
    with bcol2:
        st.metric("Cost Efficiency", f"{grade.get('cost_score', 0):.3f}")
    with bcol3:
        st.metric("Latency Quality", f"{grade.get('latency_score', 0):.3f}")
    with bcol4:
        st.metric("Scaling Stability", f"{grade.get('stability_score', 0):.3f}")

    # ── Charts ──
    st.markdown("---")
    fig = create_dashboard_plots(
        history, episode_info, title=f"Simulation: {agent_type} on {task} (Score: {score:.3f})"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Episode Statistics ──
    with st.expander("📋 Detailed Episode Statistics"):
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.json(
                {
                    "Total Steps": episode_info.total_steps,
                    "Total Requests": f"{episode_info.total_requests:.0f}",
                    "Total Served": f"{episode_info.total_served:.0f}",
                    "Total Dropped": f"{episode_info.total_dropped:.0f}",
                    "Drop Rate": f"{episode_info.drop_rate:.4%}",
                    "SLA Compliance": f"{episode_info.sla_compliance_rate:.2%}",
                }
            )
        with stat_col2:
            st.json(
                {
                    "Total Cost": f"${episode_info.total_cost:.2f}",
                    "Avg Latency (ms)": f"{episode_info.avg_latency:.2f}",
                    "Max Latency (ms)": f"{episode_info.max_latency:.2f}",
                    "Avg CPU Load": f"{episode_info.avg_cpu_load:.2%}",
                    "Avg Servers": f"{episode_info.avg_active_servers:.2f}",
                    "Peak Servers": episode_info.peak_servers,
                    "Cost per Request": f"${episode_info.cost_per_request:.4f}",
                }
            )

# ─────────────────────────────────────────────────────────────────────────────
# Agent Comparison
# ─────────────────────────────────────────────────────────────────────────────

if compare_button:
    env, task_key = get_env(task, seed)

    agents = [
        ("Static(3)", StaticAgent(target_servers=3)),
        ("Static(5)", StaticAgent(target_servers=5)),
        ("Static(10)", StaticAgent(target_servers=10)),
        ("Threshold", ThresholdAgent()),
        ("Predictive", PredictiveHeuristicAgent()),
    ]

    results = []

    progress = st.progress(0, "Running agent comparison...")

    for i, (name, agent) in enumerate(agents):
        env_fresh, _ = get_env(task, seed)
        info, history, reward = run_episode(env_fresh, agent, seed)

        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tasks", f"task_{task_key}.yaml")
        try:
            grade = grade_task(yaml_path, info, history)
            score = grade["total_score"]
        except:
            score = 0.0
            grade = {"sla_score": 0, "cost_score": 0, "latency_score": 0, "stability_score": 0}

        results.append(
            {
                "Agent": name,
                "Score": score,
                "SLA": grade.get("sla_score", 0),
                "Cost": grade.get("cost_score", 0),
                "Latency": grade.get("latency_score", 0),
                "Stability": grade.get("stability_score", 0),
                "Total Cost ($)": round(info.total_cost, 2),
                "Drop Rate (%)": round(info.drop_rate * 100, 3),
                "Avg Latency (ms)": round(info.avg_latency, 1),
                "Peak Servers": info.peak_servers,
            }
        )

        progress.progress((i + 1) / len(agents), f"Evaluated {name}")

    progress.empty()

    # Results Table
    st.markdown("### 🏆 Agent Comparison Results")
    df = pd.DataFrame(results)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)

    st.dataframe(
        df.style.background_gradient(cmap="RdYlGn", subset=["Score", "SLA", "Cost", "Latency", "Stability"]),
        use_container_width=True,
        height=250,
    )

    # Comparison Charts
    fig_comp = go.Figure()
    categories = ["SLA", "Cost", "Latency", "Stability"]

    colors = ["#00d2ff", "#764ba2", "#f7971e", "#96c93d", "#eb3349"]

    for i, row in df.iterrows():
        fig_comp.add_trace(
            go.Scatterpolar(
                r=[row["SLA"], row["Cost"], row["Latency"], row["Stability"]],
                theta=categories,
                fill="toself",
                name=row["Agent"],
                line=dict(color=colors[i % len(colors)], width=2),
                opacity=0.7,
            )
        )

    fig_comp.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor="rgba(15,12,41,0.8)",
        ),
        template="plotly_dark",
        title="Agent Performance Radar",
        height=500,
        font=dict(family="Inter"),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig_comp, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Info Section (shown by default)
# ─────────────────────────────────────────────────────────────────────────────

if not run_button and not compare_button:
    st.markdown("### 🎯 How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 📡 Traffic Simulation
        Realistic workload patterns with:
        - Daily seasonality
        - Flash-sale spikes
        - Random micro-bursts
        - Organic growth trends
        """)

    with col2:
        st.markdown("""
        #### 🤖 Auto-Scaling Agent
        Decides at each step:
        - **Scale Up** (+1 or +3 servers)
        - **Hold** (maintain fleet)
        - **Scale Down** (-1 or -3 servers)
        
        Servers take 3 steps to warm up!
        """)

    with col3:
        st.markdown("""
        #### 📊 Grading System
        Multi-objective scoring (0-1):
        - **SLA Compliance** (40%)
        - **Cost Efficiency** (30%)
        - **Latency Quality** (20%)
        - **Scaling Stability** (10%)
        """)

    st.markdown("---")
    st.markdown("👈 **Use the sidebar controls to configure and run a simulation!**")
