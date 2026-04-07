"""
Cloud Cost Optimizer - Production Ingress API

This is the 'Industrial Bridge' for real companies. It allows real systems
to send live telemetry and receive scaling decisions from the AI.

Features:
1. POST /telemetry: Receive real-world metrics from production servers.
2. GET /decision: Provide the current scaling decision to the infrastructure provider (e.g. Terraform or Kubernetes).
3. GET /metrics: Standard Prometheus-compatible metrics endpoint.
4. GET /health: Health check for load balancers and k8s probes.

Security:
- API key authentication via X-API-Key header or INGRESS_API_KEY env var.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

# Attempt to load environment variables from .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("IngressAPI")

app = FastAPI(
    title="Cloud Cost Optimizer Ingress API",
    version="1.0.0",
    description="Production bridge for real-time cloud telemetry and AI-driven scaling decisions.",
)

# ── Configuration ────────────────────────────────────────────────────────────
INGRESS_API_KEY = os.environ.get("INGRESS_API_KEY", None)
START_TIME = time.time()

# In-memory 'Last State' (In production, use Redis)
current_telemetry = {
    "timestamp": time.time(),
    "cpu_load": 0.0,
    "request_count": 0,
    "latency_ms": 0.0,
    "active_servers": 1,
}

current_decision = {
    "action": "NO_OP",
    "reasoning": "Waiting for live data batch...",
    "timestamp": time.time(),
}

# ── Metrics Counters ─────────────────────────────────────────────────────────
_metrics = {
    "telemetry_received": 0,
    "decisions_posted": 0,
    "decisions_served": 0,
    "errors": 0,
}


# ── Auth Dependency ──────────────────────────────────────────────────────────
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Optional API key validation. If INGRESS_API_KEY is set, enforce it."""
    if INGRESS_API_KEY is not None:
        if x_api_key != INGRESS_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


# ── Models ───────────────────────────────────────────────────────────────────
class Telemetry(BaseModel):
    cpu_load: float
    request_count: int
    latency_ms: float
    active_servers: int


class Decision(BaseModel):
    action: str
    reasoning: str


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "online", "mode": "production_bridge", "version": "1.0.0"}


@app.post("/telemetry", dependencies=[Depends(verify_api_key)])
async def report_telemetry(data: Telemetry):
    """
    Real-world servers call this endpoint to report their state.
    """
    global current_telemetry
    current_telemetry = data.model_dump()
    current_telemetry["timestamp"] = time.time()
    _metrics["telemetry_received"] += 1

    logger.info(f"📡 Telemetry Received: CPU={data.cpu_load:.2f}, RPS={data.request_count}")
    return {"status": "received"}


@app.get("/telemetry/latest")
def get_latest_telemetry():
    """
    The RL Simulator pulls from here instead of generating math data.
    """
    return current_telemetry


@app.post("/decision", dependencies=[Depends(verify_api_key)])
def post_decision(decision: Decision):
    """
    The AI Agent calls this to post its scaling decision.
    """
    global current_decision
    current_decision = decision.model_dump()
    current_decision["timestamp"] = time.time()
    _metrics["decisions_posted"] += 1

    logger.info(f"🤖 AI Decision Posted: {decision.action} | {decision.reasoning}")
    return {"status": "decision_updated"}


@app.get("/decision")
def get_decision():
    """
    Infrastructure providers (Jenkins, Terraform, K8s Controllers)
    poll this to see if they should scale up/down.
    """
    _metrics["decisions_served"] += 1
    return current_decision


@app.get("/health")
def health():
    """Kubernetes liveness / readiness probe."""
    uptime_seconds = time.time() - START_TIME
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime_seconds, 1),
        "telemetry_count": _metrics["telemetry_received"],
        "decisions_count": _metrics["decisions_posted"],
    }


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus-compatible metrics scrape endpoint."""
    lines = [
        f"# HELP ingress_telemetry_received_total Total telemetry payloads received",
        f"# TYPE ingress_telemetry_received_total counter",
        f'ingress_telemetry_received_total {_metrics["telemetry_received"]}',
        f"# HELP ingress_decisions_posted_total Total AI decisions posted",
        f"# TYPE ingress_decisions_posted_total counter",
        f'ingress_decisions_posted_total {_metrics["decisions_posted"]}',
        f"# HELP ingress_decisions_served_total Total decisions served to infra",
        f"# TYPE ingress_decisions_served_total counter",
        f'ingress_decisions_served_total {_metrics["decisions_served"]}',
        f"# HELP ingress_uptime_seconds Server uptime in seconds",
        f"# TYPE ingress_uptime_seconds gauge",
        f"ingress_uptime_seconds {time.time() - START_TIME:.1f}",
    ]
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse("\n".join(lines), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("INGRESS_HOST", "0.0.0.0")
    port = int(os.environ.get("INGRESS_PORT", "8000"))
    logger.info(f"🚀 Starting Ingress API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
