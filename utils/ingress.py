import json
import os
import time
from collections import deque
from threading import Lock

from fastapi import BackgroundTasks, FastAPI, Request
from pydantic import BaseModel

app = FastAPI(title="Cloud Cost Optimizer Ingress API")

# Store the last 288 steps (roughly 24 hours of 5-min intervals)
# For real-time, we aggregate hits into windows.
TRAFFIC_DATA_FILE = "production_traffic.json"
WINDOW_SIZE_SECONDS = 10  # 10-second aggregation windows
MAX_HISTORICAL_STEPS = 1000


class TrafficState:
    def __init__(self):
        self.lock = Lock()
        self.current_window_hits = 0
        self.window_start_time = time.time()
        self.history = deque(maxlen=MAX_HISTORICAL_STEPS)
        self.load_history()

    def load_history(self):
        if os.path.exists(TRAFFIC_DATA_FILE):
            try:
                with open(TRAFFIC_DATA_FILE, "r") as f:
                    data = json.load(f)
                    self.history = deque(data, maxlen=MAX_HISTORICAL_STEPS)
            except (json.JSONDecodeError, IOError):
                pass

    def save_history(self):
        with open(TRAFFIC_DATA_FILE, "w") as f:
            json.dump(list(self.history), f)

    def record_hit(self):
        with self.lock:
            now = time.time()
            if now - self.window_start_time >= WINDOW_SIZE_SECONDS:
                # Close current window and start new one
                rps = self.current_window_hits / WINDOW_SIZE_SECONDS
                self.history.append({"timestamp": self.window_start_time, "rps": rps})
                self.current_window_hits = 1
                self.window_start_time = now
                self.save_history()
            else:
                self.current_window_hits += 1

    def record_batch(self, rps: float, timestamp: float = None):
        with self.lock:
            self.history.append({"timestamp": timestamp or time.time(), "rps": rps})
            self.save_history()


state = TrafficState()


class BatchMetric(BaseModel):
    rps: float
    timestamp: float = None
    api_key: str


@app.get("/track")
async def track_hit(background_tasks: BackgroundTasks):
    """
    Endpoint for the JS Tracking Script.
    Every hit to this URL counts as 1 request.
    """
    background_tasks.add_task(state.record_hit)
    return {"status": "ok"}


@app.post("/metrics")
async def ingest_metrics(metric: BatchMetric):
    """
    Better approach: Direct API ingestion from NGINX/Prometheus/Backend.
    Allows high-throughput reporting without individual tracking calls.
    """
    state.record_batch(metric.rps, metric.timestamp)
    return {"status": "received", "current_history_size": len(state.history)}


@app.get("/data")
async def get_traffic_data():
    """Return the history for the simulator to consume."""
    return list(state.history)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
