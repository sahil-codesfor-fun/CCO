import json
import os
import random
from typing import List, Optional

import requests

from .traffic_generator import TrafficConfig, TrafficGenerator


class RealTimeTrafficGenerator(TrafficGenerator):
    """
    Consumes live traffic data from the Ingress API or a local file.

    This replaces simulated math.sin() with actual RPS (Requests Per Second)
    recorded from a real website or system.
    """

    def __init__(
        self, config: TrafficConfig, api_url: str = "http://localhost:8000/data", use_json_fallback: bool = True
    ):
        super().__init__(config)
        self.api_url = api_url
        self.use_json_fallback = use_json_fallback
        self.local_json = "production_traffic.json"
        self.history_data = []
        self.refresh_data()

    def refresh_data(self):
        """Fetch the latest history from the Ingress API."""
        try:
            response = requests.get(self.api_url, timeout=2)
            if response.status_code == 200:
                self.history_data = response.json()
                print(f"Refreshed {len(self.history_data)} points from Ingress API.")
        except Exception as e:
            print(f"Warning: Could not connect to API {self.api_url}: {e}")
            if self.use_json_fallback and os.path.exists(self.local_json):
                with open(self.local_json, "r") as f:
                    self.history_data = json.load(f)
                    print(f"Loaded {len(self.history_data)} points from local JSON fallback.")

    def generate(self, t: int) -> float:
        """
        Get traffic for time step t.
        If t exceeds available real data, it can either wrap around
        or add simulated variance to the last known point.
        """
        if not self.history_data:
            # Fallback to base simulation if no real data
            return super().generate(t)

        # If t is within history, return it
        if t < len(self.history_data):
            return self.history_data[t]["rps"]
        else:
            # Loop the history if we run out (for learning patterns)
            wrapped_t = t % len(self.history_data)
            base_rps = self.history_data[wrapped_t]["rps"]
            # Add some minor noise so it's not identical every loop
            noise = random.gauss(0, self.history_data[wrapped_t]["rps"] * 0.05)
            return max(5.0, base_rps + noise)

    def reset(self, seed: Optional[int] = None):
        super().reset(seed)
        self.refresh_data()
