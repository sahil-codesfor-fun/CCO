import random
import time

import requests

# Replaces the tracker script for more accurate server-level reporting.
# Use this when you have Prometheus or NGINX data available.
INGRESS_API = "http://localhost:8000/metrics"


def send_real_rps(rps_value: float):
    """Report actual Requests Per Second from your monitoring system."""
    payload = {"rps": rps_value, "api_key": "YOUR_SECRET_COMPANY_KEY", "timestamp": time.time()}

    try:
        response = requests.post(INGRESS_API, json=payload, timeout=2)
        if response.status_code == 200:
            print(f"[SUCCESS] Reported RPS: {rps_value:.2f}")
        else:
            print(f"[ERROR] Failed to report. Status: {response.status_code}")
    except Exception as e:
        print(f"[FAIL] Could not connect to Cloud Cost Optimizer Ingress: {e}")


if __name__ == "__main__":
    print("--- LIVE PRODUCTION METRICS SIMULATOR ---")
    print("This script simulates a company backend pushing metrics directly to us.")

    # Simulating a real business day's traffic ramping up
    while True:
        # Generate some hypothetical production RPS (between 50 and 500)
        mock_rps = 100 + (150 * (time.time() % 3600 / 3600)) + random.gauss(0, 10)

        send_real_rps(mock_rps)

        # Report every 10 seconds (standard for many metric systems)
        time.sleep(10)
