"""
Single-file load-test + plotting tool
------------------------------------
• Sends traffic to the public Gradio demo (adjust HOST if needed).
• 100 virtual users, ramp-up 10 users/s, test length 120 s.
• Collects aggregate stats once per second and plots:
      – Average response-time trend
      – 95th-percentile response-time trend
      – Requests/second trend
      – Histogram of all response times
"""

import time
from collections import defaultdict

import gevent
from locust import HttpUser, task, between
from locust.env import Environment
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────
#  ❶  Define the virtual user
# ──────────────────────────────────────────────────────────────────────────
class GradioUser(HttpUser):
    wait_time = between(1, 3)                 # think-time between calls
    host = "https://48b5bd57cf18517e5f.gradio.live"

    @task
    def get_root(self):
        self.client.get("/")

    @task(3)
    def submit_prediction(self):
        payload = {"data": ["ping"], "fn_index": 0}
        self.client.post("/api/predict/", json=payload, name="/api/predict")


# ──────────────────────────────────────────────────────────────────────────
#  ❷  Run Locust headlessly through its Python API
# ──────────────────────────────────────────────────────────────────────────
def run_locust(users=100, spawn_rate=10, duration=120):
    env = Environment(user_classes=[GradioUser])
    env.create_local_runner()

    # Kick off the swarm
    env.runner.start(user_count=users, spawn_rate=spawn_rate)

    # Collect one-second snapshots
    timeline, avg_rt, p95_rt, rps = [], [], [], []
    start = time.time()

    while time.time() - start < duration:
        total = env.stats.total  # aggregate over all endpoints

        timeline.append(time.time() - start)
        avg_rt.append(total.avg_response_time or 0)
        p95_rt.append(total.get_response_time_percentile(0.95) or 0)
        rps.append(total.current_rps or 0)

        gevent.sleep(1)

    # Save response-time distribution before shutting down
    resp_time_dist = defaultdict(int, env.stats.total.response_times)
    env.runner.quit()
    return timeline, avg_rt, p95_rt, rps, resp_time_dist


# ──────────────────────────────────────────────────────────────────────────
#  ❸  Plot helpers
# ──────────────────────────────────────────────────────────────────────────
def plot_timeseries(x, y, title, y_label):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()


def plot_histogram(dist_dict):
    # Expand {latency: count, …} into a flat list for plt.hist
    samples = []
    for latency_ms, count in dist_dict.items():
        samples.extend([latency_ms] * count)

    if not samples:  # no data?
        return

    plt.figure()
    plt.hist(samples, bins=30)
    plt.title("Response-Time Distribution")
    plt.xlabel("Response time (ms)")
    plt.ylabel("Frequency")
    plt.tight_layout()


# ──────────────────────────────────────────────────────────────────────────
#  ❹  Main
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("⏳  Running Locust load-test …")
    t, avg, p95, throughput, dist = run_locust()

    print("📈  Rendering plots …")
    plot_timeseries(t, avg, "Average Response-Time Trend", "Avg response time (ms)")
    plot_timeseries(t, p95, "95th-Percentile Response-Time Trend", "p95 response time (ms)")
    plot_timeseries(t, throughput, "Throughput Trend", "Requests per second")
    plot_histogram(dist)

    plt.show()
