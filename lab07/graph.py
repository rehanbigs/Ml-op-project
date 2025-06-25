"""
Turn gradio_test_stats_history.csv into 4 clear graphs:
  1. Average response-time vs wall clock
  2. 95th-percentile latency vs wall clock
  3. Requests per second vs wall clock
  4. Histogram of all response times
"""

import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "gradio_test_stats_history.csv"

# ------------------------------------------------------------------ #
# 1) Load and keep only the 'Aggregated' row (whole-test roll-up)
# ------------------------------------------------------------------ #
df = pd.read_csv(CSV_FILE)
agg = df[df["Name"] == "Aggregated"].copy()
agg["time"] = pd.to_datetime(agg["Timestamp"], unit="s")

# ------------------------------------------------------------------ #
# 2) Time-series charts
# ------------------------------------------------------------------ #
plt.figure()
plt.plot(agg["time"], agg["Average Response Time"])
plt.title("Average Response-Time Trend")
plt.xlabel("Time")
plt.ylabel("Avg response time (ms)")
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.plot(agg["time"], agg["95%"])
plt.title("95th-Percentile Response-Time Trend")
plt.xlabel("Time")
plt.ylabel("p95 latency (ms)")
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.plot(agg["time"], agg["Requests/s"])
plt.title("Throughput Trend (Requests / second)")
plt.xlabel("Time")
plt.ylabel("RPS")
plt.grid(True)
plt.tight_layout()

# ------------------------------------------------------------------ #
# 3) Histogram of every request Latency (all endpoints)
#    – uses the large 'Response Time' column
# ------------------------------------------------------------------ #
raw = pd.read_csv("gradio_test_stats.csv")          # one row per endpoint
hist_data = raw["Avg Response Time"].values

plt.figure()
plt.hist(hist_data, bins=30)
plt.title("Distribution of Average Response Times")
plt.xlabel("Response time (ms)")
plt.ylabel("Frequency (# endpoints)")
plt.tight_layout()

plt.show()
