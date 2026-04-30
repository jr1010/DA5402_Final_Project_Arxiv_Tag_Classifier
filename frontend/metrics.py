from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import psutil
import os


# ---------------------------
# Counters
# ---------------------------

REQUEST_COUNTER = Counter(
    "arxiv_requests_total",
    "Total prediction requests",
    ["mode"]  # single / bulk
)

TEXTS_PROCESSED = Counter(
    "texts_processed_total",
    "Total texts processed",
    ["mode"]
)

ERROR_COUNTER = Counter(
    "prediction_errors_total",
    "Total prediction errors"
)


# ---------------------------
# Gauges
# ---------------------------

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of active requests"
)

MEMORY_USAGE = Gauge(
    "memory_usage_mb",
    "Memory usage of the application (MB)"
)


# ---------------------------
# Histogram (Latency)
# ---------------------------

FRONTEND_LATENCY = Histogram(
    "frontend_latency_seconds",
    "End-to-end latency from frontend",
    ["mode"],
    buckets=(
        0.01,   # 10 ms
        0.025,  # 25 ms
        0.05,   # 50 ms
        0.075,  # 75 ms
        0.1,    # 100 ms
        0.2,    # 200 ms
        0.5,    # 500 ms
        1,      # 1 s
        2       # 2 s
    )
)


# ---------------------------
# Summary (Bulk size)
# ---------------------------

BATCH_SIZE = Summary(
    "bulk_batch_size",
    "Number of texts in bulk prediction"
)


# ---------------------------
# Utility: Memory Update
# ---------------------------

def update_memory():
    process = psutil.Process(os.getpid())
    MEMORY_USAGE.set(process.memory_info().rss / (1024 * 1024))