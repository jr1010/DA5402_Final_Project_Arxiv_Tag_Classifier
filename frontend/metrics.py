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
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10)
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