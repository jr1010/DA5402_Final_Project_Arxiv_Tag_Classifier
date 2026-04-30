import streamlit as st
import time
import socket

# -------------------------
# START METRICS SERVER
# -------------------------
from prometheus_client import start_http_server

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0
    
if not is_port_in_use(8001):
    start_http_server(8001)

# if "metrics_started" not in st.session_state:
#     start_http_server(8001)
#     st.session_state["metrics_started"] = True

# -------------------------
# Metrics (IMPORTANT: import once)
# -------------------------
import metrics as metrics

from utils import check_ready, predict_single, predict_batch
from components.input_form import render_input_form
from components.bulk_upload import render_bulk_upload
from components.results import show_single_results, show_bulk_results


# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Arxiv Classifier",
    layout="wide"
)

st.title("Arxiv Research Paper Classifier")

st.markdown("""
Classify research papers using **Title + Abstract**.

Supports both single prediction and bulk CSV processing.
""")


# -------------------------
# Sidebar (Mode Selection)
# -------------------------
mode = st.sidebar.radio(
    "Choose Mode",
    ["Single Prediction", "Bulk Prediction"]
)

with st.sidebar.expander("Instructions"):
    st.markdown("""
    **Single Mode**
    - Enter title and abstract
    - Click Predict

    **Bulk Mode**
    - Upload CSV with columns:
        - title
        - abstract
    """)


# -------------------------
# Readiness Gate (CRITICAL)
# -------------------------
status_placeholder = st.empty()

ready = False

for _ in range(10):
    if check_ready():
        ready = True
        break
    status_placeholder.warning("Model is initializing...")
    time.sleep(2)

if not ready:
    status_placeholder.error("Backend not ready. Please try again later.")
    st.stop()

status_placeholder.success("System Ready")


# -------------------------
# Update system metrics
# -------------------------
metrics.update_memory()


# =========================
# SINGLE MODE
# =========================
if mode == "Single Prediction":

    title, abstract, submit = render_input_form()

    if submit:

        # -------------------------
        # Validation
        # -------------------------
        if len(title.strip()) < 5:
            st.warning("Title too short")
            st.stop()

        if len(abstract.strip()) < 20:
            st.warning("Abstract too short")
            st.stop()

        text = title + " " + abstract

        # -------------------------
        # Metrics: start
        # -------------------------
        metrics.REQUEST_COUNTER.labels(mode="single").inc(0)
        metrics.ACTIVE_REQUESTS.inc()
        start_time = time.time()

        # -------------------------
        # Prediction
        # -------------------------
        with st.spinner("Analyzing..."):

            try:
                result = predict_single(text)
                labels = result.get("labels", [])

                latency = time.time() - start_time

                # -------------------------
                # Metrics: success
                # -------------------------
                metrics.FRONTEND_LATENCY.labels(mode='single').observe(latency)
                metrics.TEXTS_PROCESSED.labels(mode="single").inc(0)

                show_single_results(labels)

            except Exception:
                metrics.ERROR_COUNTER.inc()
                st.error("Prediction failed. Please try again.")

            finally:
                metrics.ACTIVE_REQUESTS.dec()


# =========================
# BULK MODE
# =========================
else:

    df, run = render_bulk_upload()

    if df is not None and run:

        texts = (df["title"] + " " + df["abstract"]).tolist()

        # -------------------------
        # Metrics: start
        # -------------------------
        metrics.REQUEST_COUNTER.labels(mode="bulk").inc()
        metrics.ACTIVE_REQUESTS.inc()
        start_time = time.time()

        with st.spinner("Processing batch..."):

            try:
                result = predict_batch(texts)
                predictions = result.get("predictions", [])

                latency = time.time() - start_time

                # -------------------------
                # Metrics: success
                # -------------------------
                metrics.FRONTEND_LATENCY.labels(mode='bulk').observe(latency)
                metrics.TEXTS_PROCESSED.labels(mode="bulk").inc(len(texts))
                metrics.BATCH_SIZE.observe(len(texts))

                show_bulk_results(df, predictions)

            except Exception:
                metrics.ERROR_COUNTER.inc()
                st.error("Bulk prediction failed. Please try again.")

            finally:
                metrics.ACTIVE_REQUESTS.dec()