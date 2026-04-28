import subprocess
import logging
import argparse
import sys
import time
import requests
import signal
import os

# ----------------------------
# CONFIG
# ----------------------------
MLFLOW_PORT = 5001
MLFLOW_MODEL_PATH = "models:/arxiv-classifier/Latest"

MONITORING_DIR = "monitoring"

PROMETHEUS_BIN = os.path.join(MONITORING_DIR, "prometheus", "prometheus")
ALERTMANAGER_BIN = os.path.join(MONITORING_DIR, "alertmanager", "alertmanager")
NODE_EXPORTER_BIN = os.path.join(MONITORING_DIR, "node-exporter", "node_exporter")

PROMETHEUS_CONFIG = os.path.join(MONITORING_DIR, "prometheus.yml")
ALERTMANAGER_CONFIG = os.path.join(MONITORING_DIR, "alertmanager.yml")

DOCKER_CMD = ["docker", "compose"]

processes = []

# ----------------------------
# LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------
# UTIL
# ----------------------------
def is_port_active(port):
    try:
        requests.get(f"http://localhost:{port}", timeout=1)
        return True
    except:
        return False


def run_process(cmd, name):
    logger.info(f"Starting {name}: {' '.join(cmd)}")

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = open(os.path.join(log_dir, f"{name.lower().replace(' ', '_')}.log"), "a")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        stdin=subprocess.DEVNULL,
        start_new_session=True  # 🔥 THIS detaches it
    )

    processes.append(proc)
    return proc


# ----------------------------
# MLFLOW
# ----------------------------
def stop_mlflow():
    try:
        output = subprocess.check_output(["lsof", "-i", f":{MLFLOW_PORT}"])
        pids = [
            line.split()[1]
            for line in output.decode().split("\n")[1:]
            if line
        ]
        if pids:
            subprocess.call(["kill", "-9"] + pids)
            logger.info("Stopped MLflow")
    except:
        logger.info("No MLflow running")


def start_mlflow():
    stop_mlflow()

    cmd = [
        "mlflow", "models", "serve",
        "-m", MLFLOW_MODEL_PATH,
        "-p", str(MLFLOW_PORT),
        "--host", "0.0.0.0"
    ]

    run_process(cmd, "MLflow")

    logger.info("Waiting for MLflow...")
    for _ in range(20):
        if is_port_active(MLFLOW_PORT):
            logger.info("MLflow ready ✅")
            return
        time.sleep(1)

    raise RuntimeError("MLflow failed to start")


# ----------------------------
# MONITORING (LOCAL BINARIES)
# ----------------------------
def start_monitoring():
    logger.info("Starting monitoring stack (local binaries)...")

    # Prometheus
    run_process([
        PROMETHEUS_BIN,
        f"--config.file={PROMETHEUS_CONFIG}"
    ], "Prometheus")

    # Alertmanager
    run_process([
        ALERTMANAGER_BIN,
        f"--config.file={ALERTMANAGER_CONFIG}"
    ], "Alertmanager")

    # Node Exporter
    run_process([
        NODE_EXPORTER_BIN
    ], "Node Exporter")


# ----------------------------
# DOCKER
# ----------------------------
def docker_up(monitoring: bool):
    if monitoring:
        logger.info("Starting Docker (frontend + backend + grafana)")
        cmd = DOCKER_CMD + ["up"]
    else:
        logger.info("Starting Docker (frontend + backend only)")
        cmd = DOCKER_CMD + ["up", "frontend", "backend"]

    run_process(cmd, "Docker Compose")


def docker_down():
    logger.info("Stopping Docker services...")
    subprocess.call(DOCKER_CMD + ["down"])


# ----------------------------
# STOP
# ----------------------------
def stop_all():
    logger.info("Stopping all services...")

    docker_down()
    stop_mlflow()

    for p in processes:
        try:
            p.terminate()
        except:
            pass

    logger.info("All services stopped ✅")


# ----------------------------
# SIGNAL HANDLING
# ----------------------------
def shutdown(signum, frame):
    stop_all()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop"])
    parser.add_argument("--monitoring", action="store_true")

    args = parser.parse_args()

    if args.action == "stop":
        stop_all()
        return

    logger.info("Starting system...")

    # Start MLflow
    start_mlflow()

    # Start monitoring if enabled
    if args.monitoring:
        start_monitoring()

    # Start Docker (Grafana included if monitoring=True)
    docker_up(args.monitoring)

    # Keep alive
    # for p in processes:
    #     p.wait()
    logger.info("All services started successfully ✅")
    logger.info("Run 'python run.py stop' to stop everything")

if __name__ == "__main__":
    main()