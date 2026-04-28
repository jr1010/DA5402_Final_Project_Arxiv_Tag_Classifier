import requests
import yaml
import os


def load_config(path="frontend/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


config = load_config()
BASE_URL = os.getenv("BACKEND_URL", config["backend"]["base_url"])


# -------------------------
# Health / Ready
# -------------------------
def check_ready():
    try:
        r = requests.get(f"{BASE_URL}/ready", timeout=2)
        return r.status_code == 200
    except:
        return False


# -------------------------
# Prediction APIs
# -------------------------
def predict_single(text):
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"text": text}
    )
    r.raise_for_status()
    return r.json()


def predict_batch(texts):
    r = requests.post(
        f"{BASE_URL}/predict-batch",
        json={"texts": texts}
    )
    r.raise_for_status()
    return r.json()