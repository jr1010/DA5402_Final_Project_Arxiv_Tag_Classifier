import requests
import joblib
import numpy as np
import pandas as pd
import os
import logging

from backend.utils import load_config


# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------
# Global state
# -------------------------
STATE = "starting"   # starting | ready | error
ERROR_MSG = None

vectorizer = None
mlb = None
feature_names = None


# -------------------------
# Load config
# -------------------------
config = load_config()

MLFLOW_URL = os.getenv("MLFLOW_URL", config["mlflow"]["url"])
THRESHOLD = config["inference"]["threshold"]


# -------------------------
# Initialization
# -------------------------
def initialize():
    global vectorizer, mlb, STATE, ERROR_MSG

    logger.info("Initializing inference components")

    try:
        vectorizer = joblib.load(config["paths"]["vectorizer"])
        mlb = joblib.load(config["paths"]["mlb"])
        feature_names = vectorizer.get_feature_names_out()

        # Warmup
        _ = vectorizer.transform(["test"]).toarray()

        STATE = "ready"
        ERROR_MSG = None

        logger.info("Inference initialized successfully")

    except Exception:
        STATE = "error"
        ERROR_MSG = "Initialization failed"
        logger.exception("Initialization failed")
        raise


# -------------------------
# MLflow call
# -------------------------
def call_mlflow(X):
    global STATE, ERROR_MSG

    try:
        logger.info(f"Calling MLflow service | batch_size={X.shape[0]}")

        response = requests.post(
            MLFLOW_URL,
            json={"inputs": X.tolist()},
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        # -------------------------
        # Extract predictions
        # -------------------------
        if isinstance(data, dict) and "predictions" in data:
            preds = np.array(data["predictions"])

        elif isinstance(data, list):
            preds = np.array(data)

        else:
            raise RuntimeError(f"Unexpected MLflow response format")

        # -------------------------
        # Sanity check
        # -------------------------
        if preds.ndim != 2:
            raise RuntimeError(f"Invalid prediction shape: {preds.shape}")

        logger.info(f"MLflow response received | shape={preds.shape}")

        return preds

    except Exception:
        STATE = "error"
        ERROR_MSG = "MLflow call failed"
        logger.exception("MLflow call failed")
        raise RuntimeError("Model service unavailable")


# -------------------------
# Prediction logic
# -------------------------
def predict_batch(texts):
    logger.info(f"Received batch prediction request | size={len(texts)}")

    if STATE != "ready":
        logger.error(f"Model not ready | state={STATE}")
        raise RuntimeError("Model not ready")

    if not texts or len(texts) == 0:
        logger.warning("Empty input received")
        return []

    try:
        # -------------------------
        # Preprocess
        # -------------------------
        X = vectorizer.transform(texts)
        X_dense = X.toarray()

        logger.info(f"Vectorized input | shape={X_dense.shape}")

        # -------------------------
        # Model inference
        # -------------------------
        probs = call_mlflow(X_dense)

        # -------------------------
        # Postprocess
        # -------------------------
        preds = (probs >= THRESHOLD).astype(int)
        labels = mlb.inverse_transform(preds)

        logger.info("Prediction completed successfully")

        return [list(l) for l in labels]

    except Exception:
        logger.exception("Batch prediction failed")
        raise


def predict_single(text):
    logger.info("Received single prediction request")
    return predict_batch([text])[0]