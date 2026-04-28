import os
import json
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from utils import load_config


# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------
# Paths (static)
# -------------------------
VAL_PATH = "data/processed/val.csv"
OUTPUT_PATH = "artifacts/metrics.json"


# -------------------------
# Utils
# -------------------------
def process_labels(series):
    return series.apply(lambda x: str(x).split())


# -------------------------
# Main
# -------------------------
def main():
    logger.info("Starting evaluation")

    try:
        # -------------------------
        # Load config
        # -------------------------
        config = load_config()

        VECTORIZER_PATH = config["paths"]["vectorizer"]
        MLB_PATH = config["paths"]["mlb"]
        THRESHOLD = config["inference"]["threshold"]

        MODEL_PATH = "artifacts/model.pkl"  # keep consistent with train

        logger.info(f"Using threshold: {THRESHOLD}")

        # -------------------------
        # Load data
        # -------------------------
        df = pd.read_csv(VAL_PATH)
        logger.info(f"Validation data shape: {df.shape}")

        if "text" not in df.columns or "categories" not in df.columns:
            raise ValueError("Missing required columns in validation data")

        texts = df["text"]
        y_labels = process_labels(df["categories"])

        # -------------------------
        # Load artifacts
        # -------------------------
        logger.info("Loading artifacts")

        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        mlb = joblib.load(MLB_PATH)

        # -------------------------
        # Transform
        # -------------------------
        X = vectorizer.transform(texts)
        X_dense = X.toarray()

        y_true = mlb.transform(y_labels)

        # -------------------------
        # Predict
        # -------------------------
        logger.info("Running inference")

        probs = model.predict_proba(X_dense)
        y_pred = (probs >= THRESHOLD).astype(int)

        # -------------------------
        # Metrics
        # -------------------------
        logger.info("Computing metrics")

        metrics = {
            "f1_micro": f1_score(y_true, y_pred, average="micro"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
            "subset_accuracy": accuracy_score(y_true, y_pred),
            "avg_labels_per_sample": float(np.mean(np.sum(y_pred, axis=1)))
        }

        # -------------------------
        # Save metrics
        # -------------------------
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        with open(OUTPUT_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Saved metrics to {OUTPUT_PATH}")
        logger.info(f"Metrics: {metrics}")

    except Exception:
        logger.exception("Evaluation failed")
        raise


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    main()