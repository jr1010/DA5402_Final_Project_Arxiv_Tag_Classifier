import os
import joblib
import pandas as pd
import mlflow
import mlflow.lightgbm
import numpy as np
import logging
import random

from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
from mlflow.tracking import MlflowClient

from utils import (
    load_config,
    process_labels,
    fit_mlb,
    transform_mlb,
    generate_param_grid,
    apply_threshold,
    compute_metrics,
)

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# Paths
# -------------------------
TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
VECTORIZER_PATH = "artifacts/vectorizer.pkl"
OUTPUT_DIR = "artifacts"


def main():
    logger.info("Starting training with HPO")

    try:
        params = load_config()
        SEED = params["training"]["random_state"]

        random.seed(SEED)
        np.random.seed(SEED)

        threshold = params["training"]["threshold"]
        fixed_params = params["model"]["fixed"]
        search_space = params["model"]["search"]

        # -------------------------
        # MLflow setup
        # -------------------------
        # mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

        mlflow.set_experiment("arxiv-classification")

        # Enable system metrics
        # mlflow.system_metrics.enable()
        mlflow.enable_system_metrics_logging()

        # Enable autologging for LightGBM
        mlflow.lightgbm.autolog(log_models=False)

        client = MlflowClient()
        exp = client.get_experiment_by_name("arxiv-classification")
        logger.info(f"Artifact location: {exp.artifact_location}")

        # -------------------------
        # Load data
        # -------------------------
        logger.info("Loading datasets")
        train_df = pd.read_csv(TRAIN_PATH)
        val_df = pd.read_csv(VAL_PATH)
        logger.info(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")

        # -------------------------
        # Labels
        # -------------------------
        logger.info("Processing labels")
        y_train_labels = process_labels(train_df["categories"])
        y_val_labels = process_labels(val_df["categories"])

        mlb, y_train = fit_mlb(y_train_labels)
        y_val = transform_mlb(mlb, y_val_labels)

        # -------------------------
        # Load vectorizer
        # -------------------------
        logger.info("Loading vectorizer")
        vectorizer = joblib.load(VECTORIZER_PATH)

        X_train = vectorizer.transform(train_df["text"])
        X_val = vectorizer.transform(val_df["text"])

        logger.info(f"Feature shapes -> Train: {X_train.shape}, Val: {X_val.shape}")

        # -------------------------
        # Generate HPO grid
        # -------------------------
        param_grid = generate_param_grid(search_space)
        logger.info(f"Total experiments: {len(param_grid)}")

        best_score = 0
        best_model = None
        best_params = None

        # -------------------------
        # HPO Loop
        # -------------------------
        for i, hp in enumerate(param_grid):
            logger.info(f"Running experiment {i+1}/{len(param_grid)}")

            full_params = {
                **fixed_params,
                **hp,
                "random_state": SEED,
                "feature_fraction_seed": SEED,
                "bagging_seed": SEED,
                "force_col_wise": True
            }

            with mlflow.start_run(run_name=f"exp_{i}"):

                model = OneVsRestClassifier(
                    LGBMClassifier(**full_params)
                )

                # Train
                model.fit(X_train, y_train)

                # Log
                mlflow.lightgbm.log_model(
                    model,
                    artifact_path="model"
                )

                # Predict
                probs = model.predict_proba(X_val)
                y_pred = apply_threshold(probs, threshold)

                # Metrics
                metrics = compute_metrics(y_val, y_pred)

                from sklearn.metrics import f1_score
                per_label_f1 = f1_score(y_val, y_pred, average=None)
                avg_labels = np.mean(np.sum(y_pred, axis=1))

                logger.info(f"Metrics: {metrics}")

                # Manual logging (kept intentionally)
                mlflow.log_params(full_params)

                mlflow.log_metric("f1_micro", metrics["f1_micro"])
                mlflow.log_metric("f1_macro", metrics["f1_macro"])

                for k, v in metrics.items():
                    if k not in ["f1_micro", "f1_macro"]:
                        mlflow.log_metric(k, v)

                mlflow.log_metric("avg_labels_per_sample", avg_labels)

                for j, label in enumerate(mlb.classes_):
                    mlflow.log_metric(f"f1_{label}", per_label_f1[j])

                # Track best
                if metrics["f1_micro"] > best_score:
                    logger.info(f"New best model found with F1_micro={metrics['f1_micro']}")
                    best_score = metrics["f1_micro"]
                    best_model = model
                    best_params = full_params

        # -------------------------
        # Save best model locally
        # -------------------------
        logger.info("Saving best model locally")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        joblib.dump(best_model, f"{OUTPUT_DIR}/model.pkl")
        joblib.dump(mlb, f"{OUTPUT_DIR}/mlb.pkl")

        logger.info(f"Best F1 Micro: {best_score}")
        logger.info(f"Best Params: {best_params}")

        # -------------------------
        # Register best model
        # -------------------------
        logger.info("Registering best model")

        with mlflow.start_run(run_name="best_model") as run:

            mlflow.log_params(best_params)
            mlflow.log_metric("best_f1_micro", best_score)

            mlflow.lightgbm.log_model(
                best_model,
                artifact_path="model",
            )

            run_id = run.info.run_id

            mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name="arxiv-classifier"
            )

            mlflow.log_artifact(f"{OUTPUT_DIR}/mlb.pkl")

        logger.info("Training pipeline completed successfully")

    except Exception:
        logger.exception("Training pipeline failed")
        raise


if __name__ == "__main__":
    main()