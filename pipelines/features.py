import os
import logging
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import load_config

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
OUTPUT_PATH = "artifacts/vectorizer.pkl"


# -------------------------
# Main
# -------------------------
def main():
    logger.info("Starting feature engineering (fit on train)")

    try:
        params = load_config()

        data_params = params["data"]
        max_features = data_params["max_features"]
        ngram_range = tuple(data_params["ngram_range"])
        stop_words = data_params.get("stop_words", None)
        sublinear_tf = data_params.get("sublinear_tf", False)

        # -------------------------
        # Load train data
        # -------------------------
        logger.info(f"Loading training data from {TRAIN_PATH}")
        df = pd.read_csv(TRAIN_PATH)
        logger.info(f"Loaded dataset with shape: {df.shape}")

        if "text" not in df.columns:
            raise ValueError("Column 'text' not found in train data")

        # -------------------------
        # Initialize vectorizer
        # -------------------------
        logger.info("Initializing TF-IDF vectorizer")

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
            sublinear_tf=sublinear_tf
        )

        # -------------------------
        # Fit on train
        # -------------------------
        logger.info("Fitting vectorizer on training data")
        vectorizer.fit(df["text"])

        vocab_size = len(vectorizer.vocabulary_)
        logger.info(f"Vocabulary size: {vocab_size}")

        # -------------------------
        # Save vectorizer
        # -------------------------
        logger.info(f"Saving vectorizer to {OUTPUT_PATH}")

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        joblib.dump(vectorizer, OUTPUT_PATH)

        logger.info("Feature engineering completed successfully")

    except Exception:
        logger.exception("Feature engineering pipeline failed")
        raise


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    main()