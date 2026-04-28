import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

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
RAW_PATH = "data/raw/data.csv"
PROCESSED_DIR = "data/processed/"


# -------------------------
# Text cleaning
# -------------------------
def clean_text(text):
    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.strip()
    return text


# -------------------------
# Label filtering
# -------------------------
def filter_labels(label_str, selected_labels):
    labels = str(label_str).split()
    labels = [l.lower() for l in labels]
    return [l for l in labels if l in selected_labels]


# -------------------------
# Main
# -------------------------
def main():
    logger.info("Starting preprocessing pipeline")

    try:
        params = load_config()

        selected_labels = set(params["labels"]["selected"])
        test_size = params["training"]["test_size"]
        random_state = params["training"]["random_state"]

        # -------------------------
        # Load data
        # -------------------------
        logger.info(f"Loading raw dataset from {RAW_PATH}")
        df = pd.read_csv(RAW_PATH)
        logger.info(f"Loaded dataset with shape: {df.shape}")

        # -------------------------
        # Basic cleaning
        # -------------------------
        logger.info("Cleaning text fields")

        df = df.dropna(subset=["title", "abstract", "categories"])

        df["title"] = df["title"].apply(clean_text)
        df["abstract"] = df["abstract"].apply(clean_text)

        df["text"] = df["title"] + " " + df["abstract"]

        # -------------------------
        # Label processing
        # -------------------------
        logger.info("Processing labels")

        df["labels"] = df["categories"].apply(
            lambda x: filter_labels(x, selected_labels)
        )

        df = df[df["labels"].map(len) > 0]

        df["categories"] = df["labels"].apply(lambda x: " ".join(x))

        # -------------------------
        # Remove duplicates
        # -------------------------
        logger.info("Removing duplicate samples")

        before = len(df)
        df = df.drop_duplicates(subset=["text"])
        df = df.reset_index(drop=True)
        after = len(df)

        logger.info(f"Removed {before - after} duplicates")
        logger.info(f"Dataset size after cleaning: {after}")

        # -------------------------
        # Train / Validation split
        # -------------------------
        logger.info("Splitting dataset")

        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

        logger.info(f"Train size: {len(train_df)}")
        logger.info(f"Validation size: {len(val_df)}")

        # -------------------------
        # Save
        # -------------------------
        logger.info(f"Saving processed data to {PROCESSED_DIR}")

        os.makedirs(PROCESSED_DIR, exist_ok=True)

        train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
        val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)

        logger.info("Preprocessing pipeline completed successfully")

    except Exception:
        logger.exception("Preprocessing pipeline failed")
        raise


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    main()