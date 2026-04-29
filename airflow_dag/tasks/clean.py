import os
import logging
import pandas as pd

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/raw_data.csv"
MASTER_PATH = "data/raw/data.csv"


def main():
    logger.info("Starting incremental cleaning + append")

    try:
        if not os.path.exists(RAW_PATH):
            raise FileNotFoundError("raw_data.csv not found")

        new_df = pd.read_csv(RAW_PATH)
        logger.info(f"New batch size: {new_df.shape}")

        required_cols = ["title", "abstract", "categories"]
        new_df = new_df.dropna(subset=required_cols)

        # ensure string type
        for col in required_cols:
            new_df[col] = new_df[col].astype(str)

        # -------------------------
        # Load existing data
        # -------------------------
        if os.path.exists(MASTER_PATH):
            existing_df = pd.read_csv(MASTER_PATH)
            logger.info(f"Existing dataset size: {existing_df.shape}")

            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            logger.info("No existing dataset found, creating new one")
            combined = new_df

        # -------------------------
        # Deduplicate
        # -------------------------
        before = len(combined)
        combined = combined.drop_duplicates(subset=["title", "abstract"])
        after = len(combined)

        logger.info(f"Removed {before - after} duplicates")
        logger.info(f"Final dataset size: {after}")

        # -------------------------
        # Save
        # -------------------------
        combined.to_csv(MASTER_PATH, index=False)
        logger.info(f"Updated master dataset at {MASTER_PATH}")

        # -------------------------
        # Delete raw batch
        # -------------------------
        os.remove(RAW_PATH)
        logger.info("Deleted raw_data.csv after processing")

    except Exception:
        logger.exception("Cleaning failed")
        raise


if __name__ == "__main__":
    main()