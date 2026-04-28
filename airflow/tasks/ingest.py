import os
import logging
import pandas as pd
import arxivscraper
from datetime import datetime, timedelta

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# Config
# -------------------------
CATEGORY = "cs"
OUTPUT_PATH = "data/raw/raw_data.csv"


# -------------------------
# Date window logic
# -------------------------
def get_month_window(execution_date: datetime):
    """
    Window: (prev_month_start, curr_month_start]
    Example:
        execution_date = 2026-03-01
        → fetch: 2026-02-02 to 2026-03-01
    """
    curr_month_start = execution_date.replace(day=1)

    prev_month_end = curr_month_start - timedelta(days=1)
    prev_month_start = prev_month_end.replace(day=1)

    # Exclusive start → +1 day
    date_from = (prev_month_start + timedelta(days=1)).strftime("%Y-%m-%d")

    # Inclusive end → current month start
    date_until = curr_month_start.strftime("%Y-%m-%d")

    return date_from, date_until


# -------------------------
# Main
# -------------------------
def main(execution_date_str=None):
    logger.info("Starting incremental ingestion")

    try:
        # -------------------------
        # Parse execution date
        # -------------------------
        if execution_date_str:
            execution_date = datetime.strptime(execution_date_str, "%Y-%m-%d")
        else:
            execution_date = datetime.today()

        date_from, date_until = get_month_window(execution_date)

        logger.info(f"Fetching data from {date_from} to {date_until}")

        # -------------------------
        # Scrape data
        # -------------------------
        scraper = arxivscraper.Scraper(
            category=CATEGORY,
            date_from=date_from,
            date_until=date_until
        )

        output = scraper.scrape()
        logger.info(f"Fetched {len(output)} papers")

        if len(output) == 0:
            raise ValueError("No data fetched for this window")

        # -------------------------
        # Convert to DataFrame
        # -------------------------
        df = pd.DataFrame(output)

        required_cols = ["title", "abstract", "categories"]
        df = df[required_cols]

        logger.info(f"Raw batch shape: {df.shape}")

        # -------------------------
        # Save batch
        # -------------------------
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)

        logger.info(f"Saved batch to {OUTPUT_PATH}")

    except Exception:
        logger.exception("Ingestion failed")
        raise


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    main()