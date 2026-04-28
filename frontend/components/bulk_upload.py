import streamlit as st
import pandas as pd


REQUIRED_COLUMNS = {"title", "abstract"}
MAX_ROWS = 1000


def render_bulk_upload():
    st.subheader("Bulk Prediction (CSV Upload)")

    st.info("""
    Upload a CSV file with the following columns:
    - title
    - abstract
    """)

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is None:
        return None, False

    try:
        df = pd.read_csv(uploaded_file)

    except Exception:
        st.error("Failed to read CSV file")
        return None, False

    # -------------------------
    # Column validation
    # -------------------------
    if not REQUIRED_COLUMNS.issubset(df.columns):
        st.error("CSV must contain 'title' and 'abstract' columns")
        st.stop()

    # -------------------------
    # Basic cleaning
    # -------------------------
    df = df.copy()

    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)

    # Remove empty rows
    df["text_len"] = df["title"].str.len() + df["abstract"].str.len()
    df = df[df["text_len"] > 10]

    df = df.drop(columns=["text_len"])

    if len(df) == 0:
        st.error("No valid rows after cleaning")
        st.stop()

    # -------------------------
    # Row limit check
    # -------------------------
    if len(df) > MAX_ROWS:
        st.warning(f"Limiting to first {MAX_ROWS} rows")
        df = df.head(MAX_ROWS)

    # -------------------------
    # Summary stats
    # -------------------------
    st.markdown("### Dataset Summary")

    col1, col2 = st.columns(2)

    col1.metric("Total Rows", len(df))
    col2.metric(
        "Avg Length",
        int((df["title"] + df["abstract"]).str.len().mean())
    )

    # -------------------------
    # Preview
    # -------------------------
    st.markdown("### Preview")

    preview_df = df[["title", "abstract"]].head(5)
    st.dataframe(preview_df, width='stretch')

    # -------------------------
    # Run button
    # -------------------------
    run = st.button(
        "Run Bulk Prediction",
        use_container_width=True
    )

    return df, run