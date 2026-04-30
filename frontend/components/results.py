import streamlit as st
import pandas as pd


LABEL_MAP = {
    "cs.lg": "Machine Learning",
    "cs.cv": "Computer Vision",
    "cs.ai": "Artificial Intelligence",
    "cs.cl": "Computation and Language",
    "cs.ro": "Robotics"
}


# -------------------------
# Helper: Format Label
# -------------------------
def format_label(tag):
    return f"{LABEL_MAP.get(tag, tag)} ({tag})"


# -------------------------
# Single Prediction Display
# -------------------------
def show_single_results(labels):
    st.markdown("## Predictions")

    if not labels:
        st.info("No labels predicted")
        return

    # -------------------------
    # Top Prediction (emphasized)
    # -------------------------
    top_label = labels[0]
    top_name = LABEL_MAP.get(top_label, top_label)

    st.markdown("### 🏆 Top Prediction")

    st.markdown(
        f"""
        <div style="
            font-size: 24px;
            font-weight: 600;
            padding: 12px 0;
        ">
            {top_name}
            <span style="opacity:0.6; font-size:14px; margin-left:8px;">
                ({top_label})
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # -------------------------
    # All Predictions
    # -------------------------
    st.markdown("### All Predictions")

    for i, tag in enumerate(labels, start=1):
        name = LABEL_MAP.get(tag, tag)

        st.markdown(
            f"""
            <div style="font-size:16px; margin-bottom:6px;">
                <span style="font-weight:600;">{i}.</span>
                <span style="margin-left:8px;">
                    {name}
                    <span style="opacity:0.6; font-size:13px;">
                        ({tag})
                    </span>
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    # # -------------------------
    # # Top Prediction
    # # -------------------------
    # st.markdown("### 🏆 Top Prediction")
    # render_row(1, labels[0])

    # # -------------------------
    # # All Predictions
    # # -------------------------
    # st.markdown("### All Predictions")

    # for i, label in enumerate(labels, start=1):
    #     render_row(i, label)


# -------------------------
# Bulk Prediction Display
# -------------------------
def show_bulk_results(df, predictions):
    st.markdown("## Results")

    pretty_preds = [
        ", ".join(format_label(p) for p in pred)
        for pred in predictions
    ]

    df = df.copy()
    df["Predictions"] = pretty_preds

    st.success("Batch processing complete")

    st.dataframe(df, width='stretch')

    # -------------------------
    # Download button
    # -------------------------
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )