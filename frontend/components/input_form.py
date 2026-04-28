import streamlit as st
import time


MAX_TITLE_LEN = 200
MAX_ABSTRACT_LEN = 3000

MIN_TITLE_LEN = 5
MIN_ABSTRACT_LEN = 20

PAUSE_THRESHOLD = 0.7  # seconds
RERUN_INTERVAL = 0.3   # seconds


def render_input_form():
    st.subheader("Enter Paper Details")

    # -------------------------
    # Initialize session state
    # -------------------------
    if "title" not in st.session_state:
        st.session_state["title"] = ""

    if "abstract" not in st.session_state:
        st.session_state["abstract"] = ""

    if "last_abstract" not in st.session_state:
        st.session_state["last_abstract"] = ""

    if "last_edit_time" not in st.session_state:
        st.session_state["last_edit_time"] = time.time()

    # -------------------------
    # Layout
    # -------------------------
    col1, col2 = st.columns([2, 1])

    with col1:

        # -------------------------
        # Title Input (normal live)
        # -------------------------
        title = st.text_input(
            "Title",
            key="title",
            max_chars=MAX_TITLE_LEN,
            placeholder="e.g., Attention Is All You Need",
            help="Enter the research paper title"
        )

        st.caption(f"{len(title)}/{MAX_TITLE_LEN} characters")

        if 0 < len(title.strip()) < MIN_TITLE_LEN:
            st.warning("Title is too short")

        # -------------------------
        # Abstract Input (debounced)
        # -------------------------
        abstract = st.text_area(
            "Abstract",
            key="abstract",
            height=220,
            max_chars=MAX_ABSTRACT_LEN,
            placeholder="Summarize the paper's objective, methodology, and key results...",
            help="Enter the abstract (more detail improves prediction)"
        )

        # Detect changes
        if abstract != st.session_state["last_abstract"]:
            st.session_state["last_abstract"] = abstract
            st.session_state["last_edit_time"] = time.time()

        # Time since last edit
        time_since_edit = time.time() - st.session_state["last_edit_time"]
        is_stable = time_since_edit > PAUSE_THRESHOLD

        # Character counter (always visible)
        st.caption(f"{len(abstract)}/{MAX_ABSTRACT_LEN} characters")

        # Validation
        if 0 < len(abstract.strip()) < MIN_ABSTRACT_LEN:
            st.warning("Abstract is too short")

        # UX feedback
        if not is_stable:
            st.caption("⏳ Waiting for typing to pause...")

        # -------------------------
        # Validation
        # -------------------------
        is_invalid = (
            len(title.strip()) < MIN_TITLE_LEN or
            len(abstract.strip()) < MIN_ABSTRACT_LEN
        )

        # -------------------------
        # Submit Button
        # -------------------------
        submit = st.button(
            "Predict",
            width='stretch',
            disabled=is_invalid
        )

    with col2:
        st.markdown("### Tips")
        st.markdown("""
        - Use complete abstracts  
        - Include key technical terms  
        - Avoid very short inputs  
        """)

        st.markdown("### Limits")
        st.markdown(f"""
        - Title ≥ {MIN_TITLE_LEN} chars  
        - Abstract ≥ {MIN_ABSTRACT_LEN} chars  
        - Title ≤ {MAX_TITLE_LEN} chars  
        - Abstract ≤ {MAX_ABSTRACT_LEN} chars  
        """)

    # -------------------------
    # Controlled rerun (debounce loop)
    # -------------------------
    if not is_stable:
        time.sleep(RERUN_INTERVAL)
        st.rerun()

    return title, abstract, submit