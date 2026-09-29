"""
Streamlit dashboard — a pure frontend over the FastAPI service. Holds no
pipeline logic itself; every action here is just an HTTP call to the API
we already built and tested in Sub-phase 5.1.
"""

import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Pipeline", layout="wide")
st.title("📚 RAG Pipeline — Hybrid Search over FastAPI Docs")

with st.sidebar:
    st.header("Settings")
    strategy = st.selectbox(
        "Chunking strategy",
        options=["structural", "fixed", "semantic"],
        help="Which chunking strategy's index to search against.",
    )
    top_k = st.slider("Number of chunks to use", min_value=1, max_value=10, value=5)
    compare_mode = st.checkbox(
        "Compare hybrid vs. dense-only",
        help="Shows retrieval results side by side for both retrieval modes.",
    )

question = st.text_input("Ask a question about FastAPI:", placeholder="How do I use BackgroundTasks?")
ask_clicked = st.button("Ask", type="primary")


def call_ask_api(q: str, strat: str, k: int,mode: str = "hybrid") -> dict | None:
    """Calls the /v1/ask endpoint and returns the parsed response, or
    None (with an error shown to the user) if the call fails."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/ask",
            json={"question": q, "strategy": strat, "top_k": k,"retrieval_mode": mode},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Is `uvicorn src.api.main:app` running?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e}")
        return None


def render_answer(result: dict) -> None:
    """Renders one /v1/ask response: the answer, confidence breakdown,
    and per-citation verification detail."""
    if result["insufficient_context"]:
        st.warning(f"⚠️ {result['answer']}")
        return

    st.markdown("### Answer")
    st.write(result["answer"])

    conf = result["confidence"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieval confidence", f"{conf['retrieval_confidence']:.0%}")
    col2.metric(
        "Citation coverage",
        f"{conf['citation_coverage']:.0%}" if conf["citation_coverage"] is not None else "N/A",
    )
    col3.metric(
        "Composite confidence",
        f"{conf['composite_score']:.0%}" if conf["composite_score"] is not None else "N/A",
    )

    if result["citations"]:
        st.markdown("### Citations")
        for c in result["citations"]:
            icon = "✅" if c["verified"] else "❌"
            st.markdown(f"{icon} **[{c['citation_number']}]** `{c['source_path']}` — {c['reason']}")

    st.markdown("### Retrieved sources")
    for src in result["retrieved_sources"]:
        st.markdown(f"- `{src}`")


if ask_clicked and question.strip():
    if compare_mode:
        col_hybrid, col_dense = st.columns(2)
        with col_hybrid:
            st.subheader("Hybrid (dense + BM25 + rerank)")
            with st.spinner("Thinking..."):
                result = call_ask_api(question, strategy, top_k, mode="hybrid")
            if result:
                render_answer(result)
        with col_dense:
            st.subheader("Dense-only")
            with st.spinner("Thinking..."):
                result = call_ask_api(question, strategy, top_k, mode="dense_only")
            if result:
                render_answer(result)
    else:
        with st.spinner("Thinking..."):
            result = call_ask_api(question, strategy, top_k, mode="hybrid")
        if result:
            render_answer(result)
elif ask_clicked:
    st.warning("Please enter a question.")