"""
SupportIQ demo app.

Run with:
    streamlit run app.py

This is the "evaluator experience" for the project: type a customer
message, click Analyze, and see the SupportIQAgent's full pipeline trace
run live - predicted intent + confidence, the top-k historical cases it
retrieved as evidence, the grounded reply it generated, and the
escalation decision.

Prerequisites (see README / project docs for details):
    1. models/classifier.pkl, models/faiss_index.bin, and
       models/corpus_metadata.csv must already exist - build them once
       with:  python tools/build_artifacts.py
    2. .env must contain a valid GROQ_API_KEY (and optionally
       GROQ_MODEL) for the reply-generation step to work.

Nothing about the underlying model/retrieval/generation/escalation logic
lives in this file - it only calls SupportIQAgent.run() and renders the
result. All of the actual pipeline logic is in src/pipeline/.
"""

import streamlit as st

from src.pipeline.agent import SupportIQAgent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SupportIQ",
    page_icon="🛠️",
)


# ============================================================
# LOAD THE AGENT (ONCE PER SESSION)
# ============================================================
#
# @st.cache_resource makes Streamlit build the SupportIQAgent only the
# first time this runs, and reuse the same instance (same loaded
# classifier, FAISS index, embedding model) across every rerun caused by
# a user interaction - without this, Streamlit would reload every
# artifact from disk on every single button click.

@st.cache_resource
def load_agent() -> SupportIQAgent:
    return SupportIQAgent()


agent = load_agent()


# ============================================================
# HEADER
# ============================================================

st.title("SupportIQ")
st.caption("AI Customer Support Agent — Intent Classification, Retrieval-Grounded Reply Generation, and Escalation")


# ============================================================
# INPUT
# ============================================================

customer_message = st.text_area(
    "Customer message",
    placeholder="My package hasn't arrived yet.",
    height=100,
)

analyze_clicked = st.button("Analyze", type="primary")


# ============================================================
# RUN THE PIPELINE AND DISPLAY THE RESULT
# ============================================================

if analyze_clicked:

    if not customer_message.strip():
        st.warning("Please enter a customer message first.")

    else:
        # Run the full classify -> retrieve -> escalate -> generate
        # pipeline. A spinner is shown because the Groq call in
        # generate_reply() takes a couple of seconds.
        with st.spinner("Running SupportIQ pipeline..."):
            result = agent.run(customer_message)

        st.divider()

        # --- Intent + confidence ------------------------------------------
        st.subheader("Intent")
        st.write(result["intent"])
        st.caption(f"Confidence: {result['confidence']:.2f}")

        # --- Retrieved historical evidence ----------------------------------
        st.subheader("Historical Evidence")
        for case in result["retrieved_cases"]:
            st.markdown(
                f"**{case['rank']}. Similarity: {case['similarity']:.2f} "
                f"— {case['intent']}**"
            )
            with st.expander("View conversation"):
                st.text(case["conversation"])

        # --- Generated reply --------------------------------------------------
        st.subheader("AI Response")
        st.success(result["reply"])

        # --- Escalation decision -----------------------------------------------
        st.subheader("Decision")
        if result["escalation_decision"] == "ESCALATE":
            st.error(result["escalation_decision"])
        else:
            st.info(result["escalation_decision"])
        st.caption(f"Reason: {result['escalation_reason']}")
