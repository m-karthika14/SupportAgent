
import streamlit as st
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT = Path("evaluation/reply_judge_input.csv")
OUTPUT = Path("evaluation/human_judge.csv")

criteria = {
    "human_correctness": "Correctness",
    "human_groundedness": "Groundedness",
    "human_relevance": "Relevance",
    "human_tone": "Tone",
    "human_actionability": "Actionability"
}

descriptions = {
    "human_correctness":
        "Does the reply correctly address the customer's issue?",

    "human_groundedness":
        "Is the reply supported by the retrieved historical evidence?",

    "human_relevance":
        "Is the reply directly relevant to the customer's message?",

    "human_tone":
        "Is the reply professional, polite, and appropriate?",

    "human_actionability":
        "Does the reply give the customer a useful next step?"
}

# ============================================================
# LOAD DATA
# ============================================================

if not INPUT.exists():
    st.error(f"Input file not found: {INPUT}")
    st.stop()

source = pd.read_csv(INPUT)

# Create human evaluation file if it doesn't exist
if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
    df = pd.read_csv(OUTPUT)
else:
    df = source[
        [
            "root_tweet_id",
            "customer_text",
            "golden_intent",
            "predicted_intent",
            "reply",
            "retrieved_evidence"
        ]
    ].copy()

    for col in criteria:
        df[col] = pd.NA

    df["human_notes"] = ""

    df.to_csv(OUTPUT, index=False)

# ============================================================
# SESSION STATE
# ============================================================

if "index" not in st.session_state:
    st.session_state.index = 0

if "saved" not in st.session_state:
    st.session_state.saved = 0

idx = st.session_state.index

# Skip already-rated examples
while idx < len(df):
    row = df.iloc[idx]

    if all(
        pd.notna(row[col]) and str(row[col]).strip() != ""
        for col in criteria
    ):
        idx += 1
    else:
        break

st.session_state.index = idx

# ============================================================
# HEADER
# ============================================================

st.title("SupportIQ — Human Reply Evaluation")

completed = sum(
    df["human_correctness"].notna()
)

st.progress(
    min(completed / len(df), 1.0),
    text=f"Completed: {completed}/{len(df)}"
)

if idx >= len(df):
    st.success("🎉 All 38 replies have been rated!")
    st.write(f"Saved to: `{OUTPUT}`")
    st.stop()

row = df.iloc[idx]

st.subheader(f"Example {idx + 1} / {len(df)}")

st.caption(f"ID: {row['root_tweet_id']}")

# ============================================================
# CUSTOMER
# ============================================================

st.markdown("### 👤 Customer message")

st.info(str(row["customer_text"]))

# ============================================================
# INTENT CONTEXT
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Golden intent**")
    st.write(row["golden_intent"])

with col2:
    st.markdown("**Predicted intent**")
    st.write(row["predicted_intent"])

# ============================================================
# GENERATED REPLY
# ============================================================

st.markdown("### 🤖 Generated reply")

st.success(str(row["reply"]))

# ============================================================
# RETRIEVED EVIDENCE
# ============================================================

with st.expander("View retrieved historical evidence"):
    st.code(str(row["retrieved_evidence"]))

st.divider()

# ============================================================
# RATINGS
# ============================================================

st.markdown("## Rate this reply")

ratings = {}

for col, name in criteria.items():

    st.markdown(f"**{name}**")

    st.caption(descriptions[col])

    ratings[col] = st.radio(
        label=name,
        options=[1, 2, 3, 4, 5],
        horizontal=True,
        key=f"{col}_{idx}",
        index=None
    )

# ============================================================
# NOTES
# ============================================================

notes = st.text_area(
    "Optional notes",
    key=f"notes_{idx}",
    placeholder="Why did you give these scores?"
)

# ============================================================
# SAVE
# ============================================================

if st.button("💾 Save & Next", type="primary"):

    missing = [
        name
        for col, name in criteria.items()
        if ratings[col] is None
    ]

    if missing:
        st.warning(
            "Please rate: " + ", ".join(missing)
        )
        st.stop()

    # Save ratings
    for col in criteria:
        df.loc[idx, col] = ratings[col]

    df.loc[idx, "human_notes"] = notes

    df.to_csv(OUTPUT, index=False)

    st.session_state.index += 1
    st.rerun()
