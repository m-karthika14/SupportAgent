#!/usr/bin/env python3
"""
Build and save the SupportIQ pipeline artifacts (classifier + retrieval index).

WHY THIS FILE EXISTS
---------------------
Every piece of the SupportIQ agent (the Hybrid TF-IDF + MiniLM intent
classifier, and the MiniLM + FAISS retrieval index) was originally
developed and validated inside two Jupyter notebooks:

    notebooks/05_classifier_evaluation.ipynb   -> trains the Hybrid classifier
                                                   (reports 54.0% accuracy /
                                                   0.484 macro F1 on the
                                                   200-example Golden Set)
    notebooks/06_retrieval_evaluation.ipynb    -> builds the FAISS retrieval
                                                   index over the 10K dev
                                                   corpus, and defines the
                                                   generation + escalation
                                                   logic

Those notebooks are great for showing *how* the model was built, but they
are not something a Streamlit demo app can "import" - the classifier and
the FAISS index only exist as in-memory variables while the notebook
kernel is running.

This script re-runs EXACTLY the same training/indexing code that produced
the reported numbers (same TF-IDF settings, same MiniLM model, same
train/test split with the same random_state, same Logistic Regression
settings) and then SAVES the fitted objects to disk, once, under
models/. Nothing about the modeling logic is changed here - this file
only adds persistence so the demo app can load pre-built artifacts
instantly instead of retraining/re-embedding on every launch.

USAGE
-----
    python tools/build_artifacts.py

Run this once (and again any time dev_pseudo_labeled.csv changes). It
writes three files:

    models/classifier.pkl        - fitted TF-IDF vectorizer + fitted
                                    Hybrid Logistic Regression classifier
    models/faiss_index.bin       - FAISS IndexFlatIP over the full 10K
                                    conversations (cosine similarity, since
                                    embeddings are L2-normalized)
    models/corpus_metadata.csv   - the 10K conversations + intents, in the
                                    exact same row order as the FAISS
                                    index, so "FAISS row i" always means
                                    "corpus_metadata row i"
"""

from pathlib import Path

import joblib
import pandas as pd
import faiss
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

# Same 10K pseudo-labeled dataset used everywhere else in the project -
# it doubles as (a) the training data for the classifier and (b) the
# retrieval corpus that gets embedded and indexed with FAISS.
DEV_PATH = "data/processed/dev_pseudo_labeled.csv"

MODELS_DIR = Path("models")


# ============================================================
# HELPER: extract only the customer's turns from a conversation
# ============================================================
#
# Identical to the extract_customer_text() helper defined in both
# notebooks/05_classifier_evaluation.ipynb and
# notebooks/06_retrieval_evaluation.ipynb. The classifier was trained on
# CUSTOMER-only text (not the AMAZON: replies), so a request must be
# reduced the same way at prediction time or the feature distribution
# won't match what the model learned.

def extract_customer_text(conversation):
    lines = str(conversation).splitlines()

    customer_messages = []

    for line in lines:
        if line.startswith("CUSTOMER:"):
            # Strip the "CUSTOMER:" prefix, keep only the message text.
            message = line.replace("CUSTOMER:", "", 1).strip()
            customer_messages.append(message)

    return " ".join(customer_messages)


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 - Load the 10K dev pseudo-labeled dataset
    # ------------------------------------------------------------------
    print("Loading 10K dev pseudo-labeled dataset...")
    dev = pd.read_csv(DEV_PATH)

    # Clean whitespace off the label column, same as the notebooks do.
    dev["pseudo_intent"] = dev["pseudo_intent"].str.strip()

    # Pre-compute the customer-only text once; used for classifier
    # training below (retrieval embeddings use the FULL conversation
    # text instead - see STEP 5).
    dev["customer_text"] = dev["conversation"].apply(extract_customer_text)

    # ------------------------------------------------------------------
    # STEP 2 - Reproduce the exact 8K/2K split from notebooks/05 & 06
    # ------------------------------------------------------------------
    # Same test_size, same random_state, same stratify column as the
    # notebooks -> this produces the identical train/test split, which is
    # what makes the reported 54.0% Golden Set accuracy reproducible here.
    dev_train, dev_test = train_test_split(
        dev,
        test_size=2000,
        random_state=42,
        stratify=dev["pseudo_intent"],
    )
    print(f"Train examples: {len(dev_train)}   Test examples: {len(dev_test)}")

    # ------------------------------------------------------------------
    # STEP 3 - Load the MiniLM embedding model
    # ------------------------------------------------------------------
    # Same pretrained model used for BOTH the classifier's semantic
    # features and the retrieval embeddings, kept consistent on purpose
    # so a single downloaded model serves the whole pipeline.
    print("Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ------------------------------------------------------------------
    # STEP 4 - Train the Hybrid (TF-IDF + MiniLM) classifier
    # ------------------------------------------------------------------
    # This is the "Hybrid" model from the results table (58.0% / 0.503 on
    # the 2K pseudo-label test set, 54.0% / 0.484 on the 200-example
    # Golden Set) - TF-IDF lexical features concatenated with MiniLM
    # semantic embeddings, fed into one balanced Logistic Regression.

    print("Fitting TF-IDF vectorizer on the training split only...")
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50000,
    )
    # Fit ONLY on training data - fitting on test/golden data too would
    # leak vocabulary information and inflate the reported accuracy.
    X_train_tfidf = tfidf_vectorizer.fit_transform(dev_train["customer_text"])

    print("Encoding the training split with MiniLM (semantic features)...")
    X_train_embed = embedding_model.encode(
        dev_train["customer_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Concatenate the sparse TF-IDF matrix with the dense MiniLM
    # embeddings (converted to sparse) into one combined feature matrix.
    X_train_hybrid = hstack([
        X_train_tfidf,
        csr_matrix(X_train_embed),
    ])

    print("Training the Hybrid Logistic Regression classifier...")
    hybrid_classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # corrects for the heavy class imbalance
                                  # (Delivery Issue has ~4000+ examples,
                                  # Prime Membership has only ~16)
        random_state=42,
    )
    hybrid_classifier.fit(X_train_hybrid, dev_train["pseudo_intent"])

    # Save both fitted pieces the classifier needs at inference time.
    # (MiniLM itself is NOT re-saved here - it's a pretrained model
    # pulled by name at runtime, exactly like the notebooks do, and
    # sentence-transformers caches the downloaded weights locally after
    # the first run.)
    classifier_artifact = {
        "tfidf_vectorizer": tfidf_vectorizer,
        "hybrid_classifier": hybrid_classifier,
    }
    joblib.dump(classifier_artifact, MODELS_DIR / "classifier.pkl")
    print(f"Saved {MODELS_DIR / 'classifier.pkl'}")

    # ------------------------------------------------------------------
    # STEP 5 - Build the FAISS retrieval index
    # ------------------------------------------------------------------
    # Exactly as in notebooks/06_retrieval_evaluation.ipynb: embed the
    # FULL conversation text (customer AND Amazon turns - not just the
    # customer_text used for classification) for ALL 10,000 rows, not
    # just the training split, since the retrieval corpus is meant to
    # cover the whole dataset.
    print("Encoding the full 10K retrieval corpus with MiniLM (slow step)...")
    retrieval_embeddings = embedding_model.encode(
        dev["conversation"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    dimension = retrieval_embeddings.shape[1]

    # A flat inner-product index. Because every embedding is
    # L2-normalized, inner product is equivalent to cosine similarity.
    retrieval_index = faiss.IndexFlatIP(dimension)
    retrieval_index.add(retrieval_embeddings)

    faiss.write_index(retrieval_index, str(MODELS_DIR / "faiss_index.bin"))
    print(f"Saved {MODELS_DIR / 'faiss_index.bin'} ({retrieval_index.ntotal} vectors)")

    # ------------------------------------------------------------------
    # STEP 6 - Save row-aligned corpus metadata
    # ------------------------------------------------------------------
    # CRITICAL: FAISS only returns row numbers (0, 1, 2, ...), not the
    # original conversation text. So "FAISS row i" must always map to
    # "corpus_metadata row i" - this file must be built from the exact
    # same (unfiltered, unsorted) dataframe and row order used to build
    # retrieval_embeddings above, or lookups will point at the wrong case.
    corpus_metadata = dev[["root_tweet_id", "conversation", "pseudo_intent"]].reset_index(drop=True)
    corpus_metadata.to_csv(MODELS_DIR / "corpus_metadata.csv", index=False)
    print(f"Saved {MODELS_DIR / 'corpus_metadata.csv'} ({len(corpus_metadata)} rows)")

    print("\nAll SupportIQ artifacts are in models/ - app.py can now load")
    print("them directly. No retraining or FAISS rebuilding happens at demo time.")


if __name__ == "__main__":
    main()
