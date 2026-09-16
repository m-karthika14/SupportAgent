"""
Historical case retriever for SupportIQ.

This module wraps the FAISS retrieval index that was built in
notebooks/06_retrieval_evaluation.ipynb over the full 10,000-conversation
dev_pseudo_labeled corpus (Recall@5 = 70.0%, Recall@10 = 79.0%, MRR =
0.515 against the 200-example Golden Set, as measured in that notebook).

The index itself (models/faiss_index.bin) and its row-aligned metadata
(models/corpus_metadata.csv) are built ONCE by tools/build_artifacts.py.
This module only loads and searches them - it never re-embeds the 10K
corpus or rebuilds the index, which is what makes app startup fast.
"""

from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class CaseRetriever:
    """Loads the FAISS index + its metadata once (in __init__), then
    answers as many retrieve() calls as needed without touching disk
    again."""

    def __init__(self):
        # A flat inner-product FAISS index (IndexFlatIP). Because every
        # vector going in was L2-normalized (see build_artifacts.py and
        # the retrieve() method below), inner product is mathematically
        # equivalent to cosine similarity - this is just an exact,
        # brute-force nearest-neighbor search over 10,000 vectors, which
        # is small enough that an approximate index isn't needed.
        self.index = faiss.read_index(str(MODELS_DIR / "faiss_index.bin"))

        # Row i of this dataframe corresponds EXACTLY to row i (vector i)
        # of the FAISS index above - both were built from the same
        # dataframe in the same order inside build_artifacts.py. Never
        # sort, filter, or reload this independently of the index, or the
        # rank-to-conversation mapping below will silently point at the
        # wrong historical case.
        self.corpus = pd.read_csv(MODELS_DIR / "corpus_metadata.csv")

        # Same pretrained MiniLM model used to build the index in the
        # first place - the query embedding must live in the exact same
        # vector space as the indexed embeddings for similarity search to
        # be meaningful.
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query_text: str, top_k: int = 5) -> list:
        """
        Reproduces retrieve_cases() from notebooks/06 exactly:

            1. Embed the query text with MiniLM (L2-normalized, same as
               every embedding already sitting in the FAISS index).
            2. Search the FAISS index for the top_k nearest vectors
               (highest inner product == highest cosine similarity).
            3. Look up each hit's row in corpus_metadata.csv (by FAISS
               row number) to recover its original conversation + intent.
            4. Return a ranked list of dicts, rank 1 = most similar.

        Returns: [{"rank": int, "similarity": float, "intent": str,
                   "conversation": str}, ...]
        """

        # --- Step 1: embed the query -------------------------------------
        query_embedding = self.embedding_model.encode(
            [query_text],
            normalize_embeddings=True,
        ).astype("float32")

        # --- Step 2: search the FAISS index -------------------------------
        # scores/indices are both shape (1, top_k) since we searched with
        # a single query vector; [0] unwraps that outer batch dimension.
        scores, indices = self.index.search(query_embedding, top_k)

        # --- Step 3 & 4: look up each hit and build the ranked result ----
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            case = self.corpus.iloc[idx]

            results.append({
                "rank": rank,
                "similarity": float(score),
                "intent": case["pseudo_intent"],
                "conversation": case["conversation"],
            })

        return results
