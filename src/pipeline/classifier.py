"""
Intent classifier for SupportIQ.

This module wraps the Hybrid (TF-IDF + MiniLM) Logistic Regression
classifier that was developed and validated in
notebooks/05_classifier_evaluation.ipynb, where it reported:

    58.0% accuracy / 0.503 macro F1  on the 2K pseudo-label test set
    54.0% accuracy / 0.484 macro F1  on the 200-example Golden Set

The actual TRAINING code (the code that produces those numbers) lives in
tools/build_artifacts.py, which fits the TF-IDF vectorizer + Logistic
Regression classifier once and saves them to models/classifier.pkl. This
module does NOT retrain anything - it only loads the already-fitted
pieces and reproduces the exact classify_intent() logic from
notebooks/05 & 06, so a prediction made here matches what the notebook
would have predicted for the same input.
"""

from pathlib import Path

import joblib
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

# models/ lives at the repo root, two directories up from this file
# (src/pipeline/classifier.py -> src/pipeline -> src -> repo root -> models).
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class IntentClassifier:
    """Loads the fitted TF-IDF vectorizer + Hybrid classifier once (in
    __init__), then classifies as many messages as needed via classify()
    without re-loading anything from disk in between calls."""

    def __init__(self):
        # classifier.pkl holds a dict with two fitted objects:
        #   - tfidf_vectorizer   : sklearn TfidfVectorizer, already fit()
        #   - hybrid_classifier  : sklearn LogisticRegression, already fit()
        # Both were saved together by tools/build_artifacts.py so they can
        # never accidentally get out of sync with each other.
        artifact = joblib.load(MODELS_DIR / "classifier.pkl")
        self.tfidf_vectorizer = artifact["tfidf_vectorizer"]
        self.hybrid_classifier = artifact["hybrid_classifier"]

        # The same pretrained MiniLM model used everywhere else in the
        # pipeline (classifier features here, retrieval embeddings in
        # retriever.py). Loading it fresh here (rather than saving/loading
        # the model weights ourselves) exactly matches how the notebooks
        # obtain it, and sentence-transformers caches the downloaded
        # weights locally after the first run, so this is fast after the
        # very first startup.
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def classify(self, customer_message: str) -> dict:
        """
        Reproduces classify_intent() from notebooks/05 & 06 exactly:

            1. Build TF-IDF features for the message using the vectorizer
               that was FIT on the training split (never re-fit here).
            2. Build a MiniLM semantic embedding for the same message.
            3. Concatenate (hstack) both feature types into one combined
               "hybrid" feature vector - the same feature space the
               classifier was trained on.
            4. Predict the intent label and report the model's confidence
               (the highest predicted-class probability).

        Returns a dict: {"intent": <str>, "confidence": <float 0-1>}
        """

        # --- Step 1: TF-IDF (lexical) features -------------------------
        # .transform(), not .fit_transform() - the vectorizer's vocabulary
        # is frozen from training; a new message just gets projected into
        # that existing vocabulary.
        tfidf_features = self.tfidf_vectorizer.transform([customer_message])

        # --- Step 2: MiniLM (semantic) features -------------------------
        embedding_features = self.embedding_model.encode(
            [customer_message],
            normalize_embeddings=True,
        )

        # --- Step 3: combine both feature types --------------------------
        # Must match training exactly: TF-IDF first, then the dense MiniLM
        # embedding converted to a sparse matrix, concatenated column-wise.
        hybrid_features = hstack([
            tfidf_features,
            csr_matrix(embedding_features),
        ])

        # --- Step 4: predict intent + confidence -------------------------
        predicted_intent = self.hybrid_classifier.predict(hybrid_features)[0]

        # predict_proba returns one probability per class; the highest one
        # is how confident the model is in its own top prediction.
        probabilities = self.hybrid_classifier.predict_proba(hybrid_features)[0]
        confidence = float(probabilities.max())

        return {
            "intent": predicted_intent,
            "confidence": confidence,
        }
