"""
SupportIQAgent - the orchestrator that ties the four pipeline stages
together: classify -> retrieve -> escalate -> generate.

This file REPLACES the previous placeholder version, which called
classify_intent(), retrieve_cases(), escalation_policy(), and
generate_reply() without ever defining or importing any of them (those
functions only existed as in-memory variables inside
notebooks/06_retrieval_evaluation.ipynb while its kernel was running, so
running this file standalone raised a NameError). The logic below is
unchanged from that notebook's run_agent() function - only the missing
imports have been wired up, pointing at the extracted modules:

    classify_intent()    -> src/pipeline/classifier.py  (IntentClassifier)
    retrieve_cases()     -> src/pipeline/retriever.py   (CaseRetriever)
    generate_reply()     -> src/pipeline/generator.py
    escalation_policy()  -> src/pipeline/escalation.py
"""

from src.pipeline.classifier import IntentClassifier
from src.pipeline.retriever import CaseRetriever
from src.pipeline.generator import generate_reply
from src.pipeline.escalation import escalation_policy


class SupportIQAgent:
    """
    Loads the classifier + retriever ONCE (they hold the fitted
    TF-IDF/Logistic Regression model, the FAISS index, and the MiniLM
    embedding model), then answers as many run() calls as needed without
    reloading anything from disk in between - this is what makes the
    Streamlit demo responsive after the first (slower) startup.
    """

    def __init__(self):
        # Both of these load pre-built artifacts from models/ (produced
        # once by tools/build_artifacts.py) - no training or FAISS index
        # building happens here, only loading.
        self.classifier = IntentClassifier()
        self.retriever = CaseRetriever()

    def run(self, customer_message: str, top_k: int = 5) -> dict:
        """
        Run the full SupportIQ pipeline on one customer message and
        return a complete trace of every stage's output, so the demo UI
        can display each step (not just the final reply).
        """

        # --- 1. Classify intent ------------------------------------------
        # Hybrid TF-IDF + MiniLM Logistic Regression classifier (see
        # notebooks/05_classifier_evaluation.ipynb for how it was trained
        # and validated).
        classification = self.classifier.classify(customer_message)
        intent = classification["intent"]
        confidence = classification["confidence"]

        # --- 2. Retrieve historical evidence -------------------------------
        # MiniLM embeddings + FAISS nearest-neighbor search over the 10K
        # dev corpus (see notebooks/06_retrieval_evaluation.ipynb).
        retrieved_cases = self.retriever.retrieve(customer_message, top_k=top_k)

        # --- 3. Decide escalation BEFORE generation -------------------------
        # Escalation is decided before the reply is drafted (same order
        # as the notebook) so a human-routed case doesn't need to wait on
        # an LLM call at all in a stricter production setup - here the
        # reply is still generated either way so the demo can show both.
        escalation = escalation_policy(
            customer_message=customer_message,
            intent=intent,
            retrieved_cases=retrieved_cases,
        )

        # --- 4. Generate a grounded reply ------------------------------------
        # Groq LLM call, grounded ONLY in the retrieved historical cases.
        reply = generate_reply(
            customer_message=customer_message,
            intent=intent,
            retrieved_cases=retrieved_cases,
        )

        # --- 5. Return the complete trace ------------------------------------
        # Same shape as run_agent() in notebooks/06, so any existing
        # evaluation code that consumes this dict keeps working unchanged.
        return {
            "customer_message": customer_message,
            "intent": intent,
            "confidence": confidence,
            "retrieved_cases": retrieved_cases,
            "reply": reply,
            "escalation_decision": escalation["decision"],
            "escalation_reason": escalation["reason"],
        }
