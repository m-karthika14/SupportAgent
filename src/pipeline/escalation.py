"""
Escalation policy for SupportIQ.

A straight extraction of escalation_policy() from
notebooks/06_retrieval_evaluation.ipynb. It is a simple, explainable
rule-based gate (not a learned model) that decides whether a case should
be auto-handled by the generated reply or escalated to a human agent.
The rules and thresholds are unchanged from the notebook.
"""


def escalation_policy(customer_message: str, intent: str, retrieved_cases: list) -> dict:
    """
    Decide AUTO-HANDLE vs ESCALATE for a customer message.

    Three escalation triggers are checked, in order:
        1. High-risk keywords (security/financial/legal/safety terms) -
           these are cases where an AI-drafted reply could cause real
           harm if wrong, so they always go to a human.
        2. The customer explicitly asks for a human agent.
        3. The best-matching retrieved case is too dissimilar
           (similarity < 0.50) - if there's no good historical precedent,
           the generated reply is unlikely to be well-grounded, so it's
           safer to escalate than to guess.

    If none of those trigger, the case is auto-handled.

    Returns: {"decision": "ESCALATE" | "AUTO-HANDLE", "reason": str}
    """

    text = customer_message.lower()

    # ------------------------------------------------------------------
    # High-risk account / financial / legal / safety signals. Any one of
    # these words appearing anywhere in the message is enough to
    # escalate, regardless of the retrieved evidence's similarity.
    # ------------------------------------------------------------------
    high_risk_keywords = [
        "hacked",
        "hack",
        "account stolen",
        "unauthorized",
        "unauthorised",
        "without my permission",
        "fraud",
        "scam",
        "stolen",
        "chargeback",
        "lawsuit",
        "legal",
        "police",
        "danger",
        "injured",
        "unsafe",
    ]

    # ------------------------------------------------------------------
    # The customer directly asking to speak with a person, rather than an
    # automated system.
    # ------------------------------------------------------------------
    human_request_keywords = [
        "human",
        "agent",
        "representative",
        "real person",
        "speak to someone",
        "talk to someone",
        "speak with someone",
    ]

    # The strongest evidence match we found for this message - used as a
    # proxy for "how confident should we be that the retrieved cases are
    # actually relevant to this customer's situation".
    max_similarity = max(
        [case["similarity"] for case in retrieved_cases],
        default=0.0,
    )

    # --- Trigger 1: high-risk issue --------------------------------------
    if any(keyword in text for keyword in high_risk_keywords):
        return {
            "decision": "ESCALATE",
            "reason": "High-risk account, financial, legal, or safety-related issue.",
        }

    # --- Trigger 2: explicit human request -------------------------------
    if any(keyword in text for keyword in human_request_keywords):
        return {
            "decision": "ESCALATE",
            "reason": "Customer explicitly requested human assistance.",
        }

    # --- Trigger 3: insufficient historical evidence ---------------------
    if max_similarity < 0.50:
        return {
            "decision": "ESCALATE",
            "reason": "Insufficient similarity to historical support cases.",
        }

    # --- No trigger fired: safe to auto-handle ---------------------------
    return {
        "decision": "AUTO-HANDLE",
        "reason": "No escalation trigger detected and relevant historical evidence was retrieved.",
    }
