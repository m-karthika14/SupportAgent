"""
Grounded reply generator for SupportIQ.

This module is a straight extraction of the generate_reply() function
from notebooks/06_retrieval_evaluation.ipynb (the final version of that
cell, after the "STRICT RULES" prompt was tightened). It calls Groq's
OpenAI-compatible chat completions endpoint to draft a customer-facing
reply that is grounded ONLY in the historical cases retrieved by
CaseRetriever - it must not invent policies, links, refunds, or actions
that aren't supported by that evidence.

Nothing about the prompt, the rules, or the model call has been changed
here - this file only moves the function out of the notebook so it can
be imported by src/pipeline/agent.py and app.py.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Pick up GROQ_API_KEY / GROQ_MODEL from the repo's .env file. override=True
# matches the notebooks' behavior of always preferring .env values over any
# stale environment variables already set in the shell.
load_dotenv(override=True)

# Groq exposes an OpenAI-compatible API, so the standard `openai` client
# is reused here with Groq's base_url instead of OpenAI's - no separate
# Groq SDK is required.
_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Falls back to the same default model the notebooks use if GROQ_MODEL
# isn't set in .env.
_MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def generate_reply(customer_message: str, intent: str, retrieved_cases: list) -> str:
    """
    Draft a grounded customer support reply.

    Args:
        customer_message: the raw text the customer typed.
        intent: the intent label predicted by IntentClassifier.
        retrieved_cases: the ranked list of historical cases returned by
            CaseRetriever.retrieve() - each one is shown to the LLM as
            evidence it is allowed to reason from (and forbidden from
            going beyond).

    Returns: the plain-text reply, ready to show the customer.
    """

    # ------------------------------------------------------------------
    # Build a human-readable block of historical evidence for the prompt,
    # one CASE per retrieved conversation, in rank order.
    # ------------------------------------------------------------------
    evidence = ""
    for case in retrieved_cases:
        evidence += f"""
CASE {case['rank']}
Similarity: {case['similarity']:.3f}
Historical Intent: {case['intent']}

{case['conversation']}

---
"""

    # ------------------------------------------------------------------
    # The exact grounding prompt from notebooks/06 (final version). The
    # numbered rules exist specifically to stop the LLM from hallucinating
    # policies, refund amounts, or URLs that aren't actually in the
    # retrieved evidence.
    # ------------------------------------------------------------------
    prompt = f"""
You are an Amazon customer support assistant.

CUSTOMER MESSAGE:
{customer_message}

CLASSIFIED INTENT:
{intent}

HISTORICAL SUPPORT CASES:
{evidence}

Your task is to draft a concise customer-facing support reply grounded ONLY in the historical cases.

STRICT RULES:
1. Use the historical cases as evidence for how similar issues were handled.
2. Do NOT invent policies, refunds, replacements, links, phone numbers, guarantees,
   shipping methods, seller types, or other facts.
3. Do NOT create or reproduce URLs unless a URL is explicitly present in the
   historical evidence and is directly relevant.
4. Do NOT claim that Amazon has taken an action unless the evidence supports it.
5. If the evidence does not provide a specific resolution, acknowledge the issue
   and ask an appropriate clarifying question or direct the customer to support.
6. Be concise, polite, professional, and helpful.
7. Do not mention historical cases, retrieval, similarity, AI, or classification.
8. Return ONLY the customer-facing reply.

Draft the reply:
"""

    response = _client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a careful, evidence-grounded customer support assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        # NOTE: this 300-token cap is copied verbatim from the notebook.
        # During human evaluation of the 38-reply sample, several replies
        # came back cut off mid-sentence/mid-URL because of this limit -
        # worth raising (e.g. to 500) if that truncation becomes a problem
        # for the demo, but left unchanged here to match reported behavior.
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()
