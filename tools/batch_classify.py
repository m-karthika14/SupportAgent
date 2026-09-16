#!/usr/bin/env python3
"""Batch classify conversations using Claude API."""

import json
import pandas as pd
import os
import sys

# Load batch
with open("data/processed/batch_to_classify.json", encoding="utf-8") as f:
    batch = json.load(f)

# Load existing results
output_path = "data/processed/dev_pseudo_labeled.csv"
if os.path.exists(output_path):
    results = pd.read_csv(output_path)
else:
    results = pd.DataFrame(columns=["root_tweet_id", "thread_size", "conversation", "pseudo_intent"])

intents = [
    "Delivery Issue",
    "Damaged / Wrong / Missing Item",
    "Order Management",
    "Return / Refund",
    "Payment / Billing",
    "Account / Access / Security",
    "Prime Membership",
    "Digital Content",
    "Product / Device Support",
    "Promotion / Gift Card / Credit",
    "Other / Unclear",
]

VALID_INTENTS = set(intents)
intent_text = "\n".join(f"{i}. {intent}" for i, intent in enumerate(intents, start=1))

print(f"Classifying {len(batch)} conversations...")
print(f"Existing results: {len(results)}")

# Import Claude client
try:
    from anthropic import Anthropic
    client = Anthropic()
    print("Using Anthropic Claude API\n")
except ImportError:
    print("ERROR: anthropic SDK not installed")
    sys.exit(1)

def classify(text):
    """Classify using Claude."""
    prompt = f"""You are a customer support intent classifier.

Classify the customer's primary problem into exactly ONE intent:

{intent_text}

Rules:
- Focus on the customer's messages, not responses
- Identify the main unresolved problem
- Choose the most specific intent
- Return JSON only: {{"intent": "Delivery Issue"}}

Conversation:
{text}"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )

        text_resp = response.content[0].text.strip()

        # Extract JSON
        if "{" in text_resp:
            start = text_resp.find("{")
            end = text_resp.rfind("}") + 1
            if start >= 0 and end > start:
                text_resp = text_resp[start:end]

        data = json.loads(text_resp)
        intent = data.get("intent")

        if intent not in VALID_INTENTS:
            return "Other / Unclear"  # Default fallback

        return intent

    except Exception as e:
        print(f"    Error: {e}")
        return "Other / Unclear"

# Classify each
for i, item in enumerate(batch, start=1):
    try:
        intent = classify(item["conversation"])

        result = {
            "root_tweet_id": item["root_tweet_id"],
            "thread_size": item["thread_size"],
            "conversation": item["conversation"],
            "pseudo_intent": intent,
        }

        results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)

        if i % 10 == 0:
            results.to_csv(output_path, index=False)
            print(f"[{i}/{ len(batch)}] Saved checkpoint - {intent}")
        else:
            print(f"[{i}/{len(batch)}] {intent}")

    except Exception as e:
        print(f"[{i}/{len(batch)}] Failed: {e}")
        continue

# Final save
results.to_csv(output_path, index=False)

print("\n" + "="*70)
print("COMPLETE")
print(f"Total classified: {len(results)}")
print(f"Saved to: {output_path}")
print("="*70)
