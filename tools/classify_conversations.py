#!/usr/bin/env python3
"""
Classify conversations into intent labels using Claude.
"""

import os
import pandas as pd
import json
from anthropic import Anthropic

# Paths
DEV_PATH = "data/processed/amazon_conversations.csv"
GOLDEN_PATH = "data/golden/golden_set.csv"
OUTPUT_PATH = "data/processed/dev_pseudo_labeled.csv"

INTENTS = [
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

VALID_INTENTS = set(INTENTS)

# Initialize Claude client
client = Anthropic()

# Load data
print("Loading data...")
df = pd.read_csv(DEV_PATH)
golden = pd.read_csv(GOLDEN_PATH)

# Remove golden set to prevent data leakage
golden_ids = set(golden["root_tweet_id"])
df = df[~df["root_tweet_id"].isin(golden_ids)].copy()

# Get reproducible 10k sample
dev_sample = df.sample(n=min(10000, len(df)), random_state=42).copy()

print(f"Total dev data: {len(df)}")
print(f"Classifying: {len(dev_sample)} conversations")
print(f"Output: {OUTPUT_PATH}\n")

# Check for existing progress
if os.path.exists(OUTPUT_PATH):
    results = pd.read_csv(OUTPUT_PATH)
    completed_ids = set(results["root_tweet_id"])
    dev_sample = dev_sample[~dev_sample["root_tweet_id"].isin(completed_ids)].copy()
    print(f"Already classified: {len(completed_ids)}")
    print(f"Remaining: {len(dev_sample)}\n")
else:
    results = pd.DataFrame(columns=["root_tweet_id", "thread_size", "conversation", "pseudo_intent"])

# Prepare intent list for prompt
intent_text = "\n".join(f"{i}. {intent}" for i, intent in enumerate(INTENTS, start=1))


def classify(conversation):
    """Classify a single conversation using Claude."""
    prompt = f"""You are an intent classifier for customer support conversations.

Classify the CUSTOMER'S primary problem into exactly ONE of these intents:

{intent_text}

Rules:
- Focus on the customer's messages, not support responses.
- Identify the main unresolved customer problem.
- If multiple issues exist, focus on the latest one.
- Choose the most specific applicable intent.
- Return valid JSON only with no explanation.

Conversation:
{conversation}"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Extract JSON
    if "{{" in text or "{" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    data = json.loads(text)
    intent = data.get("intent")

    if intent not in VALID_INTENTS:
        raise ValueError(f"Invalid intent: {intent}")

    return intent


# Classify all conversations
for i, (_, row) in enumerate(dev_sample.iterrows(), start=1):
    conv_id = row["root_tweet_id"]

    try:
        intent = classify(row["conversation"])

        result = {
            "root_tweet_id": conv_id,
            "thread_size": row["thread_size"],
            "conversation": row["conversation"],
            "pseudo_intent": intent,
        }

        results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
        results.to_csv(OUTPUT_PATH, index=False)

        if i % 10 == 0:
            print(f"[{i}/{len(dev_sample)}] Classified: {intent}")

    except Exception as e:
        print(f"[{i}/{len(dev_sample)}] Error on {conv_id}: {e}")
        continue

print("\n" + "="*70)
print("CLASSIFICATION COMPLETE")
print("="*70)
print(f"Total classified: {len(results)}")
print(f"Saved to: {OUTPUT_PATH}")
