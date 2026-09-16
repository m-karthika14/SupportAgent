#!/usr/bin/env python3
"""Process all remaining conversations in batches."""
import json
import pandas as pd
import os

# Intent list
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

def quick_classify(text):
    """Heuristic classification."""
    text_lower = text.lower()

    if any(k in text_lower for k in ["delivered", "delivery", "arrive", "arrival", "shipped", "pending", "on the way", "delayed", "missing", "late", "not arrive", "still not", "waiting for", "prime", "2 days", "next day"]):
        if any(k in text_lower for k in ["damage", "damaged", "broken", "wrong", "incorrect", "different", "missing item"]):
            return "Damaged / Wrong / Missing Item"
        return "Delivery Issue"

    if any(k in text_lower for k in ["return", "refund", "money back", "reimburs"]):
        return "Return / Refund"

    if any(k in text_lower for k in ["charge", "billing", "payment", "card", "credit", "bank", "pay", "currency", "price"]):
        if "delivery" not in text_lower:
            return "Payment / Billing"

    if any(k in text_lower for k in ["account", "password", "email", "hacked", "login", "security"]):
        return "Account / Access / Security"

    if any(k in text_lower for k in ["cancel", "order", "purchase", "buy", "quantity"]):
        if "refund" not in text_lower:
            return "Order Management"

    if any(k in text_lower for k in ["membership", "subscription"]):
        if "delivery" not in text_lower:
            return "Prime Membership"

    if any(k in text_lower for k in ["video", "ebook", "kindle", "digital", "download", "book", "show", "music"]):
        return "Digital Content"

    if any(k in text_lower for k in ["product", "device", "app", "echo", "technical", "bug"]):
        return "Product / Device Support"

    if any(k in text_lower for k in ["discount", "coupon", "gift card", "cashback", "sale"]):
        return "Promotion / Gift Card / Credit"

    return "Other / Unclear"

# Load data
print("Loading data...")
df = pd.read_csv("data/processed/amazon_conversations.csv")
golden = pd.read_csv("data/golden/golden_set.csv")
golden_ids = set(golden["root_tweet_id"])
df = df[~df["root_tweet_id"].isin(golden_ids)].copy()
dev_sample = df.sample(n=10000, random_state=42).copy()

# Load existing results
output_path = "data/processed/dev_pseudo_labeled.csv"
if os.path.exists(output_path):
    results = pd.read_csv(output_path)
else:
    results = pd.DataFrame(columns=["root_tweet_id", "thread_size", "conversation", "pseudo_intent"])

completed = set(results.get("root_tweet_id", []))
remaining = dev_sample[~dev_sample["root_tweet_id"].isin(completed)].copy()

print(f"Completed: {len(results)}/10000")
print(f"Remaining: {len(remaining)}")
print("\nProcessing remaining conversations...")

# Classify all remaining
for i, (_, row) in enumerate(remaining.iterrows(), start=1):
    intent = quick_classify(row["conversation"])
    if intent not in VALID_INTENTS:
        intent = "Other / Unclear"

    result = {
        "root_tweet_id": row["root_tweet_id"],
        "thread_size": row["thread_size"],
        "conversation": row["conversation"],
        "pseudo_intent": intent,
    }

    results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)

    # Save every 100
    if i % 100 == 0:
        results.to_csv(output_path, index=False)
        pct = (len(results) / 10000) * 100
        print(f"[{len(results)}/10000] ({pct:.1f}%)")

# Final save
results.to_csv(output_path, index=False)

print("\n" + "="*70)
print("ALL COMPLETE")
print("="*70)
print(f"Total classified: {len(results)}/10000")
print(f"File: {output_path}")
print("\nIntent distribution:")
print(results["pseudo_intent"].value_counts().to_string())
