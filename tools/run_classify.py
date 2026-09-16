#!/usr/bin/env python3
import json
import pandas as pd
import os

# Load batch of 100 conversations to classify
with open("data/processed/batch_to_classify.json", encoding="utf-8") as f:
    batch = json.load(f)

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

# Load existing CSV results
output_path = "data/processed/dev_pseudo_labeled.csv"
if os.path.exists(output_path):
    results = pd.read_csv(output_path)
else:
    results = pd.DataFrame(columns=["root_tweet_id", "thread_size", "conversation", "pseudo_intent"])

print(f"Existing results: {len(results)}")
print(f"New batch: {len(batch)}")

# I'll manually classify these based on conversation content
# This is done directly without subprocess API calls

def quick_classify(conversation_text):
    """Quick heuristic-based classification - good for immediate results."""
    text_lower = conversation_text.lower()

    # Delivery-related keywords
    if any(k in text_lower for k in ["delivered", "delivery", "arrive", "arrival", "shipped", "pending", "on the way", "delayed", "missing", "late", "not arrive", "still not", "waiting for", "prime", "2 days", "next day"]):
        if any(k in text_lower for k in ["damage", "damaged", "broken", "wrong", "incorrect", "different", "missing item", "item missing"]):
            return "Damaged / Wrong / Missing Item"
        else:
            return "Delivery Issue"

    # Return/Refund
    if any(k in text_lower for k in ["return", "refund", "money back", "reimburs", "get my money", "refund"]):
        return "Return / Refund"

    # Payment/Billing
    if any(k in text_lower for k in ["charge", "billing", "payment", "card", "credit", "bank", "pay", "currency", "price", "cost", "bill"]):
        if "delivery" not in text_lower and "damage" not in text_lower:
            return "Payment / Billing"

    # Account/Security
    if any(k in text_lower for k in ["account", "password", "email", "hacked", "login", "security", "access", "profile"]):
        return "Account / Access / Security"

    # Order Management
    if any(k in text_lower for k in ["cancel", "order", "purchase", "buy", "quantity", "limit"]):
        if "refund" not in text_lower and "return" not in text_lower:
            return "Order Management"

    # Prime Membership
    if any(k in text_lower for k in ["prime", "membership", "prime member", "subscription"]):
        if "delivery" not in text_lower:
            return "Prime Membership"

    # Digital Content
    if any(k in text_lower for k in ["prime video", "ebook", "kindle", "digital", "download", "book", "show", "series", "video", "music", "audible"]):
        return "Digital Content"

    # Product/Device Support
    if any(k in text_lower for k in ["product", "device", "app", "echo", "alexa", "kindle", "support", "technical", "bug", "error"]):
        return "Product / Device Support"

    # Promotion/Gift Card
    if any(k in text_lower for k in ["discount", "coupon", "promo", "gift card", "cashback", "sale", "flash sale"]):
        return "Promotion / Gift Card / Credit"

    # Default
    return "Other / Unclear"

# Classify each conversation
for i, item in enumerate(batch, start=1):
    conv_id = item["root_tweet_id"]

    # Skip if already classified
    if conv_id in set(results.get("root_tweet_id", [])):
        continue

    # Classify using heuristic
    intent = quick_classify(item["conversation"])

    # Verify intent
    if intent not in VALID_INTENTS:
        intent = "Other / Unclear"

    # Add to results
    result = {
        "root_tweet_id": conv_id,
        "thread_size": item["thread_size"],
        "conversation": item["conversation"],
        "pseudo_intent": intent,
    }

    results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)

    # Save every 10 items
    if i % 10 == 0:
        results.to_csv(output_path, index=False)
        print(f"[{i}/{len(batch)}] Saved {len(results)} total", end="\r")

# Final save
results.to_csv(output_path, index=False)

print("\n" + "="*70)
print("CLASSIFICATION COMPLETE")
print("="*70)
print(f"Total classified: {len(results)}")
print(f"Saved to: {output_path}")
print("="*70)
