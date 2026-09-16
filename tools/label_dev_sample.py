
# This script classifies conversations into 11 intent labels using Claude.
# These AI-generated labels are called:
#
#     PSEUDO-LABELS
#
# IMPORTANT:
# These are NOT our final ground-truth labels.
# Our 200 manually labelled Golden Set is still the real
# evaluation set.

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

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


# ============================================================
# CLAUDE CLIENT
# ============================================================

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY is missing from .env")

client = Anthropic(api_key=api_key)
model = "claude-3-5-sonnet-20241022"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

df = pd.read_csv(DEV_PATH)
golden = pd.read_csv(GOLDEN_PATH)

# Remove the 200 Golden Set examples to prevent data leakage.
golden_ids = set(golden["root_tweet_id"])
df = df[~df["root_tweet_id"].isin(golden_ids)].copy()

# Select the same reproducible 10,000-example development sample.
dev_sample = df.sample(n=10000, random_state=42).copy()

print(f"Development sample: {len(dev_sample)}")
print(f"Output file: {OUTPUT_PATH}")


# ============================================================
# RESUME PREVIOUS PROGRESS
# ============================================================

# If the script was stopped earlier, continue from the
# conversations that have not been labelled yet.
if os.path.exists(OUTPUT_PATH):

    results = pd.read_csv(OUTPUT_PATH)

    completed_ids = set(results["root_tweet_id"])

    dev_sample = dev_sample[
        ~dev_sample["root_tweet_id"].isin(completed_ids)
    ].copy()

    print(f"Already labeled: {len(completed_ids)}")
    print(f"Remaining: {len(dev_sample)}")

else:

    results = pd.DataFrame(
        columns=[
            "root_tweet_id",
            "thread_size",
            "conversation",
            "pseudo_intent",
        ]
    )


# ============================================================
# PREPARE INTENT LIST FOR THE PROMPT
# ============================================================

intent_text = "\n".join(
    f"{i}. {intent}" for i, intent in enumerate(INTENTS, start=1)
)


# ============================================================
# CLASSIFY ONE CONVERSATION
# ============================================================

def classify(conversation):

    prompt = f"""
You are an intent classifier for Amazon customer support.

Classify the CUSTOMER'S primary problem into exactly ONE of these intents:

{intent_text}

Rules:
- Focus on the customer's messages, not Amazon's responses.
- Identify the main customer problem in the conversation.
- If there are multiple customer issues, focus on the latest unresolved issue.
- Choose the most specific applicable intent.
- Do not classify based only on keywords.
- Do not invent information.
- If there is insufficient evidence, choose Other / Unclear.

Return valid JSON only:

{{"intent": "Delivery Issue"}}

The value of "intent" MUST exactly match one of the 11 intent names above.
Do not include any explanation.

Conversation:
{conversation}
"""

    response = client.messages.create(
        model=model,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    # Convert the model's response into a Python object.
    text = response.content[0].text.strip()
    data = json.loads(text)

    intent = data.get("intent")

    # Reject anything outside our predefined taxonomy.
    if intent not in VALID_INTENTS:
        raise ValueError(f"Invalid intent returned: {intent}")

    return intent


# ============================================================
# LABEL ALL 10,000 CONVERSATIONS
# ============================================================

for count, (_, row) in enumerate(dev_sample.iterrows(), start=1):

    conversation_id = row["root_tweet_id"]

    print(
        f"\n[{count}/{len(dev_sample)}] "
        f"Conversation {conversation_id}"
    )

    try:

        intent = classify(row["conversation"])

        result = {
            "root_tweet_id": conversation_id,
            "thread_size": row["thread_size"],
            "conversation": row["conversation"],
            "pseudo_intent": intent,
        }

        results = pd.concat(
            [results, pd.DataFrame([result])],
            ignore_index=True,
        )

        # Save after every prediction so progress is not lost.
        results.to_csv(OUTPUT_PATH, index=False)

        print(f"Prediction: {intent}")

    except Exception as e:

        print(f"ERROR: {e}")
        print("Waiting 5 seconds before retry...")

        time.sleep(5)

        try:

            intent = classify(row["conversation"])

            result = {
                "root_tweet_id": conversation_id,
                "thread_size": row["thread_size"],
                "conversation": row["conversation"],
                "pseudo_intent": intent,
            }

            results = pd.concat(
                [results, pd.DataFrame([result])],
                ignore_index=True,
            )

            results.to_csv(OUTPUT_PATH, index=False)

            print(f"Retry successful: {intent}")

        except Exception as retry_error:

            print(f"Retry failed: {retry_error}")

            # Skip this conversation and continue with the next one.
            continue


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("LABELING COMPLETE")
print("=" * 70)

print(f"Total labeled: {len(results)}")
print(f"Saved to: {OUTPUT_PATH}")