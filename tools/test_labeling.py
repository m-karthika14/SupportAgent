import os
import pandas as pd

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GOLDEN_PATH = "data/golden/golden_set.csv"
DEV_PATH = "data/processed/amazon_conversations.csv"

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

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL")

print("API key loaded:", bool(api_key))
print("Model:", model)

client = Groq(api_key=api_key)

# Load data
df = pd.read_csv(DEV_PATH)

# Load Golden Set IDs so we never accidentally use them
golden = pd.read_csv(GOLDEN_PATH)
golden_ids = set(golden["root_tweet_id"])

# Remove Golden Set conversations
df = df[~df["root_tweet_id"].isin(golden_ids)]

# Reproduce our 10K development sample
dev_sample = df.sample(
    n=10000,
    random_state=42
).copy()

# Test only 5 examples first
test_sample = dev_sample.head(5)

intent_text = "\n".join(
    f"{i}. {intent}" for i, intent in enumerate(INTENTS, start=1)
)

for _, row in test_sample.iterrows():

    prompt = f"""
You are an intent classifier for Amazon customer support.

Classify the CUSTOMER'S primary problem into exactly ONE of these intents:

{intent_text}

Rules:
- Focus on the customer's messages, not Amazon's responses.
- If there are multiple customer issues, classify the latest unresolved customer issue.
- Choose the most specific applicable intent.
- Do not classify based only on keywords.
- Do not invent information.
- If there is insufficient evidence, choose Other / Unclear.

Return valid JSON only in this exact format:
{{"intent": "Delivery Issue"}}

The value of "intent" MUST be exactly one of the 11 intent names above.
Do not include any explanation.

Conversation:
{row["conversation"]}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={
            "type": "json_object"
        },
    )

    predicted = response.choices[0].message.content.strip()

    print("\n" + "=" * 70)
    print("Conversation ID:", row["root_tweet_id"])
    print("\n" + row["conversation"])
    print("\nMODEL PREDICTION:")
    print(predicted)