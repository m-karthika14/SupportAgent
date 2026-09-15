import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_LABEL_MODEL")

print("API key loaded:", bool(api_key))
print("Model:", model)

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model=model,
    contents="Classify this customer support message in one short sentence: "
             "\"Where is my Amazon order? It was supposed to arrive yesterday.\""
)

print("\nGemini response:")
print(response.text)