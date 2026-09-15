#To hand label the golden dataset(or any just change the name) to the below intents run the script below


import pandas as pd

GOLDEN_PATH = "data/golden/golden_set.csv"


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


def show_example(row, number):
    print("\n" + "=" * 70)
    print(f"EXAMPLE {number}")
    print("=" * 70)

    conversation = row["conversation"]

    print("\n" + conversation)

    print("\n" + "-" * 70)
    print("Choose intent:")
    
    for i, intent in enumerate(INTENTS, start=1):
        print(f"{i}. {intent}")

    print("-" * 70)


def main():
    df = pd.read_csv(GOLDEN_PATH)
    df["intent"] = df["intent"].astype("object")

    print(f"\nGolden set loaded: {len(df)} examples")

    print("\nStarting manual annotation...")
    print("Enter the number corresponding to the correct intent.")
    print("Enter 'q' to quit.\n")

    for index, row in df.iterrows():

        show_example(row, index + 1)

        while True:
            choice = input("\nYour choice: ").strip()

            if choice.lower() == "q":
                df.to_csv(GOLDEN_PATH, index=False)
                print("\nProgress saved. Exiting...")
                return

            if choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
                intent = INTENTS[int(choice) - 1]
                df.at[index, "intent"] = intent

                print(f"✓ Saved: {intent}")
                break

            print("Invalid choice. Please enter a number from 1 to 11.")

        # Save after every annotation
        df.to_csv(GOLDEN_PATH, index=False)

    print("\n" + "=" * 70)
    print("ANNOTATION COMPLETE!")
    print("=" * 70)
    print(f"Annotated examples: {len(df)}")
    print(f"Saved to: {GOLDEN_PATH}")


if __name__ == "__main__":
    main()