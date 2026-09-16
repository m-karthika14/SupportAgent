# ============================================================
# SUPPORTIQ - MANUAL DATA ANNOTATION TOOL
# ============================================================

import sys
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_PATH = "data/golden/golden_set.csv"
DEFAULT_COLUMN = "intent"

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


# ============================================================
# SHOW ONE EXAMPLE
# ============================================================

def show_example(row, number, label_column):

    print("\n" + "=" * 70)
    print(f"EXAMPLE {number}")
    print("=" * 70)

    print("\n" + row["conversation"])

    print("\n" + "-" * 70)

    # Show the AI label when this is an audit dataset.
    if label_column == "human_label" and "pseudo_intent" in row:
        print(f"AI LABEL: {row['pseudo_intent']}")

    print("\nChoose intent:")

    for i, intent in enumerate(INTENTS, start=1):
        print(f"{i}. {intent}")

    print("-" * 70)


# ============================================================
# MAIN ANNOTATION FUNCTION
# ============================================================

def annotate(file_path, label_column):

    df = pd.read_csv(file_path)

    # Make sure the label column can store text.
    if label_column not in df.columns:
        df[label_column] = ""

    df[label_column] = df[label_column].fillna("").astype("object")

    print(f"\nDataset loaded: {len(df)} examples")
    print(f"File: {file_path}")
    print(f"Label column: {label_column}")

    print("\nStarting manual annotation...")
    print("Enter the number corresponding to the correct intent.")
    print("Enter 'q' to quit and save progress.\n")

    for index, row in df.iterrows():

        # Skip examples that have already been labelled.
        if str(row[label_column]).strip():
            continue

        show_example(
            row,
            index + 1,
            label_column
        )

        while True:

            choice = input("\nYour choice: ").strip()

            # Stop and save progress.
            if choice.lower() == "q":

                df.to_csv(file_path, index=False)

                print("\nProgress saved. Exiting...")
                return

            # Check whether the user entered a valid number.
            if choice.isdigit() and 1 <= int(choice) <= len(INTENTS):

                intent = INTENTS[int(choice) - 1]

                df.at[index, label_column] = intent

                print(f"✓ Saved: {intent}")

                break

            print(
                "Invalid choice. "
                "Please enter a number from 1 to 11."
            )

        # Save after every annotation.
        df.to_csv(file_path, index=False)

    print("\n" + "=" * 70)
    print("ANNOTATION COMPLETE!")
    print("=" * 70)

    print(f"Annotated examples: {len(df)}")
    print(f"Saved to: {file_path}")


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

if __name__ == "__main__":

    # If no arguments are provided, annotate the Golden Set.
    #
    # python tools/annotate_golden.py
    #
    # If arguments are provided, use them instead.
    #
    # python tools/annotate_golden.py FILE COLUMN

    if len(sys.argv) == 1:

        file_path = DEFAULT_PATH
        label_column = DEFAULT_COLUMN

    elif len(sys.argv) == 3:

        file_path = sys.argv[1]
        label_column = sys.argv[2]

    else:

        print(
            "\nUsage:\n"
            "  python tools/annotate_golden.py\n\n"
            "or:\n"
            "  python tools/annotate_golden.py FILE COLUMN\n"
        )

        sys.exit(1)

    annotate(file_path, label_column)