#!/usr/bin/env python3
"""Audit pseudo-labels by manually labeling 100 sample conversations."""

import pandas as pd
import os

AUDIT_PATH = "data/golden/pseudo_label_audit_100.csv"

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


def show_example(row, number, total):
    """Display conversation and prompt for intent."""
    print("\n" + "=" * 70)
    print(f"EXAMPLE {number}/{total}")
    print("=" * 70)

    print(f"\nID: {row['root_tweet_id']}")
    print(f"Thread size: {row['thread_size']}")
    print("\nConversation:")
    print("-" * 70)
    print(row["conversation"])
    print("-" * 70)

    print(f"\nPSEUDO-LABEL (AI): {row['pseudo_intent']}")
    print("\nChoose correct intent (or 0 if pseudo-label is correct):")

    for i, intent in enumerate(INTENTS, start=1):
        print(f"{i}. {intent}")

    print("0. Pseudo-label is correct (accept as-is)")
    print("'s' or 'skip' to skip this one")
    print("'q' to save and quit")


def main():
    """Main audit loop."""
    if not os.path.exists(AUDIT_PATH):
        print(f"ERROR: {AUDIT_PATH} not found")
        return

    df = pd.read_csv(AUDIT_PATH)
    df["human_label"] = df["human_label"].astype("object")

    print(f"\n{'='*70}")
    print("PSEUDO-LABEL AUDIT")
    print(f"{'='*70}")
    print(f"Total examples: {len(df)}")

    # Find unannotated examples
    unannotated = df[df["human_label"].isna()]
    if len(unannotated) == 0:
        print("\nAll examples already annotated!")
        accuracy = (df["pseudo_intent"] == df["human_label"]).sum() / len(df)
        print(f"Pseudo-label accuracy: {accuracy:.1%}")
        return

    print(f"Unannotated: {len(unannotated)}/{len(df)}")
    print("\nInstructions:")
    print("- Type 0 to accept the AI label")
    print("- Type 1-11 to choose a different label")
    print("- Type 's' to skip this example")
    print("- Type 'q' to save and quit\n")

    for idx, (index, row) in enumerate(unannotated.iterrows(), start=1):
        show_example(row, idx, len(unannotated))

        while True:
            choice = input("\nYour choice: ").strip().lower()

            if choice == "q":
                df.to_csv(AUDIT_PATH, index=False)
                print("\n✓ Progress saved. Exiting...")
                return

            if choice in ["s", "skip"]:
                print("⊘ Skipped")
                break

            if choice == "0":
                human_label = row["pseudo_intent"]
                df.at[index, "human_label"] = human_label
                status = "✓ ACCEPT" if human_label == row["pseudo_intent"] else "✓ CORRECT"
                print(f"{status}: {human_label}")
                break

            if choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
                human_label = INTENTS[int(choice) - 1]
                df.at[index, "human_label"] = human_label
                match = "✓ MATCH" if human_label == row["pseudo_intent"] else "⚠ MISMATCH"
                print(f"{match}: {human_label}")
                break

            print("Invalid choice. Please enter 0-11, 's', or 'q'.")

        # Save after each annotation
        df.to_csv(AUDIT_PATH, index=False)

    # Final summary
    df = pd.read_csv(AUDIT_PATH)
    annotated = df[df["human_label"].notna()]

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print(f"Total annotated: {len(annotated)}/{len(df)}")

    if len(annotated) > 0:
        matches = (annotated["pseudo_intent"] == annotated["human_label"]).sum()
        accuracy = matches / len(annotated)
        print(f"Pseudo-label accuracy: {accuracy:.1%} ({matches}/{len(annotated)})")
        print(f"Saved to: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
