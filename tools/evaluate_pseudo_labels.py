#!/usr/bin/env python3
"""Evaluate pseudo-label quality against golden set."""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

GOLDEN_PATH = "data/golden/golden_set.csv"
AUDIT_PATH = "data/golden/pseudo_label_audit_100.csv"
DEV_PSEUDO_PATH = "data/processed/dev_pseudo_labeled.csv"

def evaluate_audit_set():
    """Evaluate accuracy on audit set (if human labels exist)."""
    print("\n" + "="*70)
    print("AUDIT SET EVALUATION (100 examples)")
    print("="*70)

    audit = pd.read_csv(AUDIT_PATH)
    annotated = audit[audit["human_label"].notna()]

    if len(annotated) == 0:
        print("No human labels yet. Run tools/audit_pseudo_labels.py first.")
        return

    pseudo = annotated["pseudo_intent"]
    human = annotated["human_label"]

    accuracy = accuracy_score(human, pseudo)
    precision, recall, f1, support = precision_recall_fscore_support(
        human, pseudo, average="weighted", zero_division=0
    )

    print(f"\nAnnotated: {len(annotated)}/100")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")

    # Per-intent breakdown
    print("\nPer-intent accuracy:")
    intents = sorted(human.unique())
    for intent in intents:
        mask = human == intent
        acc = accuracy_score(human[mask], pseudo[mask])
        count = mask.sum()
        print(f"  {intent}: {acc:.1%} ({count} examples)")


def evaluate_pseudo_labels():
    """Compare pseudo-labels to golden set."""
    print("\n" + "="*70)
    print("PSEUDO-LABELS vs GOLDEN SET")
    print("="*70)

    golden = pd.read_csv(GOLDEN_PATH)
    dev = pd.read_csv(DEV_PSEUDO_PATH)

    # Find overlap
    golden_ids = set(golden["root_tweet_id"])
    overlap = dev[dev["root_tweet_id"].isin(golden_ids)]

    if len(overlap) == 0:
        print("No overlap between pseudo-labels and golden set (expected).")
        print("Pseudo-labels use 10k dev set, golden set is separate 200 examples.")
        return

    # Merge
    merged = golden.merge(dev, on="root_tweet_id", suffixes=("_golden", "_pseudo"))

    if len(merged) == 0:
        print("No examples in both sets.")
        return

    golden_labels = merged["intent"]
    pseudo_labels = merged["pseudo_intent"]

    accuracy = accuracy_score(golden_labels, pseudo_labels)

    print(f"\nOverlap: {len(merged)} examples")
    print(f"Accuracy: {accuracy:.1%}")

    # Confusion matrix
    intents = sorted(set(golden_labels) | set(pseudo_labels))
    cm = confusion_matrix(golden_labels, pseudo_labels, labels=intents)

    print("\nConfusion matrix:")
    print("(rows=golden, cols=pseudo)")
    for i, intent in enumerate(intents):
        print(f"  {intent}: {cm[i]}")


def stats():
    """Show dataset statistics."""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)

    dev = pd.read_csv(DEV_PSEUDO_PATH)
    golden = pd.read_csv(GOLDEN_PATH)
    audit = pd.read_csv(AUDIT_PATH)

    print(f"\nDev (pseudo-labeled): {len(dev)} examples")
    print("Intent distribution:")
    for intent, count in dev["pseudo_intent"].value_counts().items():
        pct = (count / len(dev)) * 100
        print(f"  {intent}: {count} ({pct:.1f}%)")

    print(f"\nGolden set (human-labeled): {len(golden)} examples")
    print("Intent distribution:")
    for intent, count in golden["intent"].value_counts().items():
        pct = (count / len(golden)) * 100
        print(f"  {intent}: {count} ({pct:.1f}%)")

    annotated_audit = audit[audit["human_label"].notna()]
    print(f"\nAudit set: {len(audit)} examples ({len(annotated_audit)} annotated)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PSEUDO-LABEL EVALUATION")
    print("="*70)

    stats()
    evaluate_pseudo_labels()
    evaluate_audit_set()

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Run: python tools/audit_pseudo_labels.py")
    print("   to manually label the 100-example audit set")
    print("\n2. Run: python tools/evaluate_pseudo_labels.py")
    print("   to see accuracy metrics")
    print("="*70)
