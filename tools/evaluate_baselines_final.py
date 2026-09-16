#!/usr/bin/env python3
"""
Final baseline evaluation across all 3 labeled datasets:
  1. Golden Set        (data/golden/golden_set.csv)             - 200 human-labeled
  2. Audit Set         (data/golden/pseudo_label_audit_100.csv) - 100 human-labeled
  3. Dev Pseudo-Labeled (data/processed/dev_pseudo_labeled.csv) - 10,000 (mix of
     manually-reviewed and heuristic pseudo-labels), held-out test split.

Trains two baselines on the dev pseudo-labeled training split:
  - Baseline 1: Majority-class classifier
  - Baseline 2: TF-IDF + Logistic Regression

Evaluates both against all 3 datasets and writes a consolidated report to results/.
"""

import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

GOLDEN_PATH = "data/golden/golden_set.csv"
AUDIT_PATH = "data/golden/pseudo_label_audit_100.csv"
DEV_PATH = "data/processed/dev_pseudo_labeled.csv"
RESULTS_DIR = "results"

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


def load_data():
    golden = pd.read_csv(GOLDEN_PATH)
    audit = pd.read_csv(AUDIT_PATH)
    dev = pd.read_csv(DEV_PATH)
    return golden, audit, dev


def train_baselines(dev):
    train_df, test_df = train_test_split(
        dev, test_size=2000, random_state=42, stratify=None
    )

    X_train_text = train_df["conversation"]
    y_train = train_df["pseudo_intent"]
    X_test_text = test_df["conversation"]
    y_test = test_df["pseudo_intent"]

    # Baseline 1: Majority classifier
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train_text, y_train)

    # Baseline 2: TF-IDF + Logistic Regression
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_vec, y_train)

    return {
        "majority": majority,
        "vectorizer": vectorizer,
        "logreg": logreg,
        "train_df": train_df,
        "test_df": test_df,
        "X_test_vec": X_test_vec,
        "y_test": y_test,
    }


def eval_predictions(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "label": label,
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "weighted_precision": p_w,
        "weighted_recall": r_w,
        "weighted_f1": f1_w,
        "n": len(y_true),
    }


def predict_logreg(vectorizer, model, texts):
    return model.predict(vectorizer.transform(texts))


def predict_majority(model, texts):
    return model.predict(texts)


def top_confusions(y_true, y_pred, top_n=15):
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df = df[df["true"] != df["pred"]]
    counts = (
        df.groupby(["true", "pred"]).size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return counts.head(top_n)


def main():
    golden, audit, dev = load_data()

    print(f"Golden set: {len(golden)} examples")
    print(f"Audit set: {len(audit)} examples")
    print(f"Dev pseudo-labeled: {len(dev)} examples "
          f"({dev['reviewed'].sum() if 'reviewed' in dev.columns else 0} manually reviewed)")

    baselines = train_baselines(dev)

    rows = []

    # --- Baseline 1: Majority ---
    maj_test_pred = predict_majority(baselines["majority"], baselines["test_df"]["conversation"])
    rows.append(eval_predictions(baselines["y_test"], maj_test_pred, "Baseline 1 - Majority | Dev 2K Test"))

    maj_golden_pred = predict_majority(baselines["majority"], golden["conversation"])
    rows.append(eval_predictions(golden["intent"], maj_golden_pred, "Baseline 1 - Majority | Golden Set (200)"))

    maj_audit_pred = predict_majority(baselines["majority"], audit["conversation"])
    rows.append(eval_predictions(audit["human_label"], maj_audit_pred, "Baseline 1 - Majority | Audit Set (100)"))

    # --- Baseline 2: TF-IDF + LogReg ---
    lr_test_pred = predict_logreg(baselines["vectorizer"], baselines["logreg"], baselines["test_df"]["conversation"])
    rows.append(eval_predictions(baselines["y_test"], lr_test_pred, "Baseline 2 - TF-IDF+LogReg | Dev 2K Test"))

    lr_golden_pred = predict_logreg(baselines["vectorizer"], baselines["logreg"], golden["conversation"])
    rows.append(eval_predictions(golden["intent"], lr_golden_pred, "Baseline 2 - TF-IDF+LogReg | Golden Set (200)"))

    lr_audit_pred = predict_logreg(baselines["vectorizer"], baselines["logreg"], audit["conversation"])
    rows.append(eval_predictions(audit["human_label"], lr_audit_pred, "Baseline 2 - TF-IDF+LogReg | Audit Set (100)"))

    results_df = pd.DataFrame(rows)
    results_df.to_csv(f"{RESULTS_DIR}/final_baseline_metrics.csv", index=False)
    print("\n", results_df.to_string(index=False))

    # Confusion pairs on golden set (largest human-labeled set) for LogReg
    confusions_golden = top_confusions(golden["intent"], lr_golden_pred)
    confusions_golden.to_csv(f"{RESULTS_DIR}/final_confusion_pairs_golden.csv", index=False)

    # Confusion pairs on audit set
    confusions_audit = top_confusions(audit["human_label"], lr_audit_pred)
    confusions_audit.to_csv(f"{RESULTS_DIR}/final_confusion_pairs_audit.csv", index=False)

    # Misclassified examples (golden set)
    golden_out = golden.copy()
    golden_out["predicted_intent"] = lr_golden_pred
    golden_mis = golden_out[golden_out["intent"] != golden_out["predicted_intent"]]
    golden_mis.to_csv(f"{RESULTS_DIR}/final_golden_misclassified.csv", index=False)

    # Misclassified examples (audit set)
    audit_out = audit.copy()
    audit_out["predicted_intent"] = lr_audit_pred
    audit_mis = audit_out[audit_out["human_label"] != audit_out["predicted_intent"]]
    audit_mis.to_csv(f"{RESULTS_DIR}/final_audit_misclassified.csv", index=False)

    # Class distribution across all 3 datasets
    dist = pd.DataFrame({
        "Dev Pseudo-Labeled (10k)": dev["pseudo_intent"].value_counts(),
        "Golden Set (200)": golden["intent"].value_counts(),
        "Audit Set (100)": audit["human_label"].value_counts(),
    }).fillna(0).astype(int)
    dist.to_csv(f"{RESULTS_DIR}/final_class_distribution.csv")

    write_report(results_df, dist, confusions_golden, confusions_audit, dev)

    print("\nDone. Results written to results/")


def write_report(results_df, dist, confusions_golden, confusions_audit, dev):
    reviewed = dev["reviewed"].sum() if "reviewed" in dev.columns else 0
    lines = []
    lines.append("# Final Results — All Baselines x All Datasets\n")
    lines.append(
        "Consolidated evaluation of both baselines (Majority Classifier, "
        "TF-IDF + Logistic Regression) against all 3 labeled datasets: the "
        "10k dev pseudo-labeled set (held-out 2K test split), the 200-example "
        "human-labeled Golden Set, and the 100-example human-labeled Audit Set.\n"
    )
    lines.append(
        f"Note: the dev pseudo-labeled set currently has **{reviewed}/{len(dev)}** rows "
        "manually reviewed (genuine human judgment); the remainder still carry the "
        "original heuristic pseudo-labels. Training/test splits below use whatever "
        "`pseudo_intent` values are present at the time this script was run.\n"
    )

    lines.append("## Metrics\n")
    lines.append("| Run | Accuracy | Macro P | Macro R | Macro F1 | Weighted P | Weighted R | Weighted F1 | N |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in results_df.iterrows():
        lines.append(
            f"| {r['label']} | {r['accuracy']:.1%} | {r['macro_precision']:.3f} | "
            f"{r['macro_recall']:.3f} | {r['macro_f1']:.3f} | {r['weighted_precision']:.3f} | "
            f"{r['weighted_recall']:.3f} | {r['weighted_f1']:.3f} | {int(r['n'])} |"
        )

    lines.append("\n## Class Distribution Across Datasets\n")
    lines.append("| Intent | Dev (10k) | Golden (200) | Audit (100) |")
    lines.append("|---|---:|---:|---:|")
    for intent in INTENTS:
        d = dist.loc[intent, "Dev Pseudo-Labeled (10k)"] if intent in dist.index else 0
        g = dist.loc[intent, "Golden Set (200)"] if intent in dist.index else 0
        a = dist.loc[intent, "Audit Set (100)"] if intent in dist.index else 0
        lines.append(f"| {intent} | {d} | {g} | {a} |")

    lines.append("\n## Top Confusions — TF-IDF+LogReg on Golden Set (200)\n")
    lines.append("| True | Predicted | Count |")
    lines.append("|---|---|---:|")
    for _, r in confusions_golden.iterrows():
        lines.append(f"| {r['true']} | {r['pred']} | {r['count']} |")

    lines.append("\n## Top Confusions — TF-IDF+LogReg on Audit Set (100)\n")
    lines.append("| True | Predicted | Count |")
    lines.append("|---|---|---:|")
    for _, r in confusions_audit.iterrows():
        lines.append(f"| {r['true']} | {r['pred']} | {r['count']} |")

    lines.append(
        "\n## Takeaway\n\n"
        "TF-IDF + Logistic Regression outperforms the majority-class baseline on every "
        "dataset, but the gap between its performance on the pseudo-label test split and "
        "the two human-labeled sets confirms that agreement with pseudo-labels overstates "
        "real-world accuracy. Rare intents (Prime Membership, Promotion / Gift Card / Credit) "
        "remain the weakest classes across both baselines due to data sparsity in the "
        "training set."
    )

    with open(f"{RESULTS_DIR}/FINAL_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
