# SupportIQ — Classifier Evaluation Report

## 1. Evaluation Objective

The goal of this experiment is to compare increasingly capable intent-classification approaches for the Amazon customer-support dataset.

The evaluated approaches are:

1. **Baseline 1 — Majority Classifier**
2. **Baseline 2 — TF-IDF + Logistic Regression**
3. **Semantic Transformer + Logistic Regression**
4. **Semantic Transformer + k-NN**
5. **Hybrid — TF-IDF + Sentence Transformer + Logistic Regression**

The primary evaluation set is the **200-example human-labeled Golden Set**. The 2K development test split is used as a development benchmark, while the 100-example Audit Set is a smaller human-reviewed validation set.

---

## 2. Evaluation Datasets

| Dataset | Size | Label Source | Purpose |
|---|---:|---|---|
| Dev set | 10,000 | Mixed pseudo-labels and manually reviewed labels | Development/training pool |
| Dev 2K Test | 2,000 | Labels present at split time | Held-out development benchmark |
| Golden Set | 200 | Human annotated | **Primary evaluation benchmark** |
| Audit Set | 100 | Human reviewed | Additional validation/audit |

### Label-quality note

The 10K development set is no longer purely pseudo-labeled. At the time of this evaluation, **2,329 of 10,000 rows had been manually reviewed**, while the remainder retained their existing pseudo-labels.

Therefore, the 2K development-test result should be interpreted as performance against the labels available at split/evaluation time, rather than as a fully human-validated test score.

---

## 3. Baseline 1 — Majority Classifier

The majority classifier predicts the most frequent intent for every input.

The majority intent is **Delivery Issue**.

This establishes a trivial lower-bound benchmark against which the machine-learning approaches can be compared.

### Results

| Dataset | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dev 2K Test | 39.0% | 0.035 | 0.091 | 0.051 | 0.152 | 0.390 | 0.219 | 2000 |
| Golden Set | 34.0% | 0.031 | 0.091 | 0.046 | 0.116 | 0.340 | 0.173 | 200 |
| Audit Set | 33.0% | 0.033 | 0.100 | 0.050 | 0.109 | 0.330 | 0.164 | 100 |

As expected, the majority classifier performs well only for the dominant Delivery Issue class and provides almost no useful coverage of minority intents.

---

## 4. Baseline 2 — TF-IDF + Logistic Regression

TF-IDF represents customer messages using lexical features such as individual words and word n-grams. Logistic Regression then predicts the intent.

This baseline tests whether simple, interpretable lexical features can capture enough information for intent classification.

### Results

| Dataset | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dev 2K Test | 55.0% | 0.392 | 0.450 | 0.412 | 0.605 | 0.550 | 0.565 | 2000 |
| Golden Set | 43.0% | 0.342 | 0.370 | 0.345 | 0.447 | 0.430 | 0.427 | 200 |
| Audit Set | 33.0% | 0.033 | 0.100 | 0.050 | 0.109 | 0.330 | 0.164 | 100 |

The TF-IDF baseline substantially improves over the majority classifier on the development benchmark and Golden Set in the recorded evaluation.

**Audit note:** the current consolidated run reports the TF-IDF Audit Set result as identical to the majority baseline. This should be verified against the audit prediction-generation cell before being presented as a final audit result, because the prediction array used in the notebook was also reused in earlier evaluation work.

---

## 5. Semantic Transformer + Logistic Regression

The semantic model uses the `all-MiniLM-L6-v2` Sentence Transformer to convert each customer message into a 384-dimensional semantic embedding.

These embeddings are then provided to Logistic Regression.

### Results

| Dataset | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dev 2K Test | 46.6% | 0.398 | 0.533 | 0.418 | 0.613 | 0.466 | 0.494 | 2000 |
| Golden Set | 45.5% | 0.440 | 0.489 | 0.441 | 0.543 | 0.455 | 0.463 | 200 |

Compared with TF-IDF, the semantic representation achieves higher Golden Macro F1 and slightly higher Golden accuracy in this recorded experiment.

The improvement in Macro F1 suggests that semantic features can provide better balance across intents even when overall accuracy remains moderate.

---

## 6. Semantic Transformer + k-NN

A second semantic experiment uses the same Sentence Transformer embeddings with cosine-distance k-nearest-neighbor classification.

### Results

| Dataset | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dev 2K Test | 27.5% | 0.086 | 0.088 | 0.083 | 0.218 | 0.275 | 0.240 | 2000 |
| Golden Set | 38.0% | 0.252 | 0.209 | 0.198 | 0.323 | 0.380 | 0.313 | 200 |

The k-NN approach performs substantially worse than both Logistic Regression approaches.

This indicates that, for this taxonomy and training data, simply finding nearby examples in embedding space is less effective than learning a discriminative classifier over the representations.

---

## 7. Hybrid — TF-IDF + Sentence Transformer + Logistic Regression

The Hybrid model combines two complementary representations:

- **TF-IDF:** captures exact words and phrases.
- **Sentence Transformer embeddings:** capture semantic similarity and meaning.

The features are concatenated and passed to Logistic Regression.

Conceptually:

```text
Customer Message
       |
       +--------------------+
       |                    |
       v                    v
     TF-IDF          Sentence Transformer
  lexical features     semantic features
       |                    |
       +---------+----------+
                 |
                 v
           Feature Concatenation
                 |
                 v
        Logistic Regression
                 |
                 v
          Intent Prediction
```

The recorded hybrid representation contained **32,661 TF-IDF features + 384 embedding features = 33,045 features**.

### Results

| Dataset | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dev 2K Test | 53.4% | 0.412 | 0.494 | 0.435 | 0.613 | 0.534 | 0.556 | 2000 |
| Golden Set | **54.0%** | **0.491** | **0.514** | **0.484** | **0.588** | **0.540** | **0.547** | 200 |

The Hybrid approach produces the strongest recorded result on the human Golden Set among the tested classifiers.

---

## 8. Overall Model Comparison

### Human Golden Set — Primary Benchmark

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline 1 — Majority | 34.0% | 0.031 | 0.091 | 0.046 | 0.116 | 0.340 | 0.173 |
| TF-IDF + LR | 43.0% | 0.342 | 0.370 | 0.345 | 0.447 | 0.430 | 0.427 |
| Semantic Transformer + LR | 45.5% | 0.440 | 0.489 | 0.441 | 0.543 | 0.455 | 0.463 |
| Semantic Transformer + k-NN | 38.0% | 0.252 | 0.209 | 0.198 | 0.323 | 0.380 | 0.313 |
| **Hybrid — TF-IDF + Transformer + LR** | **54.0%** | **0.491** | **0.514** | **0.484** | **0.588** | **0.540** | **0.547** |

### Development 2K Benchmark

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline 1 — Majority | 39.0% | 0.035 | 0.091 | 0.051 | 0.152 | 0.390 | 0.219 |
| TF-IDF + LR | 55.0% | 0.392 | 0.450 | 0.412 | 0.605 | 0.550 | 0.565 |
| Semantic Transformer + LR | 46.6% | 0.398 | 0.533 | 0.418 | 0.613 | 0.466 | 0.494 |
| Semantic Transformer + k-NN | 27.5% | 0.086 | 0.088 | 0.083 | 0.218 | 0.275 | 0.240 |
| **Hybrid — TF-IDF + Transformer + LR** | **53.4%** | **0.412** | **0.494** | **0.435** | **0.613** | **0.534** | **0.556** |

---

## 9. Model Progression

The experiments show the following progression on the human Golden Set:

```text
Majority Classifier
34.0% Accuracy
0.046 Macro F1
        |
        v
TF-IDF + Logistic Regression
43.0% Accuracy
0.345 Macro F1
        |
        v
Semantic Transformer + Logistic Regression
45.5% Accuracy
0.441 Macro F1
        |
        v
Hybrid
54.0% Accuracy
0.484 Macro F1
```

The Hybrid model improves Golden accuracy by **11.0 percentage points** over TF-IDF + Logistic Regression:

```text
54.0% - 43.0% = 11.0 percentage points
```

It also improves Golden Macro F1 from **0.345 to 0.484**.

---

## 10. Error Patterns

The Golden Set confusion analysis identified several recurring error patterns.

### 10.1 Ambiguous delivery language

Delivery-related messages are sometimes confused with Other / Unclear, Product / Device Support, or Order Management.

Example:

> “My package was supposed to arrive yesterday but I still haven't received it.”

The lexical overlap around packages and orders can make the precise intent difficult to distinguish.

### 10.2 Shipment problem vs. item problem

The classifier can confuse:

- a problem with the shipment/delivery process
- a problem with the item that arrived

Example:

> “My package arrived, but Amazon sent me the wrong phone.”

The presence of words such as “package” and “arrived” can pull the prediction toward Delivery Issue even though the primary problem concerns the wrong item.

### 10.3 Order-related lexical overlap

Words such as “order”, “package”, and “arrived” appear across several intents.

For example:

> “My order arrived, but the item is damaged.”

The word “order” may encourage an Order Management prediction even though the primary problem is a damaged item.

### 10.4 Rare intents

Low-frequency intents have fewer examples available for learning.

This particularly affects:

- Prime Membership
- Promotion / Gift Card / Credit

Sparse representation makes these categories harder to learn consistently.

### 10.5 Semantic ambiguity

Some messages require understanding the customer's underlying problem rather than matching individual words.

Examples include:

> “I can't access my Amazon account.”

and

> “I can't watch the movie I purchased.”

These require distinguishing account access from digital-content access despite potentially overlapping vocabulary.

---

## 11. What Is Misleading About My Headline Number?

The headline result of **54.0% Golden Set accuracy** should not be interpreted as “the support agent is 54% correct in production.”

Several limitations apply:

1. The Golden Set contains only **200 examples**.
2. Some intents have relatively few examples.
3. The dataset comes from historical Twitter customer-support conversations and may not represent current support traffic.
4. The 2K development benchmark contains labels that are not fully human validated.
5. Historical support conversations contain noise, ambiguity, and incomplete context.
6. Intent classification is only one component of the complete SupportIQ system.
7. A correct intent prediction does not guarantee that the generated response will be correct or safe.

Therefore, **54.0% is a benchmark result for this specific human-labeled 200-example evaluation set**, not a production accuracy estimate.

---

## 12. Key Findings

### Finding 1 — A trivial baseline is insufficient

The majority classifier reaches only **34.0% accuracy** on the Golden Set and has a Macro F1 of **0.046**.

This establishes that the task cannot be solved adequately by always predicting the dominant class.

### Finding 2 — Lexical features provide a substantial improvement

TF-IDF + Logistic Regression increases Golden accuracy from **34.0% to 43.0%**.

This demonstrates that customer-message vocabulary contains useful intent information.

### Finding 3 — Semantic representations improve class-balanced performance

The Semantic Transformer + Logistic Regression model reaches **0.441 Macro F1** on the Golden Set compared with **0.345** for TF-IDF + Logistic Regression.

This indicates that semantic representations can help distinguish intents that use different wording for similar problems.

### Finding 4 — k-NN is not effective for this setup

Semantic k-NN reaches only **38.0% Golden accuracy** and **0.198 Macro F1**.

The result suggests that semantic similarity alone is not sufficient for this multi-class intent taxonomy.

### Finding 5 — Combining lexical and semantic information performs strongly

The Hybrid model reaches:

- **54.0% Golden accuracy**
- **0.491 Macro Precision**
- **0.514 Macro Recall**
- **0.484 Macro F1**
- **0.547 Weighted F1**

This is the strongest recorded Golden Set result among the tested approaches.

---

## 13. Limitations

The results should be interpreted with the following limitations:

- The primary human Golden Set contains only 200 examples.
- Several intents have small support in the Golden Set.
- The development data contains pseudo-labels and manually reviewed labels.
- The Audit Set is only 100 examples.
- The dataset contains noisy and sometimes incomplete Twitter conversations.
- The classifier currently predicts one intent even when a conversation could contain multiple issues.
- These experiments evaluate classification only; retrieval quality, generated response quality, and escalation behavior require separate evaluation.

---

## 14. Decision

Based on the recorded experiments, the **Hybrid TF-IDF + Sentence Transformer + Logistic Regression** model is selected as the classifier candidate for the next SupportIQ pipeline stage.

The reason is empirical rather than theoretical: it produced the highest recorded performance on the **200-example human Golden Set**, with **54.0% accuracy and 0.484 Macro F1**.

The next stage is therefore to integrate the classifier with:

```text
Customer Message
      ↓
Hybrid Intent Classifier
      ↓
Historical Retrieval
      ↓
Evidence-Grounded Response Generation
      ↓
Escalation Policy
      ↓
Final Support Action
```

Classification performance will not be treated as a proxy for the complete agent's quality. Retrieval, response grounding, response quality, and escalation will be evaluated separately.
