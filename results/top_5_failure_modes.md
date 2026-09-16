# Baseline 2 — Top 5 Failure Modes

**Model:** TF-IDF + Logistic Regression  
**Evaluation Set:** 200 Human-Labeled Golden Examples  
**Accuracy:** 41.5%  
**Macro F1:** 0.330

---

## Overview

The confusion matrix shows that the TF-IDF + Logistic Regression baseline performs reasonably when an intent has distinctive vocabulary, but struggles when different support intents use similar words or when an intent has very few training examples.

---

## 1. Delivery Problems Are Sometimes Classified as Other

### What happens?

The model sometimes recognizes that a customer is discussing an order or package but fails to identify that the main problem is specifically related to delivery.

### Example

> "My package was supposed to arrive yesterday but I still haven't received it."

**Human label:** `Delivery Issue`  
**Possible model prediction:** `Other / Unclear`

### Evidence

| Confusion | Count |
|---|---:|
| Delivery Issue → Other / Unclear | **13** |
| Delivery Issue → Product / Device Support | **5** |
| Delivery Issue → Order Management | **4** |

### Why?

TF-IDF mainly relies on words and short phrases. It does not fully understand the overall meaning of the customer's situation.

---

## 2. Delivery Problem vs. Item Problem

### What happens?

The model sometimes confuses a problem with the delivery or package with a problem involving the item inside the package.

### Example

> "My package arrived, but Amazon sent me the wrong phone."

**Human label:** `Damaged / Wrong / Missing Item`  
**Possible model prediction:** `Delivery Issue`

### Evidence

| Confusion | Count |
|---|---:|
| Damaged / Wrong / Missing Item → Delivery Issue | **7** |
| Delivery Issue → Damaged / Wrong / Missing Item | **5** |

### Why?

Both intents can contain similar words such as `package`, `order`, `arrived`, `delivery`, and `received`.

---

## 3. Order Appears in Many Different Problems

### What happens?

The word **order** appears across many support situations. A customer may mention an order while actually reporting a damaged item, missing item, payment problem, or another issue.

### Example

> "My order arrived, but the item is damaged."

**Human label:** `Damaged / Wrong / Missing Item`  
**Possible model prediction:** `Order Management`

### Evidence

| Confusion | Count |
|---|---:|
| Damaged / Wrong / Missing Item → Order Management | **6** |
| Other / Unclear → Order Management | **3** |

### Why?

TF-IDF learns associations between words and intents, but it does not explicitly reason about which part of the message represents the customer's primary problem.

---

## 4. Rare Intents Have Too Few Training Examples

### What happens?

Some intents are extremely rare in the 10,000-example pseudo-labeled development set.

| Intent | Training Examples |
|---|---:|
| Prime Membership | **16** |
| Promotion / Gift Card / Credit | **40** |

For comparison:

| Intent | Training Examples |
|---|---:|
| Delivery Issue | **4,046** |
| Other / Unclear | **2,274** |

### Example

> "I accidentally signed up for Prime Student."

**Human label:** `Prime Membership`  
**Possible model prediction:** `Other / Unclear`

### Evidence

For `Prime Membership`:

| Prediction | Count |
|---|---:|
| Other / Unclear | **6** |
| Delivery Issue | **6** |
| Correct | **0** |

For `Promotion / Gift Card / Credit`:

| Prediction | Count |
|---|---:|
| Payment / Billing | **4** |
| Other / Unclear | **4** |
| Correct | **0** |

### Why?

The classifier has very few training examples from which to learn the language patterns of these intents.

This is primarily a **data sparsity problem**.

---

## 5. The Model Relies on Words Rather Than Full Meaning

### What happens?

Some messages require understanding the situation rather than simply matching keywords.

### Example

> "I can't access my Amazon account."

**Human label:** `Account / Access / Security`  
**Possible model prediction:** `Other / Unclear`

Another example:

> "I can't watch the movie I purchased."

**Human label:** `Digital Content`  
**Possible model prediction:** `Delivery Issue`

### Evidence

| Confusion | Count |
|---|---:|
| Digital Content → Delivery Issue | **5** |
| Account / Access / Security → Other / Unclear | **4** |
| Payment / Billing → Other / Unclear | **3** |

### Why?

TF-IDF represents lexical patterns rather than deeper semantic relationships. Two messages can use overlapping vocabulary while expressing different customer problems.

---

# Summary

| # | Failure Mode | Main Cause |
|---|---|---|
| 1 | Delivery → Other | Ambiguous delivery language |
| 2 | Delivery vs. Item Problem | Similar package/order vocabulary |
| 3 | Order Word Confusion | Order appears across many intents |
| 4 | Rare Intents | Insufficient training examples |
| 5 | Semantic Ambiguity | Lexical features do not capture full meaning |

---

## Main Limitation of the Baseline

> **TF-IDF is effective at capturing important words and short phrases, but has limited ability to understand context, intent boundaries, and deeper semantic meaning.**

These limitations motivate moving beyond the lexical baseline toward semantic representations and historical support evidence.