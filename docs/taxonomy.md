# SupportIQ Intent Taxonomy

## Purpose

This taxonomy defines the customer-support intents used for
intent classification in SupportIQ.

The taxonomy was derived by manually reviewing a sample of
AmazonHelp customer conversations from the Customer Support on Twitter
dataset.

---

## Intent Labels

### 1. Delivery Issue

Problems related to the delivery of an order.

Includes:
- Late delivery
- Delayed shipment
- Missing delivery
- Delivery tracking
- Delivery date problems
- Carrier/delivery problems
- Package marked delivered but not received

Does NOT include:
- Damaged products after delivery
- Returning an item
- Payment problems

Example:
"My package was supposed to arrive yesterday but it hasn't."

---

### 2. Damaged / Wrong / Missing Item

Problems with the physical item received or expected inside the package.

Includes:
- Damaged product
- Wrong product received
- Missing item from package
- Broken product
- Damaged packaging when it affects the item

Does NOT include:
- Package completely missing during delivery
- Return/refund requests where the item problem is not the primary issue

Example:
"I ordered a blue shirt but received a red one."

---

### 3. Order Management

Problems involving managing or changing an order.

Includes:
- Cancel order
- Modify order
- Change order details
- Order confirmation
- Order status
- Problems placing an order
- Pre-order issues

Does NOT include:
- Delivery delays
- Refund requests
- Payment problems

Example:
"I want to cancel my order."

---

### 4. Return / Refund

Requests or problems involving returning an item or receiving a refund.

Includes:
- Return request
- Refund request
- Refund not received
- Return status
- Questions about return process

Does NOT include:
- General order cancellation before fulfillment
- Payment failures

Example:
"I returned my item two weeks ago but haven't received my refund."

---

### 5. Payment / Billing

Problems involving payment or charges.

Includes:
- Payment failure
- Unexpected charge
- Duplicate charge
- Billing problems
- Payment method problems
- Incorrect amount charged

Does NOT include:
- Refund status after a return

Example:
"I was charged twice for the same order."

---

### 6. Account / Access / Security

Problems involving the customer's Amazon account.

Includes:
- Login problems
- Password problems
- Account access
- Locked account
- Account security
- Suspicious account activity

Example:
"I can't log into my Amazon account."

---

### 7. Prime Membership

Problems specifically related to Amazon Prime membership.

Includes:
- Prime subscription
- Prime trial
- Prime membership charges
- Prime benefits
- Prime membership cancellation

Does NOT include:
- Prime Video content problems

Example:
"Why was I charged for Prime?"

---

### 8. Digital Content

Problems involving Amazon digital services or digital content.

Includes:
- Prime Video
- Digital purchases
- Kindle content
- Digital downloads
- Digital streaming problems
- Digital content access

Does NOT include:
- Physical Amazon devices

Example:
"Prime Video isn't playing on my account."

---

### 9. Product / Device Support

Problems involving Amazon products or devices.

Includes:
- Echo
- Fire TV
- Kindle device
- Alexa
- Device setup
- Device malfunction
- Product troubleshooting

Example:
"My Echo won't connect to Wi-Fi."

---

### 10. Promotion / Gift Card / Credit

Problems involving promotions, gift cards, discounts, or promotional credits.

Includes:
- Gift cards
- Promotional codes
- Discounts
- Promotional credits
- Cashback/promotional offers

Example:
"My promotional credit hasn't appeared."

---

### 11. Other / Unclear

Messages that do not clearly belong to another intent.

Includes:
- Ambiguous requests
- Insufficient information
- General comments
- Irrelevant messages
- Messages outside the supported intent categories

Example:
"Amazon, please help me."

---

# Annotation Rules

## Rule 1 — Label the customer's primary problem

Choose the intent representing the main issue the customer wants resolved.

## Rule 2 — Do not label based on keywords alone

Use the meaning of the complete customer message.

## Rule 3 — Choose one intent

Each golden-set example receives exactly one primary intent.

## Rule 4 — Prefer the most specific applicable intent

For example:

"My Prime subscription was charged twice."

→ Prime Membership

rather than Payment / Billing.

## Rule 5 — Use Other / Unclear when evidence is insufficient

Do not force an ambiguous message into an unrelated intent.

## Rule 6 — Historical Amazon responses are not used to determine the label

The intent label is based primarily on the customer's problem.

---

# Taxonomy Version

Version: 1.0

Number of intents: 11

Golden set size: 200

