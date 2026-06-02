# Credit Risk Model for Xente eCommerce Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Project Overview
Build a credit scoring model for a buy‑now‑pay‑later service using transaction data from the Xente platform (Kaggle challenge). Since no default labels exist, we create a proxy target via RFM (Recency, Frequency, Monetary) analysis and K‑means clustering. The output is a risk probability and a credit score.

## Credit Scoring Business Understanding

### How does Basel II influence the need for an interpretable and well‑documented model?
Basel II requires banks to hold capital against credit risk based on internal ratings. An interpretable model (e.g., logistic regression with Weight of Evidence) is essential for regulatory approval because supervisors must understand risk drivers. Poor documentation or a black‑box model violates Basel II’s principles of transparency.

### Without a direct "default" label, why is a proxy variable necessary, and what business risks does it introduce?
We must approximate credit risk using a proxy – in this case, RFM + clustering to identify disengaged customers (low recency, low frequency, low monetary value) as high‑risk.  
Risks: misclassification of creditworthy customers, proxy decay over time, and regulatory scrutiny. Mitigations include continuous monitoring, fallback rules, and thorough documentation.

### Trade‑offs: Logistic Regression+WoE vs. Gradient Boosting in a regulated context
| Aspect | Logistic Regression + WoE | Gradient Boosting (XGBoost) |
|--------|---------------------------|------------------------------|
| Interpretability | High – each bin’s weight is transparent | Low – needs post‑hoc explanations |
| Performance | Good for linear relationships | Higher AUC, captures interactions |
| Regulatory acceptance | Widely accepted | Often requires extra validation |
| Development time | Fast | Slower |

Bank decision: For regulatory capital models, interpretability is mandatory. For early‑warning systems, gradient boosting may be used to maximise recall.

## Repository Structure
