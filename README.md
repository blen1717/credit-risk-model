## Credit Scoring Business Understanding

### How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well‑documented model?

Basel II requires banks to hold capital against credit risk based on internal ratings and loss estimates. For a model to be approved for regulatory capital calculation (the Internal Ratings‑Based approach), it must be interpretable and well‑documented. The supervisor must understand how risk drivers are selected, how the model behaves, and why certain borrowers are classified as high‑risk. An interpretable model (e.g., logistic regression with Weight of Evidence transformations) allows auditors and regulators to validate each variable’s contribution. Poor documentation or a black‑box model would violate Basel II’s principles of transparency and sound risk management.

### Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy‑based prediction introduce?

The raw transaction data contains no historical default flag. We need a proxy to approximate credit risk. A common approach is to identify disengaged customers using RFM (Recency, Frequency, Monetary) analysis. Customers who rarely buy, spend little, and have not transacted recently are more likely to default if extended credit.  

Business risks of proxy‑based prediction:
- Misclassification – a genuinely creditworthy customer who simply uses another payment method may be labelled high‑risk, denying them credit (lost revenue, poor customer experience).
- Proxy decay – the relationship between RFM and default may change over time (e.g., during economic downturns, even active customers may default).
- Regulatory scrutiny – using a proxy instead of actual defaults requires justification; if the proxy is poorly correlated with true credit risk, the model may not meet Basel II expectations.

### What are the key trade‑offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high‑performance model (e.g., Gradient Boosting) in a regulated financial context?

| Aspect | Logistic Regression + WoE | Gradient Boosting (XGBoost, LightGBM) |
|--------|---------------------------|----------------------------------------|
| Interpretability | High – coefficients directly show the impact of each bin; easy to explain to regulators. | Low – complex ensembles are black‑boxes; requires post‑hoc explanations (SHAP, LIME) which may not satisfy Basel II. |
| Predictive performance | Good for linear relationships; may underfit complex interactions. | Usually higher AUC and recall; captures non‑linear patterns and interactions. |
| Regulatory acceptance | Widely accepted for scorecards; easy to document and monitor. | Often requires additional validation and may face higher scrutiny. |
| Development time | Fast to train and tune. | Slower; more hyperparameters. |

In a regulated setting, interpretability often trumps marginal performance gains. Many banks use logistic regression scorecards for regulatory capital models, while using gradient boosting for early warning systems or non‑regulatory applications. The trade‑off is between accuracy and accountability.  
