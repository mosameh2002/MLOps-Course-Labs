
#  Churn Prediction Lab – Model Registration Summary

## Overview
This lab explores predicting bank customer churn using MLflow for tracking experiments and model management.

We tested multiple models and tracked them using MLflow, logging their parameters, performance metrics, and artifacts such as confusion matrices.

---

##  Models Trained

| Version | Model Type             | Accuracy | Precision | Recall | F1 Score | Stage      |
|---------|------------------------|----------|-----------|--------|----------|------------|
| 1       | Logistic Regression    | ~72%     | ~71%      | ~68%   | ~69.5%   | Staging    |
| 2       | Decision Tree Classifier | ~74%     | ~73%      | ~71%   | ~72%     | Production |

---

##  Model Staging Justification

- **Version 2 (Decision Tree Classifier)** was selected for **Production** because it offered the best trade-off between precision and recall and had the highest F1 Score.
- **Version 1 (Logistic Regression)** was kept in **Staging** for monitoring and fallback, as it is simpler and still reasonably performant.

---

##  Artifacts Provided

- MLflow UI screenshots of run metrics
- Confusion matrix of the Decision Tree model
- Model registry showing stages for both versions

---

##  Submission Steps

-  Models registered and stages assigned
-  Screenshots attached in the GitHub repo as a private comment
