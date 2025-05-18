# MLOps Course Labs

Welcome to the lab repository for the [MLOps Course](https://github.com/Heba-Atef99/MLOps-Course).

Throughout this hands-on journey, you'll develop a **Bank Customer Churn Prediction** application—starting from the research phase and progressing through the full MLOps lifecycle, all the way to deployment.

> **Note:** Currently, the repository contains only the `research` branch. The remaining branches will be built step by step by the reader during the course days, as part of the learning experience.

---

##  Dataset

- **Source:** [Kaggle - Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
- The dataset contains 10,000+ customer records with features such as:
  - CreditScore, Geography, Gender, Age
  - Balance, NumOfProducts, HasCrCard, IsActiveMember
  - EstimatedSalary
  - Target variable: `Exited` (1 = churned, 0 = stayed)

---

##  Setup & Running

1. Clone the repo and switch to `research` branch.
2. Create and activate a virtual environment.
3. Install dependencies using `requirements.txt`.
4. Place the CSV in the `data/` directory.
5. Run:
   ```bash
   mlflow ui  # in one terminal
   python src/train.py  # in another
