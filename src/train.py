"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def rebalance(data):
    """
    Resample data to keep balance between target classes.

    Args:
        data (pd.DataFrame): Original dataset

    Returns:
        pd.DataFrame: Balanced dataset
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )
    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): Raw input data

    Returns:
        tuple: column transformer, X_train, X_test, y_train, y_test
    """
    filter_feat = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]

    data = df[filter_feat]
    data_bal = rebalance(data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )

    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough"
    )

    X_train = pd.DataFrame(
        col_transf.fit_transform(X_train),
        columns=col_transf.get_feature_names_out()
    )

    X_test = pd.DataFrame(
        col_transf.transform(X_test),
        columns=col_transf.get_feature_names_out()
    )

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train a decision tree classifier.

    Args:
        X_train (pd.DataFrame): Features
        y_train (pd.Series): Target

    Returns:
        DecisionTreeClassifier: Trained model
    """
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def main():
    # Set MLflow tracking URI and experiment
    mlflow.set_experiment("Churn Prediction Experiment")

    with mlflow.start_run():
        df = pd.read_csv("C:/Users/mo/Desktop/MLops/MLOps-Course-Labs/data/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        # Log model parameters
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("max_depth", 5)

        model = train(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.set_tag("stage", "decision_tree_test")

        # Log model with input/output schema
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        joblib.dump(model, "model.pkl")
        joblib.dump(X_train.columns.tolist(), "columns.pkl")  
        # Log confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=model.classes_)
        conf_mat_disp.plot()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.show()


if __name__ == "__main__":
    main()
