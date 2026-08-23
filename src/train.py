import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    # 1.5.1
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 1.5.2
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # 1.5.3
    with mlflow.start_run():
        # 1.5.4
        mlflow.log_params(params)

        # 1.5.5
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 1.5.6
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # 1.5.7
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # 1.5.8
        mlflow.sklearn.log_model(model, "model")

        # 1.5.9
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # 1.5.10
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # 1.5.11
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # 1.5.12
    return float(acc)


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
