import os
import json
import pandas as pd
import pytest
from src.train import train

def _make_temp_data(tmp_path):
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    
    df_train = pd.DataFrame({
        "fixed_acidity": [7.4, 7.8],
        "volatile_acidity": [0.7, 0.88],
        "citric_acid": [0, 0],
        "residual_sugar": [1.9, 2.6],
        "chlorides": [0.076, 0.098],
        "free_sulfur_dioxide": [11, 25],
        "total_sulfur_dioxide": [34, 67],
        "density": [0.9978, 0.9968],
        "pH": [3.51, 3.2],
        "sulphates": [0.56, 0.68],
        "alcohol": [9.4, 9.8],
        "wine_type": [0, 0],
        "target": [0, 1]
    })
    
    df_eval = df_train.copy()
    
    df_train.to_csv(train_path, index=False)
    df_eval.to_csv(eval_path, index=False)
    
    return str(train_path), str(eval_path)


def test_train_returns_float(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    res = train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(res, float)
    assert 0.0 <= res <= 1.0


def test_metrics_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json", "r") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics


def test_model_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert os.path.exists("models/model.pkl")
