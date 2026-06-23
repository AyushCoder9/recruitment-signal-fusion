"""LightGBM LambdaMART wrapper. Directly optimizes NDCG (the real metric). Optional: the
system degrades to rubric-only if no model is present. Feature columns are the numeric
columns of the feature store, in a fixed sorted order shared by train and predict.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

NON_FEATURE = {"candidate_id"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns
            if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(cols)


def matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df.reindex(columns=cols).to_numpy(dtype=np.float32)
    return np.nan_to_num(X, nan=0.0)


def train(df: pd.DataFrame, labels: pd.Series, params: dict, cols: list[str] | None = None):
    import lightgbm as lgb
    cols = cols or feature_columns(df)
    sub = df.loc[labels.index]
    X = matrix(sub, cols)
    y = labels.to_numpy()
    n = len(X)
    val_frac = params.pop("val_fraction", 0.2)
    seed = params.get("seed", 42)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    p = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10, 50],
        "num_leaves": params.get("num_leaves", 31),
        "learning_rate": params.get("learning_rate", 0.05),
        "min_data_in_leaf": params.get("min_data_in_leaf", 20),
        "lambda_l2": params.get("lambda_l2", 1.0),
        "seed": seed, "deterministic": True, "force_col_wise": True,
        "num_threads": 1, "verbosity": -1,
    }
    tr = lgb.Dataset(X[tr_idx], label=y[tr_idx], group=[len(tr_idx)], feature_name=cols)
    va = lgb.Dataset(X[val_idx], label=y[val_idx], group=[len(val_idx)], reference=tr)
    model = lgb.train(p, tr, num_boost_round=params.get("n_estimators", 500),
                      valid_sets=[va], callbacks=[lgb.early_stopping(50, verbose=False),
                                                  lgb.log_evaluation(0)])
    return model, cols, (tr_idx, val_idx)


def predict(model, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return model.predict(matrix(df, cols))


def save(model, cols: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    with open(path + ".cols", "w") as f:
        f.write("\n".join(cols))


def load(path: str):
    import lightgbm as lgb
    if not os.path.exists(path):
        return None, None
    model = lgb.Booster(model_file=path)
    cols = open(path + ".cols").read().splitlines() if os.path.exists(path + ".cols") \
        else model.feature_name()
    return model, cols
