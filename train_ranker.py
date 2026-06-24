#!/usr/bin/env python3
"""Phase 5 — train the LightGBM LambdaMART ranker on the LLM-judge tier labels (offline).

Distills the judge's nuanced reading into a model that runs in milliseconds on CPU. Reports
NDCG/MAP/composite on a held-out split and dumps feature importances so we can confirm the
model leans on role/semantic/production/experience (and flag it if it learns something
spurious). Graceful: if labels are absent the rest of the system still runs rubric-only.

Usage: python train_ranker.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import composite
from src.io.artifacts import load_features
from src.io.config import load_config, resolve_path
from src.scoring import ranker


def main():
    cfg = load_config()
    feat = load_features(resolve_path(cfg["artifacts"]["features"]))
    labels_path = resolve_path(cfg["artifacts"]["labels"])
    labels = pd.read_parquet(labels_path)
    y = labels["tier"].reindex(feat.index).dropna().astype(int)
    print(f"labeled candidates: {len(y)}   tier dist: {y.value_counts().sort_index().to_dict()}")

    cols = ranker.feature_columns(feat)
    params = dict(cfg["ranker"])
    params.pop("enabled", None)
    params["seed"] = cfg["determinism"]["seed"]

    model, cols, (tr_idx, val_idx) = ranker.train(feat, y, params, cols)

    # evaluate on the held-out labeled split
    sub = feat.loc[y.index]
    val_ids = y.index[val_idx]
    preds = ranker.predict(model, sub.loc[val_ids], cols)
    order = [val_ids[i] for i in np.argsort(-preds)]
    rel = {cid: int(y.loc[cid]) for cid in val_ids}
    m = composite(order, rel)
    print("\nHELD-OUT METRICS:")
    for k, v in m.items():
        print(f"  {k:10s} {v:.4f}")

    imp = sorted(zip(cols, model.feature_importance(importance_type="gain")),
                 key=lambda x: -x[1])
    print("\nTOP 15 FEATURES (gain):")
    for name, g in imp[:15]:
        print(f"  {name:28s} {g:10.1f}")

    ranker.save(model, cols, resolve_path(cfg["artifacts"]["ranker"]))
    print(f"\nsaved ranker -> {resolve_path(cfg['artifacts']['ranker'])}")


if __name__ == "__main__":
    main()
