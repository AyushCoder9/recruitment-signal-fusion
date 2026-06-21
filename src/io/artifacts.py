"""Artifact IO: embeddings (npz keyed by id), anchors (npz), feature store (parquet)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def save_embeddings(path: str, ids: list[str], emb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, ids=np.array(ids, dtype=object), emb=emb.astype(np.float32))


def load_embeddings(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["ids"], d["emb"]


def save_anchors(path: str, pos: np.ndarray, neg: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, anchor_pos=pos.astype(np.float32), anchor_neg=neg.astype(np.float32))


def load_anchors(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["anchor_pos"], d["anchor_neg"]


def save_features(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)


def load_features(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
