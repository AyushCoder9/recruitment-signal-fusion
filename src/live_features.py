"""Build a feature frame live for candidates NOT in the precomputed store (the sandbox's
arbitrary <=100 uploads). Mirrors precompute: static features + dense sims to the JD anchors
+ a small BM25. The official 100k run never hits this path (all ids are cached), so it stays
network-free; only the sandbox loads the embedder here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .embed import Embedder
from .features import bm25 as bm25mod
from .features.registry import extract_static, get_reference_now
from .features.text import build_candidate_text
from .io import artifacts
from .io.config import load_config, resolve_path


def build_live_features(cands: list, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ref_now = get_reference_now()
    ids = [c.candidate_id for c in cands]
    texts = [build_candidate_text(c) for c in cands]
    rows = [extract_static(c, ref_now) for c in cands]
    df = pd.DataFrame(rows).set_index("candidate_id")

    ec = cfg["embedding"]
    embedder = Embedder(ec["model"], dim=ec["dim"], max_seq_len=ec["max_seq_len"],
                        device=ec["device"], normalize=ec["normalize"])
    emb = embedder.encode(texts, batch_size=ec["batch_size"])
    pos, neg = artifacts.load_anchors(resolve_path(cfg["artifacts"]["jd_query"]))
    df["dense_sim_pos"] = (emb @ pos).astype(np.float32)
    df["dense_sim_neg"] = (emb @ neg).astype(np.float32)
    df["dense_margin"] = df["dense_sim_pos"] - df["dense_sim_neg"]
    df["bm25_score"] = bm25mod.bm25_scores(texts)
    return df.loc[ids]
