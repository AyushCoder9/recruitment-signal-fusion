#!/usr/bin/env python3
"""Phase 3 — precompute. Offline, network/CPU/GPU all allowed (NOT the ranking step).

Produces the artifacts rank.py consumes:
  artifacts/embeddings.npz   candidate embeddings keyed by id
  artifacts/jd_query.npz     JD ideal + anti anchors
  artifacts/features.parquet static features + precomputed query features (dense, bm25)

Because the JD query is fixed, dense_sim and bm25_score are precomputed into the feature
store so the cached ranking path is a trivial parquet load. Re-runs reuse a cached
embeddings.npz unless --reembed is passed.

Usage:
  python precompute.py [--limit N] [--reembed]
"""
from __future__ import annotations

import argparse
import time
from datetime import date

import numpy as np
import pandas as pd

from src.embed import Embedder, build_anchors
from src.features import bm25 as bm25mod
from src.features.registry import extract_static, get_reference_now
from src.features.text import build_candidate_text
from src.io import artifacts
from src.io.config import load_config, resolve_path
from src.io.loaders import iter_candidates


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap candidates (smoke test)")
    ap.add_argument("--reembed", action="store_true", help="ignore cached embeddings")
    args = ap.parse_args()

    cfg = load_config()
    t0 = time.time()
    ref_now: date = get_reference_now()
    emb_path = resolve_path(cfg["artifacts"]["embeddings"])
    anchor_path = resolve_path(cfg["artifacts"]["jd_query"])
    feat_path = resolve_path(cfg["artifacts"]["features"])

    # ---- pass 1: stream candidates -> ids, texts, static feature rows ----
    ids, texts, rows = [], [], []
    for c in iter_candidates(limit=args.limit):
        ids.append(c.candidate_id)
        texts.append(build_candidate_text(c))
        rows.append(extract_static(c, ref_now))
    _log(f"parsed + featurized {len(ids):,} candidates", t0)
    feat_df = pd.DataFrame(rows).set_index("candidate_id")

    # ---- embeddings (cached unless --reembed) ----
    ec = cfg["embedding"]
    cached_ids = None
    if not args.reembed:
        import os
        if os.path.exists(emb_path):
            cids, cemb = artifacts.load_embeddings(emb_path)
            if list(cids) == ids:
                emb = cemb
                cached_ids = cids
                _log("loaded cached embeddings", t0)
    if cached_ids is None:
        _log(f"loading embedding model {ec['model']} ...", t0)
        embedder = Embedder(ec["model"], dim=ec["dim"], max_seq_len=ec["max_seq_len"],
                            device=ec["device"], normalize=ec["normalize"])
        _log(f"embedding {len(texts):,} on device={embedder.device} in chunks ...", t0)
        chunk = int(ec.get("chunk_size", 5000))
        parts = []
        for i in range(0, len(texts), chunk):
            parts.append(embedder.encode(texts[i:i + chunk], batch_size=ec["batch_size"]))
            _log(f"  embedded {min(i + chunk, len(texts)):,}/{len(texts):,}", t0)
        emb = np.vstack(parts).astype(np.float32)
        artifacts.save_embeddings(emb_path, ids, emb)
        _log(f"embedded -> {emb.shape}, saved {emb_path}", t0)
    else:
        embedder = None

    # ---- JD anchors ----
    if embedder is None:
        embedder = Embedder(ec["model"], dim=ec["dim"], max_seq_len=ec["max_seq_len"],
                            device=ec["device"], normalize=ec["normalize"])
    pos, neg = build_anchors(embedder)
    artifacts.save_anchors(anchor_path, pos, neg)
    _log(f"built anchors -> {anchor_path}", t0)

    # ---- precompute query features (JD is fixed) ----
    feat_df["dense_sim_pos"] = (emb @ pos).astype(np.float32)
    feat_df["dense_sim_neg"] = (emb @ neg).astype(np.float32)
    feat_df["dense_margin"] = feat_df["dense_sim_pos"] - feat_df["dense_sim_neg"]
    feat_df["bm25_score"] = bm25mod.bm25_scores(texts)
    _log("computed dense + bm25 query features", t0)

    artifacts.save_features(feat_path, feat_df)
    _log(f"saved feature store -> {feat_path}  shape={feat_df.shape}", t0)
    print("\nPRECOMPUTE DONE. columns:", len(feat_df.columns))
    print("dense_sim_pos  describe:\n", feat_df["dense_sim_pos"].describe())


if __name__ == "__main__":
    main()
