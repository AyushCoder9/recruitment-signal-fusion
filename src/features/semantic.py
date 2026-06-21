"""Semantic-relevance features.

Two parts:
  - STATIC (computed here, Phase 2): must-have-term overlap ratio. ONE weak signal only —
    deliberately never the primary, to avoid the keyword-stuffer trap.
  - QUERY-TIME (filled in Phase 3 precompute / Phase 8 rank): dense cosine similarity of the
    candidate embedding to a JD 'ideal-candidate' anchor (and to a NEGATIVE anchor), plus a
    BM25 lexical score. Those need the embedding model / index, so only the helpers live here.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from ..io.schema import Candidate
from . import lexicons as lx
from .text import build_candidate_text, full_text  # noqa: F401 (re-exported)

# JD must-have groups (each group = one capability; overlap = groups present / total).
MUST_HAVE_GROUPS = [
    lx.EMBEDDING_RETRIEVAL_TERMS,                 # embeddings-based retrieval
    lx.VECTOR_DB_TERMS,                           # vector DB / hybrid search
    lx.EVAL_FRAMEWORK_TERMS,                      # ranking evaluation (NDCG/MRR/MAP/AB)
    ["python"],                                   # strong python
    ["ranking", "recommendation", "search", "recsys", "retrieval"],  # ranking/recsys
]


def features(c: Candidate, ref_now: date) -> dict:
    text = full_text(c)
    present = sum(1 for grp in MUST_HAVE_GROUPS if lx.contains_any(text, grp))
    return {
        "must_have_overlap": present / len(MUST_HAVE_GROUPS),
        "must_have_count": float(present),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine of matrix `a` (n,d) against single vector `b` (d,) — assumes
    L2-normalized inputs (we normalize at embed time), so this is just a dot product."""
    return a @ b


def dense_features(cand_vecs: np.ndarray, anchor_pos: np.ndarray,
                   anchor_neg: np.ndarray) -> dict[str, np.ndarray]:
    """Query-time dense features for a batch of candidate embeddings."""
    pos = cosine(cand_vecs, anchor_pos)
    neg = cosine(cand_vecs, anchor_neg)
    return {
        "dense_sim_pos": pos,
        "dense_sim_neg": neg,
        "dense_margin": pos - neg,   # how much more 'ideal' than 'anti-ideal'
    }
