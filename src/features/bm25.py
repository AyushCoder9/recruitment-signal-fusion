"""BM25 lexical scoring. The JD query is fixed, so for the cached 100k we precompute a
single bm25_score column; for arbitrary sandbox uploads we build a small index live.
"""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOK = re.compile(r"[a-z0-9+#./-]+")

# JD query terms (the must-have vocabulary) used to score lexical relevance.
JD_QUERY = ("machine learning embeddings retrieval ranking recommendation search "
            "vector database hybrid search information retrieval nlp evaluation ndcg mrr "
            "map ab testing production deployed real users python fine-tuning learning to "
            "rank sentence transformers faiss elasticsearch")


def tokenize(text: str) -> list[str]:
    return _TOK.findall((text or "").lower())


def bm25_scores(corpus_texts: list[str], query: str = JD_QUERY,
                k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    corpus = [tokenize(t) for t in corpus_texts]
    bm = BM25Okapi(corpus, k1=k1, b=b)
    scores = bm.get_scores(tokenize(query))
    return np.asarray(scores, dtype=np.float32)
