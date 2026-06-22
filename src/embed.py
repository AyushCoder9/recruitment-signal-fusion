"""Embedding model wrapper (BGE-M3 dense) + JD anchor construction.

Embeddings are precomputed (offline, network OK), so quality > speed. We L2-normalize and
MRL-truncate to config dim so cosine == dot product and memory stays small. The JD anchors
are hand-written from the JD (must-haves + ideal profile + explicit anti-profile) — fully
deterministic, no API dependency.
"""
from __future__ import annotations

import numpy as np


def pick_device(pref: str = "auto") -> str:
    import torch
    if pref and pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class Embedder:
    def __init__(self, model_name: str, dim: int = 768, max_seq_len: int = 512,
                 device: str = "auto", normalize: bool = True):
        from sentence_transformers import SentenceTransformer
        self.device = pick_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_len
        self.dim = dim
        self.normalize = normalize

    def encode(self, texts: list[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                normalize_embeddings=False, show_progress_bar=show_progress)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape[1] > self.dim:           # MRL truncation
            emb = emb[:, :self.dim]
        if self.normalize:
            emb = l2norm(emb)
        return emb.astype(np.float32)


# ---- JD anchors (the "ideal candidate" the JD describes, and its anti-profile) ----
IDEAL_DESCRIPTIONS = [
    "Senior AI / Machine Learning Engineer with 6 to 8 years experience, 4 to 5 of them "
    "in applied ML at product companies. Built and shipped end-to-end ranking, search, and "
    "recommendation systems to real users at scale. Deep production experience with "
    "embeddings-based retrieval (sentence-transformers, BGE, E5), vector databases and hybrid "
    "search (FAISS, Elasticsearch, OpenSearch, Pinecone, Qdrant). Handles embedding drift, "
    "index refresh, retrieval-quality regression. Strong Python and code quality.",
    "Applied ML engineer who designs rigorous evaluation frameworks for ranking systems: "
    "NDCG, MRR, MAP, offline-to-online correlation, A/B testing. Strong NLP and information "
    "retrieval background. Has opinions on hybrid vs dense retrieval and when to fine-tune vs "
    "prompt, defended with systems actually built in production. Scrappy product engineer who "
    "ships fast, located in or willing to relocate to Pune or Noida, active in the job market.",
    "Backend / data engineer transitioning into ML who built a production recommendation or "
    "semantic search system, learning-to-rank with gradient boosting, real-time serving with "
    "low latency, pre-LLM-era machine learning depth (xgboost, collaborative filtering, "
    "word2vec) plus modern retrieval and LLM integration.",
]
ANTI_DESCRIPTIONS = [
    "Marketing manager, sales lead, HR manager, recruiter, graphic designer, content writer, "
    "accountant, civil or mechanical engineer. Lists many AI keywords on the skills section "
    "but the job titles and career history are not engineering or machine learning.",
    "Career entirely at IT-services and consulting firms (TCS, Infosys, Wipro, Accenture, "
    "Cognizant, Capgemini). Pure academic research with no production deployment. Computer "
    "vision, speech or robotics specialist with no NLP or information-retrieval experience. "
    "Only recent LangChain wrappers calling OpenAI, no pre-LLM machine learning depth.",
]


def build_anchors(embedder: Embedder) -> tuple[np.ndarray, np.ndarray]:
    pos = l2norm(embedder.encode(IDEAL_DESCRIPTIONS).mean(axis=0, keepdims=True))[0]
    neg = l2norm(embedder.encode(ANTI_DESCRIPTIONS).mean(axis=0, keepdims=True))[0]
    return pos.astype(np.float32), neg.astype(np.float32)
