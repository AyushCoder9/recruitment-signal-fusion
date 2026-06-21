"""Domain lexicons — the encoded JD knowledge. Every term list here is a deliberate
reading of the Redrob JD (must-haves, disqualifiers, ideal profile). Centralized so the
feature modules stay logic-only and the knowledge is auditable in one place.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Role / title taxonomy  (the decisive axis: function from titles+history)
# ---------------------------------------------------------------------------
# On-target engineering/ML/DS/research functions.
TARGET_TITLE_TERMS = [
    "machine learning", "ml engineer", "ml scientist", "applied scientist",
    "applied ml", "data scientist", "data science", "data engineer",
    "ai engineer", "ai/ml", "research engineer", "research scientist",
    "nlp engineer", "nlp scientist", "search engineer", "ranking",
    "recommendation", "recsys", "information retrieval", "ir engineer",
    "software engineer", "backend engineer", "backend developer",
    "platform engineer", "mlops", "ml platform", "deep learning",
    "computer scientist", "staff engineer", "principal engineer",
]
# Clearly off-target functions (JD: "not a fit, no matter how perfect").
OFFTARGET_TITLE_TERMS = [
    "marketing", "sales", "business development", "account manager",
    "human resource", "hr manager", "hr executive", "recruiter", "recruitment",
    "talent acquisition", "graphic design", "designer", "ux designer",
    "ui designer", "content writer", "copywriter", "accountant", "accounting",
    "finance manager", "financial analyst", "civil engineer", "mechanical engineer",
    "electrical engineer", "operations manager", "supply chain", "logistics",
    "customer success", "customer support", "administrative", "teacher",
    "professor",  # academic-only -> handled with research flag too
]
# Management titles (down-weight only if NO recent building — the 18-month rule).
MANAGER_TITLE_TERMS = [
    "engineering manager", "tech lead", "team lead", "director",
    "vp ", "vice president", "head of", "chief", "architect",
]

SENIORITY_MAP = {  # ordinal 0..1, longest keys first when matched
    "intern": 0.1, "junior": 0.25, "associate": 0.35, "":  0.5,
    "senior": 0.7, "lead": 0.75, "staff": 0.85, "principal": 0.92,
    "manager": 0.7, "director": 0.85, "vp": 0.9, "head": 0.85, "chief": 0.95,
}

# ---------------------------------------------------------------------------
# Skill taxonomy (trust-weighted; core = JD must-haves)
# ---------------------------------------------------------------------------
# Core retrieval/ranking/ML skills the JD explicitly wants.
CORE_SKILL_TERMS = [
    "embedding", "embeddings", "sentence-transformers", "sentence transformers",
    "retrieval", "rag", "semantic search", "vector search", "vector database",
    "ranking", "learning to rank", "learning-to-rank", "ltr",
    "recommendation", "recommender", "recsys", "information retrieval",
    "nlp", "natural language processing", "transformer", "transformers",
    "bert", "llm", "large language model", "fine-tuning", "fine tuning",
    "pytorch", "tensorflow", "machine learning", "deep learning",
    "elasticsearch", "opensearch", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "bm25", "okapi",
]
# JD hard must-haves (used for must-have overlap + targeted signals).
VECTOR_DB_TERMS = ["pinecone", "weaviate", "qdrant", "milvus", "faiss",
                   "opensearch", "elasticsearch", "vector database", "vector db",
                   "vector search", "hybrid search"]
EMBEDDING_RETRIEVAL_TERMS = ["embedding", "embeddings", "sentence-transformers",
                             "bge", "e5", "retrieval", "rag", "semantic search",
                             "dense retrieval", "nearest neighbor", "ann"]
EVAL_FRAMEWORK_TERMS = ["ndcg", "mrr", "map@", "mean average precision",
                        "mean reciprocal rank", "a/b test", "ab test", "ab testing",
                        "offline evaluation", "online evaluation", "precision@",
                        "recall@", "learning to rank", "relevance evaluation"]
NLP_IR_TERMS = ["nlp", "natural language", "information retrieval", "retrieval",
                "ranking", "search", "recommendation", "text", "embedding",
                "transformer", "bert", "llm", "semantic"]
CV_SPEECH_ROBOTICS_TERMS = ["computer vision", "image classification",
                            "object detection", "segmentation", "opencv",
                            "speech recognition", "asr", "text-to-speech", "tts",
                            "robotics", "ros", "slam", "lidar", "autonomous",
                            "image processing", "video analytics"]
# Recent LLM-wrapper stack (JD disqualifier if ONLY this + no pre-LLM depth).
# Deliberately the FRAMEWORK tells only — "chatgpt"/"gpt-4"/"prompt engineering" are
# ubiquitous 2026 filler (chatgpt alone matched 63% of profiles) and discriminate nothing.
LANGCHAIN_WRAPPER_TERMS = ["langchain", "llamaindex", "llama-index", "autogpt",
                           "crewai", "agent framework", "agentic framework"]
# Pre-LLM-era ML depth (presence rebuts the langchain-only disqualifier).
PRELLM_ML_TERMS = ["scikit-learn", "sklearn", "xgboost", "lightgbm", "random forest",
                   "logistic regression", "svm", "gradient boosting", "feature engineering",
                   "word2vec", "tf-idf", "tfidf", "lstm", "rnn", "cnn", "spark mllib",
                   "collaborative filtering", "matrix factorization"]

# ---------------------------------------------------------------------------
# Production / scale signal (JD: "deployed to real users", "at scale")
# ---------------------------------------------------------------------------
PRODUCTION_TERMS = ["production", "deployed", "deploy", "real users", "at scale",
                    "scale", "latency", "throughput", "qps", "rps", "serving",
                    "live", "in production", "a/b test", "millions", "billion",
                    "real-time", "real time", "pipeline", "shipped", "launched"]
SCALE_NUM = re.compile(r"\b(\d+)\s*(million|billion|m\b|b\b|k\b|tb|gb|qps|rps)", re.I)

# ---------------------------------------------------------------------------
# Research-only (JD disqualifier: pure research, no production)
# ---------------------------------------------------------------------------
RESEARCH_TERMS = ["research scientist", "phd", "postdoc", "post-doc", "research fellow",
                  "research assistant", "research associate", "laboratory", "research lab",
                  "university", "institute of technology", "academia", "academic"]

# ---------------------------------------------------------------------------
# Company pedigree (JD negative: services-only career)
# ---------------------------------------------------------------------------
SERVICES_COMPANIES = ["tcs", "tata consultancy", "infosys", "wipro", "accenture",
                      "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
                      "ltimindtree", "lti", "mphasis", "persistent systems",
                      "hexaware", "birlasoft", "coforge", "ntt data", "dxc",
                      "larsen & toubro", "l&t infotech", "igate", "syntel",
                      "mastek", "zensar", "cybage"]

# ---------------------------------------------------------------------------
# Location tiers (JD: Pune/Noida preferred; Tier-1 India ok; outside India case-by-case)
# ---------------------------------------------------------------------------
LOC_TIER1 = ["pune", "noida"]                                   # top preference
LOC_TIER2 = ["hyderabad", "mumbai", "delhi", "new delhi", "gurgaon", "gurugram",
             "bangalore", "bengaluru", "ncr"]                   # JD welcomes
LOC_TIER3 = ["chennai", "kolkata", "ahmedabad", "jaipur", "indore", "kochi",
             "coimbatore", "chandigarh"]                        # other India

_WORD = {}


def _wb(term: str) -> re.Pattern:
    """Cached word-boundary-ish matcher. Short alpha tokens get \\b; phrases substring."""
    if term not in _WORD:
        if " " in term or "-" in term or any(ch.isdigit() for ch in term) or len(term) > 4:
            _WORD[term] = re.compile(re.escape(term), re.I)
        else:
            _WORD[term] = re.compile(r"\b" + re.escape(term) + r"\b", re.I)
    return _WORD[term]


def contains_any(text: str, terms: list[str]) -> bool:
    return any(_wb(t).search(text) for t in terms)


def count_any(text: str, terms: list[str]) -> int:
    return sum(1 for t in terms if _wb(t).search(text))
