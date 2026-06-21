"""Skill features — TRUST-weighted, never a raw keyword count (that is the trap).

Each core-skill match is weighted by proficiency x log(duration) x log(endorsements),
then cross-checked against the platform skill_assessment_scores (claims 'expert' but
scored 38 -> discount). Also encodes JD-specific signals: vector-DB / embedding-retrieval
must-haves, eval-framework bonus, NLP-IR depth vs CV/speech/robotics dominance, and the
'LangChain-only recent, no pre-LLM depth' disqualifier.
"""
from __future__ import annotations

import math
from datetime import date

from ..io.schema import Candidate
from . import lexicons as lx
from .text import PROF_WEIGHT, career_text, clamp, full_text, skills_text


def _matches(term_list, name: str) -> bool:
    return lx.contains_any(name, term_list)


def features(c: Candidate, ref_now: date) -> dict:
    text = full_text(c)
    sk_text = skills_text(c)
    car_text = career_text(c)

    # ---- trust-weighted core-skill score ----
    trust = 0.0
    core_hits = 0
    for s in c.skills:
        name = (s.name or "").lower()
        if _matches(lx.CORE_SKILL_TERMS, name):
            core_hits += 1
            pw = PROF_WEIGHT.get(s.proficiency, 0.4)
            dur = math.log1p(min(s.duration_months or 0, 120)) / math.log1p(120)  # 0..1
            endo = math.log1p(min(s.endorsements or 0, 100)) / math.log1p(100)    # 0..1
            trust += pw * (0.4 + 0.4 * dur + 0.2 * endo)
    core_skill_score = clamp(trust / 4.0)  # saturate ~4 strong core skills

    # ---- assessment alignment ----
    # for skills claimed advanced/expert that ALSO have an assessment, penalize big gaps
    scores = c.redrob_signals.skill_assessment_scores or {}
    gaps = []
    for s in c.skills:
        if s.proficiency in ("advanced", "expert") and s.name in scores:
            claim = 0.8 if s.proficiency == "advanced" else 1.0
            assessed = scores[s.name] / 100.0
            gaps.append(max(0.0, claim - assessed))
    assessment_alignment = 1.0 - (sum(gaps) / len(gaps)) if gaps else 1.0  # 1 = aligned/neutral
    assessment_alignment = clamp(assessment_alignment)

    # ---- skill claim inflation (advanced/expert with ~0 months) ----
    inflation = sum(1 for s in c.skills
                    if s.proficiency in ("advanced", "expert") and (s.duration_months or 0) <= 1)

    # ---- JD must-have signals ----
    vector_db_signal = float(lx.contains_any(text, lx.VECTOR_DB_TERMS))
    embedding_retrieval_signal = float(lx.contains_any(text, lx.EMBEDDING_RETRIEVAL_TERMS))
    eval_framework_signal = float(lx.contains_any(text, lx.EVAL_FRAMEWORK_TERMS))

    # ---- NLP/IR depth vs CV/speech/robotics dominance ----
    nlp_ir = lx.count_any(text, lx.NLP_IR_TERMS)
    cv_sr = lx.count_any(text, lx.CV_SPEECH_ROBOTICS_TERMS)
    nlp_ir_depth = clamp(nlp_ir / 6.0)
    cv_speech_robotics_dominance = clamp(cv_sr / max(1, nlp_ir + cv_sr)) if (cv_sr or nlp_ir) else 0.0
    # only "dominant" when CV/SR present AND NLP/IR thin
    cv_dominant_flag = 1.0 if (cv_sr >= 2 and nlp_ir <= 1) else 0.0

    # ---- LangChain-only-recent disqualifier ----
    has_wrapper = lx.contains_any(text, lx.LANGCHAIN_WRAPPER_TERMS)
    has_prellm = lx.contains_any(text, lx.PRELLM_ML_TERMS)
    langchain_only_recent_flag = 1.0 if (has_wrapper and not has_prellm
                                         and not embedding_retrieval_signal) else 0.0

    return {
        "core_skill_score": core_skill_score,
        "core_skill_hits": float(core_hits),
        "assessment_alignment": assessment_alignment,
        "skill_claim_inflation": float(inflation),
        "vector_db_signal": vector_db_signal,
        "embedding_retrieval_signal": embedding_retrieval_signal,
        "eval_framework_signal": eval_framework_signal,
        "nlp_ir_depth": nlp_ir_depth,
        "cv_speech_robotics_dominance": cv_speech_robotics_dominance,
        "cv_dominant_flag": cv_dominant_flag,
        "langchain_only_recent_flag": langchain_only_recent_flag,
        "production_signal": float(lx.contains_any(car_text, lx.PRODUCTION_TERMS)),
        "scale_signal": float(bool(lx.SCALE_NUM.search(car_text))),
    }
