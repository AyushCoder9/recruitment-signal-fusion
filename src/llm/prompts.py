"""LLM-judge prompt = the JD rubric, distilled. The hidden ground truth was almost
certainly built by an LLM/recruiter applying this exact JD, so we hand the judge the
same criteria and ask for a 0-4 relevance tier + structured reasoning. Candidate is
serialized compactly to keep token cost (and thus rate-limit pressure) low.
"""
from __future__ import annotations

import json

from ..io.schema import Candidate

RUBRIC = """\
You are a careful technical recruiter scoring candidates for this role:

ROLE: Senior AI Engineer (founding team) at Redrob, a Series-A AI talent-intelligence
startup. They own the INTELLIGENCE LAYER: ranking, retrieval, and candidate-JD matching.
Pune/Noida hybrid. 5-9 yrs (a range, NOT a hard requirement).

WHAT THEY ACTUALLY WANT (must-haves):
- Production embeddings-based RETRIEVAL (sentence-transformers/BGE/E5) deployed to real users.
- Production VECTOR DB / hybrid search (FAISS, Pinecone, Weaviate, Qdrant, Elasticsearch...).
- Strong Python + code quality.
- RANKING EVALUATION frameworks: NDCG, MRR, MAP, A/B testing, offline-to-online.
- Scrappy product-engineer who SHIPS, tilted toward shipper over researcher.

IDEAL (tier 4): 6-8 yrs total, 4-5 in applied ML/AI at PRODUCT companies (not services);
has shipped an end-to-end ranking/search/recommendation system to real users at scale;
strong NLP/IR; designs rigorous evaluation; in or willing to relocate to Pune/Noida; active.

HARD DISQUALIFIERS (push toward tier 0):
- Off-target FUNCTION by title/career: Marketing, Sales, HR, Recruiter, Designer, Content,
  Accountant/Finance, Civil/Mechanical, Operations, generic PM/BA. NOT a fit no matter how
  many AI keywords are listed in their skills. JUDGE BY JOB TITLES + CAREER HISTORY, NOT the
  skills list.
- Pure research/academia with NO production deployment.
- "AI experience" = only recent (<12mo) LangChain-calling-OpenAI, with NO pre-LLM ML depth.
- Senior who moved to "architecture/tech lead" and hasn't written production code in 18mo.
- Entire career at IT-services/consulting (TCS, Infosys, Wipro, Accenture, Cognizant,
  Capgemini, HCL, Tech Mahindra, Mindtree...). Mixed history with product cos is FINE.
- Primary expertise computer vision / speech / robotics WITHOUT NLP/IR.
- Title-chasers (job-hop every ~1.5 yrs for title bumps).
- HONEYPOTS: internally impossible profiles (e.g. role duration longer than total career,
  years-of-experience exceeding the calendar span of their jobs, education timeline that
  can't coexist with claimed experience). These are traps -> tier 0.

READ BETWEEN THE LINES: a candidate who never writes "RAG" or "Pinecone" but whose career
shows they built a recommendation/search system at a product company IS a strong fit. A
candidate with every AI keyword but a "Marketing Manager" title is NOT.

BEHAVIORAL AVAILABILITY (a MODIFIER, not the main signal): someone perfect on paper who
hasn't logged in for months and ignores recruiters is, for hiring, not actually available
— nudge them down. But sentinel values (-1 github, -1 offer_acceptance, 0 response with 0
applications) mean NO HISTORY, not bad — do NOT penalize those.

LOCATION: Pune/Noida best; Hyderabad/Mumbai/Delhi-NCR/Bangalore welcome; other India ok;
outside India case-by-case, no visa sponsorship (down-weight unless exceptional).

TIERS:
4 = ideal / excellent fit (rare).
3 = strong fit, minor gaps.
2 = adjacent / partial (e.g. data engineer transitioning to ML; relevant but thin on
    retrieval/ranking-eval depth).
1 = weak: mostly off-target or a major disqualifier, with only faint relevant signal.
0 = not a fit: off-target function, all-services career, pure research, honeypot, or a
    keyword-stuffer whose titles/history don't support the claims.

Return STRICT JSON only: {"tier": <0-4 int>, "reasoning": "<=40 words, specific facts +
JD connection + honest concerns", "key_factors": ["<short factor>", ...]}
"""


def serialize_candidate(c: Candidate, max_desc: int = 170) -> str:
    p = c.profile
    career = [{
        "title": e.title, "company": e.company,
        "start": e.start_date, "end": e.end_date or "present",
        "duration_months": e.duration_months,
        "desc": (e.description or "")[:max_desc],
    } for e in c.career_history[:5]]
    skills = [{"name": s.name, "prof": s.proficiency, "months": s.duration_months}
              for s in c.skills[:12]]
    edu = [{"degree": e.degree, "field": e.field_of_study, "end": e.end_year}
           for e in c.education[:3]]
    s = c.redrob_signals
    obj = {
        "current_title": p.current_title, "current_company": p.current_company,
        "current_industry": p.current_industry, "headline": p.headline,
        "summary": (p.summary or "")[:280], "years_of_experience": p.years_of_experience,
        "location": p.location, "country": p.country,
        "career_history": career, "skills": skills, "education": edu,
        "signals": {
            "last_active_date": s.last_active_date, "open_to_work": s.open_to_work_flag,
            "recruiter_response_rate": s.recruiter_response_rate,
            "applications_30d": s.applications_submitted_30d,
            "github_activity_score": s.github_activity_score,
            "offer_acceptance_rate": s.offer_acceptance_rate,
            "notice_period_days": s.notice_period_days,
            "willing_to_relocate": s.willing_to_relocate,
            "skill_assessment_scores": s.skill_assessment_scores,
        },
    }
    return json.dumps(obj, ensure_ascii=False)


# Compact rubric (~350 tokens) — used for labeling to stretch the tiny free-tier daily token
# budget (TPD=100k/model). Keeps every disqualifier + the tier scale; trims prose.
COMPACT_RUBRIC = """\
Score a candidate 0-4 for: Senior AI Engineer at Redrob (Pune/Noida), owning ranking/
retrieval/matching. Must-haves: PROD embeddings retrieval + vector DB/hybrid search,
ranking eval (NDCG/MRR/MAP/A·B), strong Python, ships fast. Ideal=6-8yrs, 4-5 applied ML at
PRODUCT cos, shipped a search/recsys/ranking system to real users, NLP/IR, Pune/Noida.

JUDGE BY JOB TITLES + CAREER HISTORY, NOT the skills-keyword list.
Tier 0 (not a fit): off-target function (Marketing/Sales/HR/Designer/Content/Accountant/
Civil/Mechanical/Ops/generic PM-BA) no matter the AI keywords; entire career at IT-services
(TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/HCL/Mindtree...); pure research no prod;
LangChain-only-recent w/ no pre-LLM ML; CV/speech/robotics w/o NLP/IR; HONEYPOTS (impossible
timelines: role longer than career, YOE > calendar span, education-vs-experience contradiction).
Tier 4=ideal; 3=strong minor gaps; 2=adjacent/partial (e.g. data eng transitioning); 1=weak.
A candidate who built a recsys/search system at a product company but never says 'RAG' IS a fit;
a Marketing Manager with every AI keyword is NOT.
Behavioral availability is a down-weight modifier; sentinels (-1 github, -1 offer, 0 response
w/ 0 applications) = NO history, do NOT penalize. Outside India = down-weight (no visa).

Return STRICT JSON only: {"tier":<0-4>,"reasoning":"<=35 words: specific facts + JD link +
honest concern","key_factors":["..."]}"""


def build_messages(c: Candidate, compact: bool = True) -> list[dict]:
    return [
        {"role": "system", "content": COMPACT_RUBRIC if compact else RUBRIC},
        {"role": "user", "content": "CANDIDATE:\n" + serialize_candidate(c)},
    ]
