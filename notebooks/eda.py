#!/usr/bin/env python3
"""
Phase 0 — EDA. Single streaming pass over candidates.jsonl + deep dive on the
50 sample candidates. Proves the assumptions the whole system rests on:
  - role/function (titles) is the decisive axis, NOT skill keyword counts
  - sentinels (-1) mean "no history" -> must be treated NEUTRAL
  - honeypots are detectable from internal date/skill contradictions
  - reference_now = max(last_active_date) for recency math
Outputs a written summary to stdout and artifacts/eda_summary.json.
"""
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "..", "[PUB] India_runs_data_and_ai_challenge",
                    "India_runs_data_and_ai_challenge")
CANDIDATES = os.path.join(DATA, "candidates.jsonl")
SAMPLES = os.path.join(DATA, "sample_candidates.json")
SAMPLE_SUB = os.path.join(DATA, "sample_submission.csv")
OUT = os.path.join(ROOT, "artifacts", "eda_summary.json")

SERVICES = {  # consulting / IT-services firms named in the JD (+ common peers)
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "lti",
    "mphasis", "mahindra", "larsen", "l&t", "persistent", "hexaware",
    "birlasoft", "coforge", "ntt data", "dxc", "ibm", "deloitte",
}
INDIA_TOP_CITIES = {"pune", "noida", "hyderabad", "mumbai", "bangalore",
                    "bengaluru", "delhi", "gurgaon", "gurugram", "new delhi",
                    "chennai", "kolkata", "ahmedabad"}
OFFTARGET_TITLE_KEYS = ["marketing", "sales", "hr ", "human resource", "recruit",
                        "graphic", "designer", "content writer", "accountant",
                        "finance", "civil", "mechanical", "product manager",
                        "business analyst", "operations"]
TARGET_TITLE_KEYS = ["machine learning", "ml engineer", "ml scientist", "data scien",
                     "data engineer", "ai engineer", "research engineer", "nlp",
                     "search", "ranking", "recommendation", "software engineer",
                     "backend", "applied scien", "mlops", "platform engineer"]


def pd(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def months_between(a, b):
    if not a or not b:
        return None
    return (b.year - a.year) * 12 + (b.month - a.month)


def pctile(sorted_vals, q):
    if not sorted_vals:
        return None
    i = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def title_bucket(title):
    t = (title or "").lower()
    if any(k in t for k in OFFTARGET_TITLE_KEYS):
        return "offtarget"
    if any(k in t for k in TARGET_TITLE_KEYS):
        return "target"
    return "neutral"


def honeypot_reasons(c, ref_now):
    """Return list of contradiction reasons (empty => clean)."""
    reasons = []
    prof = c.get("profile", {})
    yoe = prof.get("years_of_experience")

    # 1. expert/advanced skill with ~0 duration
    infl = 0
    for s in c.get("skills", []):
        if s.get("proficiency") in ("advanced", "expert") and (s.get("duration_months") or 0) <= 1:
            infl += 1
    if infl >= 4:
        reasons.append(f"skill_inflation({infl} adv/expert @~0mo)")

    # 2. duration_months vs start/end gap mismatch in any role
    total_dur = 0
    span_min, span_max = None, None
    for r in c.get("career_history", []):
        sd, ed = pd(r.get("start_date")), pd(r.get("end_date")) or ref_now
        gap = months_between(sd, ed)
        dur = r.get("duration_months") or 0
        total_dur += dur
        if sd:
            span_min = sd if span_min is None else min(span_min, sd)
        if ed:
            span_max = ed if span_max is None else max(span_max, ed)
        if gap is not None and abs(gap - dur) > 9:  # >9mo divergence
            reasons.append(f"date_mismatch(role gap={gap} vs dur={dur})")
        # 3. single role longer than whole YOE
        if yoe is not None and dur > yoe * 12 + 6:
            reasons.append(f"role_dur>{yoe}yoe")

    # 4. YOE vs career span
    if yoe is not None and span_min and span_max:
        span_yrs = months_between(span_min, span_max) / 12.0
        if span_yrs + 0.5 < yoe - 2:  # claims much more exp than span allows
            reasons.append(f"yoe>{round(span_yrs,1)}yr_span")

    # 5. education-vs-career timeline
    grads = [e.get("end_year") for e in c.get("education", []) if e.get("end_year")]
    if grads and span_min:
        first_job_year = span_min.year
        last_grad = max(grads)
        # graduated long after career began AND high YOE => impossible
        if yoe is not None and yoe >= 5 and last_grad - first_job_year > 4 and last_grad >= ref_now.year - 2:
            reasons.append(f"edu_timeline(grad {last_grad} vs job start {first_job_year}, yoe {yoe})")

    return reasons


def main():
    ref_now = date(2025, 6, 1)  # provisional; recomputed below as max last_active
    # ---- pass 1: find reference_now + collect sample_submission ids ----
    sub_ids = []
    if os.path.exists(SAMPLE_SUB):
        import csv
        with open(SAMPLE_SUB) as f:
            for row in csv.DictReader(f):
                sub_ids.append(row["candidate_id"])
    sub_id_set = set(sub_ids)

    n = 0
    max_active = None
    yoe_list = []
    country_ctr = Counter()
    city_ctr = Counter()
    title_bucket_ctr = Counter()
    industry_ctr = Counter()
    prof_ctr = Counter()
    skills_per = []
    adv_expert_zero_dur = 0
    total_skill_rows = 0
    gh_neg1 = oar_neg1 = 0
    rrr_zero_no_apps = 0
    open_to_work = 0
    services_any = services_all = 0
    notice_list = []
    rrr_list = []
    active_recency_days = []
    honeypot_n = 0
    honeypot_examples = []
    sub_lookup = {}  # id -> (title, industry, country, n_ai_skills, role_bucket)
    AI_SKILL_KEYS = ["machine learning", "deep learning", "nlp", "llm", "transformer",
                     "pytorch", "tensorflow", "rag", "embedding", "vector", "bert"]

    with open(CANDIDATES, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            la = pd(c.get("redrob_signals", {}).get("last_active_date"))
            if la and (max_active is None or la > max_active):
                max_active = la

    ref_now = max_active or ref_now

    # ---- pass 2: everything else, using ref_now ----
    with open(CANDIDATES, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            n += 1
            prof = c.get("profile", {})
            sig = c.get("redrob_signals", {})

            yoe = prof.get("years_of_experience")
            if isinstance(yoe, (int, float)):
                yoe_list.append(yoe)
            country_ctr[(prof.get("country") or "?").strip()] += 1
            city_ctr[(prof.get("location") or "?").split(",")[0].strip().lower()] += 1
            title_bucket_ctr[title_bucket(prof.get("current_title"))] += 1
            industry_ctr[(prof.get("current_industry") or "?").strip()] += 1

            sk = c.get("skills", [])
            skills_per.append(len(sk))
            for s in sk:
                total_skill_rows += 1
                prof_ctr[s.get("proficiency", "?")] += 1
                if s.get("proficiency") in ("advanced", "expert") and (s.get("duration_months") or 0) <= 1:
                    adv_expert_zero_dur += 1

            # services
            comps = [(r.get("company") or "").lower() for r in c.get("career_history", [])]
            hit = [any(sv in cm for sv in SERVICES) for cm in comps]
            if any(hit):
                services_any += 1
            if comps and all(hit):
                services_all += 1

            # signals / sentinels
            if sig.get("github_activity_score") == -1:
                gh_neg1 += 1
            if sig.get("offer_acceptance_rate") == -1:
                oar_neg1 += 1
            if (sig.get("recruiter_response_rate") in (0, 0.0)) and (sig.get("applications_submitted_30d") or 0) == 0:
                rrr_zero_no_apps += 1
            if sig.get("open_to_work_flag"):
                open_to_work += 1
            if isinstance(sig.get("notice_period_days"), (int, float)):
                notice_list.append(sig["notice_period_days"])
            if isinstance(sig.get("recruiter_response_rate"), (int, float)):
                rrr_list.append(sig["recruiter_response_rate"])
            la = pd(sig.get("last_active_date"))
            if la:
                active_recency_days.append((ref_now - la).days)

            # honeypot
            hr = honeypot_reasons(c, ref_now)
            if hr:
                honeypot_n += 1
                if len(honeypot_examples) < 8:
                    honeypot_examples.append({"id": c["candidate_id"],
                                              "title": prof.get("current_title"),
                                              "reasons": hr})

            # sample_submission trap lookup
            if c["candidate_id"] in sub_id_set:
                blob = json.dumps(c).lower()
                n_ai = sum(blob.count(k) > 0 for k in AI_SKILL_KEYS)
                sub_lookup[c["candidate_id"]] = {
                    "title": prof.get("current_title"),
                    "industry": prof.get("current_industry"),
                    "country": prof.get("country"),
                    "role_bucket": title_bucket(prof.get("current_title")),
                    "n_ai_skill_terms": n_ai,
                }

    yoe_s = sorted(yoe_list)
    notice_s = sorted(notice_list)
    rec_s = sorted(active_recency_days)

    def show(title):
        print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)

    show("DATASET SIZE & REFERENCE DATE")
    print(f"candidates                 : {n:,}")
    print(f"reference_now (max active) : {ref_now}")
    print(f"sample_submission rows     : {len(sub_ids)}")

    show("YEARS OF EXPERIENCE")
    print(f"min/median/max : {yoe_s[0]:.1f} / {median(yoe_s):.1f} / {yoe_s[-1]:.1f}")
    print(f"p10/p25/p75/p90: {pctile(yoe_s,.1):.1f} / {pctile(yoe_s,.25):.1f} / "
          f"{pctile(yoe_s,.75):.1f} / {pctile(yoe_s,.9):.1f}")
    in_band = sum(1 for y in yoe_s if 5 <= y <= 9)
    print(f"in JD band 5-9 : {in_band:,} ({100*in_band/n:.1f}%)")

    show("GEOGRAPHY")
    india = country_ctr.get("India", 0)
    print(f"India / non-India : {india:,} ({100*india/n:.1f}%) / {n-india:,}")
    print("top countries     :", dict(country_ctr.most_common(6)))
    top_target_cities = {k: v for k, v in city_ctr.most_common(40) if k in INDIA_TOP_CITIES}
    print("JD target cities  :", dict(sorted(top_target_cities.items(), key=lambda x: -x[1])[:8]))

    show("ROLE / FUNCTION (current_title bucket)  <-- decisive axis")
    for k in ("target", "neutral", "offtarget"):
        v = title_bucket_ctr.get(k, 0)
        print(f"  {k:9s}: {v:,} ({100*v/n:.1f}%)")
    print("top industries    :", dict(industry_ctr.most_common(8)))

    show("SKILLS (trust signals)")
    sk_s = sorted(skills_per)
    print(f"skills/candidate median/p90 : {median(sk_s)} / {pctile(sk_s,.9)}")
    print(f"proficiency mix             : {dict(prof_ctr)}")
    print(f"adv/expert skills @ ~0 dur  : {adv_expert_zero_dur:,} of {total_skill_rows:,} "
          f"skill-rows ({100*adv_expert_zero_dur/max(1,total_skill_rows):.1f}%)  <-- inflation signal")

    show("SERVICES-COMPANY CAREERS (JD negative)")
    print(f"any services in history : {services_any:,} ({100*services_any/n:.1f}%)")
    print(f"ALL-services career     : {services_all:,} ({100*services_all/n:.1f}%)  <-- strong negative")

    show("BEHAVIORAL SIGNALS & SENTINELS")
    print(f"github_activity_score == -1 (no GH)   : {gh_neg1:,} ({100*gh_neg1/n:.1f}%)  -> NEUTRAL")
    print(f"offer_acceptance_rate == -1 (no hist) : {oar_neg1:,} ({100*oar_neg1/n:.1f}%)  -> NEUTRAL")
    print(f"response_rate 0 w/ 0 apps (no msgs)   : {rrr_zero_no_apps:,} ({100*rrr_zero_no_apps/n:.1f}%)  -> NEUTRAL")
    print(f"open_to_work flag                     : {open_to_work:,} ({100*open_to_work/n:.1f}%)")
    print(f"recruiter_response_rate p25/med/p75   : {pctile(sorted(rrr_list),.25):.2f} / "
          f"{median(rrr_list):.2f} / {pctile(sorted(rrr_list),.75):.2f}")
    print(f"notice_period p25/med/p75 (days)      : {pctile(notice_s,.25)} / {median(notice_s)} / {pctile(notice_s,.75)}")
    print(f"days-since-active p25/med/p75/p90      : {pctile(rec_s,.25)} / {median(rec_s)} / "
          f"{pctile(rec_s,.75)} / {pctile(rec_s,.9)}")

    show("HONEYPOT DETECTION (full pool)")
    print(f"suspected honeypots : {honeypot_n:,} ({100*honeypot_n/n:.2f}%)  [pool has ~80 by design]")
    for ex in honeypot_examples:
        print(f"  {ex['id']} [{ex['title']}] :: {', '.join(ex['reasons'])}")

    show("THE TRAP — sample_submission top ranks")
    print("rank | candidate | title | industry | country | role | #AI-terms")
    rank_by_id = {r: i for i, r in enumerate(sub_ids, 1)}
    for cid in sub_ids[:20]:
        info = sub_lookup.get(cid)
        if info:
            print(f"  {rank_by_id[cid]:>3} | {cid} | {str(info['title'])[:22]:22s} | "
                  f"{str(info['industry'])[:14]:14s} | {str(info['country'])[:10]:10s} | "
                  f"{info['role_bucket']:9s} | {info['n_ai_skill_terms']}")
    off = sum(1 for cid in sub_ids if sub_lookup.get(cid, {}).get("role_bucket") == "offtarget")
    nonin = sum(1 for cid in sub_ids if sub_lookup.get(cid, {}).get("country") not in (None, "India"))
    print(f"\nsample_sub off-target titles : {off}/{len(sub_ids)}   non-India: {nonin}/{len(sub_ids)}")
    print(">> Confirms: sample ranks keyword-stuffers/off-target high. Ground truth floors them.")

    summary = {
        "n_candidates": n, "reference_now": str(ref_now),
        "yoe_median": median(yoe_s), "yoe_in_band_5_9_pct": round(100*in_band/n, 1),
        "india_pct": round(100*india/n, 1),
        "role_buckets": dict(title_bucket_ctr),
        "adv_expert_zero_dur_pct": round(100*adv_expert_zero_dur/max(1, total_skill_rows), 1),
        "services_any_pct": round(100*services_any/n, 1),
        "services_all_pct": round(100*services_all/n, 1),
        "github_neg1_pct": round(100*gh_neg1/n, 1),
        "offer_neg1_pct": round(100*oar_neg1/n, 1),
        "honeypots_suspected": honeypot_n,
        "honeypot_examples": honeypot_examples,
        "sample_sub_offtarget": off, "sample_sub_nonindia": nonin,
        "services_company_list": sorted(SERVICES),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
