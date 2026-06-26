---
title: Recruitment Signal Fusion — Candidate Ranker
emoji: 🎯
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Intelligent Candidate Discovery & Ranking — Live Sandbox

Upload a JSONL of up to 100 candidate profiles (schema = `candidate_schema.json`) and the
engine ranks them for a *Senior AI Engineer (founding team)* JD: a ranked table with
fact-grounded reasoning, a per-candidate factor-contribution chart, and a downloadable CSV.

Runs on CPU; embeds uploaded profiles on the fly with `bge-small-en-v1.5`. The decisive signal
is role/function from job titles + career history — so keyword-stuffers and off-target profiles
are demoted, honeypots (impossible timelines) are floored, and reasoning never hallucinates.

Code + full technical report: https://github.com/AyushCoder9/recruitment-signal-fusion
