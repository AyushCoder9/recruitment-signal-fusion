#!/usr/bin/env python3
"""Phase 4 — LLM-judge distillation labels (OFFLINE precompute; not the ranking step).

Stratifies ~3k candidates across the tier spectrum (oversampling the rare likely-fits and
ALL honeypots so the ranker sees positives + traps), labels each with a Groq judge applying
the exact JD rubric -> {tier 0-4, reasoning, key_factors}. Checkpointed to labels.jsonl
(resumable) and finalized to labels.parquet. A second model cross-checks a subset for an
agreement metric. Graceful: the rest of the system still runs rubric-only without this.

Usage: python label_with_llm.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.io.artifacts import load_features
from src.io.config import load_config, resolve_path
from src.io.loaders import iter_candidates
from src.llm.clients import make_client
from src.llm.prompts import build_messages


def stratified_sample(feat: pd.DataFrame, n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    idx = feat.index.to_numpy()

    likely_high = feat[(feat.role_target_best >= 1.0) &
                       ((feat.embedding_retrieval_signal > 0) | (feat.eval_framework_signal > 0) |
                        (feat.vector_db_signal > 0))].index.to_numpy()
    mid = feat[(feat.relevant_history_flag > 0) & (feat.role_target_best < 1.0)].index.to_numpy()
    offtarget = feat[(feat.is_offtarget_current > 0) | (feat.all_services_career_flag > 0)].index.to_numpy()
    foreign = feat[feat.in_india_flag == 0].index.to_numpy()
    honeypot = feat[feat.honeypot_flag > 0].index.to_numpy()

    def take(arr, k):
        if len(arr) == 0:
            return np.array([], dtype=object)
        return rng.choice(arr, size=min(k, len(arr)), replace=False)

    picks = set(honeypot.tolist())                      # ALL honeypots
    picks |= set(take(likely_high, int(0.30 * n)).tolist())
    picks |= set(take(mid, int(0.22 * n)).tolist())
    picks |= set(take(offtarget, int(0.25 * n)).tolist())
    picks |= set(take(foreign, int(0.10 * n)).tolist())
    # fill remainder with a uniform random draw for coverage
    if len(picks) < n:
        picks |= set(take(idx, n - len(picks)).tolist())
    out = list(picks)[:n]
    return out


def load_checkpoint(path: str) -> dict:
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["candidate_id"]] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap labels (testing)")
    args = ap.parse_args()

    cfg = load_config()
    j = cfg["llm_judge"]
    seed = cfg["determinism"]["seed"]
    feat = load_features(resolve_path(cfg["artifacts"]["features"]))

    n = args.limit or j["sample_size"]
    sample_ids = set(stratified_sample(feat, n, seed))
    print(f"stratified sample: {len(sample_ids)} candidates")

    # fetch candidate objects for the sample (single stream)
    cands = {c.candidate_id: c for c in iter_candidates() if c.candidate_id in sample_ids}
    print(f"loaded {len(cands)} candidate objects")

    ckpt = resolve_path("artifacts/labels.jsonl")
    done = load_checkpoint(ckpt)
    todo = [cid for cid in sample_ids if cid not in done]
    print(f"already labeled: {len(done)}  to label: {len(todo)}")

    # Stack several Groq models, each with its own daily-token (TPD) budget, over a shared
    # queue. When a model's TPD exhausts it requeues its current item and bows out; the rest
    # finish what the daily budget allows. Resumable across days as TPD resets.
    from src.llm.clients import TPDExhausted
    specs = j["models"]
    clients = [(s, make_client(s)) for s in dict.fromkeys(specs)]
    workers_per = j.get("workers_per_model", 2)
    q = queue.Queue()
    for cid in todo:
        q.put(cid)

    lock = threading.Lock()
    fh = open(ckpt, "a")
    counters = {"done": 0, "ok": 0, "exhausted": []}
    attempts: dict[str, int] = {}                # per-candidate failure count (cap requeues)

    def worker(spec, client):
        while True:
            try:
                cid = q.get_nowait()
            except queue.Empty:
                return
            try:
                res = client.judge(build_messages(cands[cid]))
            except TPDExhausted:
                q.put(cid)                       # let another model take it
                with lock:
                    if spec not in counters["exhausted"]:
                        counters["exhausted"].append(spec)
                        print(f"  [quota exhausted] {spec} bows out", flush=True)
                return
            except Exception:
                res = None
            if res is None:                      # transient failure: requeue (cap 3) so no
                with lock:                       # candidate is silently dropped
                    attempts[cid] = attempts.get(cid, 0) + 1
                    retry = attempts[cid] < 3
                if retry:
                    q.put(cid)
                    continue
            with lock:
                counters["done"] += 1
                if res is not None:
                    counters["ok"] += 1
                    fh.write(json.dumps({"candidate_id": cid, "model": spec, **res}) + "\n")
                    fh.flush()
                if counters["done"] % 25 == 0:
                    print(f"  processed {counters['done']} ok={counters['ok']} "
                          f"(queue~{q.qsize()})", flush=True)

    threads = []
    for spec, client in clients:
        for _ in range(workers_per):
            t = threading.Thread(target=worker, args=(spec, client), daemon=True)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    fh.close()
    print(f"exhausted models: {counters['exhausted']}")

    # finalize parquet
    done = load_checkpoint(ckpt)
    rows = [{"candidate_id": cid, "tier": r["tier"], "reasoning": r.get("reasoning", ""),
             "key_factors": "; ".join(r.get("key_factors", [])), "model": r.get("model", "")}
            for cid, r in done.items() if cid in sample_ids]
    labels = pd.DataFrame(rows).set_index("candidate_id")
    out = resolve_path(cfg["artifacts"]["labels"])
    labels.to_parquet(out)
    print(f"\nsaved {len(labels)} labels -> {out}")
    print("tier distribution:\n", labels.tier.value_counts().sort_index())


if __name__ == "__main__":
    main()
