"""Streaming candidate loaders.

The full candidates.jsonl is ~465MB / 100k rows — never load it whole. iter_candidates
yields parsed Candidate objects one at a time. load_sample_candidates loads the small
50-row JSON array for tests/fixtures.
"""
from __future__ import annotations

import json
from typing import Iterator

from .config import load_config, resolve_path
from .schema import Candidate


def iter_raw(path: str | None = None) -> Iterator[dict]:
    """Yield raw candidate dicts from a JSONL file, streaming line by line."""
    if path is None:
        path = resolve_path(load_config()["paths"]["candidates_jsonl"])
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_candidates(path: str | None = None, limit: int | None = None) -> Iterator[Candidate]:
    """Yield validated Candidate models, streaming. `limit` caps for quick runs."""
    for i, raw in enumerate(iter_raw(path)):
        if limit is not None and i >= limit:
            break
        yield Candidate.model_validate(raw)


def load_sample_candidates(path: str | None = None) -> list[Candidate]:
    if path is None:
        path = resolve_path(load_config()["paths"]["sample_candidates"])
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Candidate.model_validate(c) for c in data]


def count_candidates(path: str | None = None) -> int:
    return sum(1 for _ in iter_raw(path))
