"""Config loader. Reads config.yaml and resolves ${data_dir} style interpolation.

Paths in config are relative to the repo root (redrob-ranker/). resolve_path() turns
them into absolute paths anchored at the repo root so scripts work from any cwd.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VAR = re.compile(r"\$\{([^}]+)\}")


def _interp(value: str, scope: dict[str, Any]) -> str:
    # one-level ${key} interpolation against the same mapping (e.g. data_dir)
    def repl(m):
        return str(scope.get(m.group(1), m.group(0)))
    prev = None
    out = value
    while prev != out:
        prev, out = out, _VAR.sub(repl, out)
    return out


def _resolve_paths(paths: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(paths)
    for k, v in paths.items():
        if isinstance(v, str):
            resolved[k] = _interp(v, resolved)
    return resolved


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    path = path or os.path.join(REPO_ROOT, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "paths" in cfg:
        cfg["paths"] = _resolve_paths(cfg["paths"])
    return cfg


def resolve_path(p: str) -> str:
    """Absolute path anchored at repo root (leaves already-absolute paths intact)."""
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO_ROOT, p))
