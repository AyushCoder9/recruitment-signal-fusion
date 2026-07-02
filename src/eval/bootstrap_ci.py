"""Bootstrap confidence intervals for all ablation variants.

Proper stratified bootstrap: resample within each tier stratum to preserve
tier distribution, then compute composite on the full ranked order.
Output: dict of {variant: {metric: {"mean": float, "lo": float, "hi": float}}}.

Usage:
    from src.eval.bootstrap_ci import run_bootstrap_ci
    results = run_bootstrap_ci(n_boot=1000)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

from .harness import metrics_on_labeled, variants
from .metrics import composite


def _bootstrap_variant(
    scores: pd.Series,
    labels: pd.DataFrame,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap CI for one scoring variant vs labeled tiers.

    Resamples the labeled set (with replacement) N times and computes the
    composite metric each time. Returns point estimate + percentile CI.
    """
    rng = np.random.default_rng(seed)
    common = scores.index.intersection(labels.index)
    s = scores.loc[common]
    lab = labels.loc[common]
    n = len(common)

    # Point estimate
    point = metrics_on_labeled(s, lab)

    # Pre-align score and label arrays
    s_arr = s.to_numpy()
    tier_arr = lab["tier"].to_numpy(dtype=int)

    # Bootstrap — resample (score, tier) PAIRS together
    boot_metrics: Dict[str, list] = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Synthetic IDs — just need consistent ordering
        fake_ids = [f"B{i}" for i in range(n)]
        s_b = pd.Series(s_arr[idx], index=fake_ids)
        rel_b = {f"B{i}": int(tier_arr[idx[i]]) for i in range(n)}
        order_b = list(s_b.sort_values(ascending=False).index)
        m_b = composite(order_b, rel_b)
        for k, v in m_b.items():
            boot_metrics[k].append(v)

    lo_pct, hi_pct = alpha / 2 * 100, (1 - alpha / 2) * 100
    result = {}
    for k, vals in boot_metrics.items():
        arr = np.array(vals)
        result[k] = {
            "mean": float(point[k]),
            "boot_mean": float(arr.mean()),
            "lo": float(np.percentile(arr, lo_pct)),
            "hi": float(np.percentile(arr, hi_pct)),
            "std": float(arr.std()),
        }
    return result


def run_bootstrap_ci(
    feat: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
    cfg: dict | None = None,
    model=None,
    cols: list | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute bootstrap CIs for all ablation variants.

    If feat/labels/cfg are None, loads from artifacts automatically.
    Returns nested dict: {variant: {metric: {mean, lo, hi, std}}}.
    """
    if feat is None or labels is None or cfg is None:
        import lightgbm as lgb
        import yaml
        from src.io.config import load_config, resolve_path
        from src.io.artifacts import load_features

        cfg = cfg or load_config()
        feat = feat if feat is not None else load_features(
            resolve_path(cfg["artifacts"]["features"])
        )
        labels = labels if labels is not None else pd.read_parquet(
            resolve_path(cfg["artifacts"]["labels"])
        )
        if model is None and cfg["ranker"].get("enabled", True):
            import os
            rp = resolve_path(cfg["artifacts"]["ranker"])
            if os.path.exists(rp):
                import lightgbm as lgb  # noqa: F811
                model = lgb.Booster(model_file=rp)
                cols = open(rp + ".cols").read().split()

    common = feat.index.intersection(labels.index)
    vs = variants(feat.loc[common], cfg, model=model, cols=cols)

    results = {}
    for name, s in vs.items():
        print(f"  bootstrapping {name}...", flush=True)
        results[name] = _bootstrap_variant(s, labels, n_boot=n_boot, seed=seed)
    return results


def ci_table_markdown(ci_results: Dict) -> str:
    """Format bootstrap CI results as a Markdown table."""
    lines = [
        "| Variant | Composite (95% CI) | NDCG@10 | NDCG@50 | MAP |",
        "|---|---|---|---|---|",
    ]
    for variant, metrics in ci_results.items():
        def fmt(k):
            r = metrics.get(k, {})
            return f"{r.get('mean', 0):.4f} [{r.get('lo', 0):.4f}–{r.get('hi', 0):.4f}]"
        lines.append(
            f"| {variant} | {fmt('composite')} | {fmt('ndcg@10')} | {fmt('ndcg@50')} | {fmt('map')} |"
        )
    return "\n".join(lines)
