#!/usr/bin/env python3
"""Replicate-level gate for the rate regime in which direction is reliable.

The historical DNNaic score was trained to separate any positive migration rate
from the zero-migration control and then evaluated on a high-rate subset.  That is
not the operational question.  Here the target is defined directly as

    y = 1{direction in A/B/C and migration rate >= 2.5e-4},

with weak positive rates and controls both in the negative class.  Each biological
replicate is represented once by its rarefaction curve: the 27 non-depth features
at eight log-spaced depths, followed by their across-depth mean (243 coordinates).
All reported training predictions are repeated five-fold out-of-fold predictions,
so no genealogy is scored by a model trained on that genealogy.

Usage:
    DNNAIC_DATA=/path/to/simulation_data python scripts/appreciable_gate.py
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import dnnaic

APPRECIABLE = 2.5e-4


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(datasets):
    root = Path(os.environ.get("DNNAIC_DATA", "data/simulation_data")).resolve()
    runtime = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ("numpy", "scikit-learn"):
        runtime[package] = importlib_metadata.version(package)
    files = {}
    for dataset in datasets:
        files[dataset] = {}
        for filename in ("X.npy", "direction.npy", "groups.npy", "magnitude.npy"):
            path = root / dataset / filename
            files[dataset][filename] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {"runtime": runtime, "input_arrays": files}


def wilson_ci(correct, z=1.959963984540054):
    n = len(correct)
    p = float(np.sum(correct)) / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [round(float(max(0.0, center - half)), 4),
            round(float(min(1.0, center + half)), 4)]


def replicate_table(X, direction, groups, magnitude, curve_depths=8):
    uniq, first, inv = np.unique(groups, return_index=True, return_inverse=True)
    order = np.lexsort((X[:, 0], inv))
    per = np.bincount(inv)
    if per.min() != per.max():
        raise ValueError("every replicate must carry the same rarefaction-depth grid")
    nd = int(per[0])
    table = X[order].reshape(len(uniq), nd, X.shape[1])
    idx = np.unique(np.round(np.geomspace(1, nd - 1, curve_depths)).astype(int))
    curves = table[:, :, 1:]
    features = np.concatenate([curves[:, idx, :].reshape(len(uniq), -1), curves.mean(1)], axis=1)
    return features, direction[first], magnitude[first]


def folds(n, seed, n_splits=5):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold = np.empty(n, dtype=int)
    for f in range(n_splits):
        fold[perm[f::n_splits]] = f
    for f in range(n_splits):
        yield np.where(fold != f)[0], np.where(fold == f)[0]


def ece(y, p, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        use = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1.0))
        if use.any():
            total += use.mean() * abs(float(y[use].mean()) - float(p[use].mean()))
    return total


def repeated_oof(X, y, seeds):
    all_prob = []
    for seed in seeds:
        prob = np.full(len(y), np.nan)
        for train, test in folds(len(y), seed):
            scale = StandardScaler().fit(X[train])
            model = LogisticRegression(max_iter=3000, C=1.0).fit(
                scale.transform(X[train]), y[train])
            prob[test] = model.predict_proba(scale.transform(X[test]))[:, 1]
        if np.isnan(prob).any():
            raise RuntimeError("incomplete out-of-fold predictions")
        all_prob.append(prob)
    return np.mean(all_prob, axis=0)


def metrics(y, p):
    call = p >= 0.5
    pos = y == 1
    neg = ~pos
    correct = call == y
    sensitivity_correct = call[pos]
    specificity_correct = ~call[neg]
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece_10bin": round(float(ece(y, p)), 4),
        "accuracy_at_0.5": round(float(correct.mean()), 4),
        "accuracy_count": [int(correct.sum()), int(len(correct))],
        "accuracy_wilson_95": wilson_ci(correct),
        "sensitivity_at_0.5": round(float(sensitivity_correct.mean()), 4),
        "sensitivity_count": [int(sensitivity_correct.sum()), int(len(sensitivity_correct))],
        "sensitivity_wilson_95": wilson_ci(sensitivity_correct),
        "specificity_at_0.5": round(float(specificity_correct.mean()), 4),
        "specificity_count": [int(specificity_correct.sum()), int(len(specificity_correct))],
        "specificity_wilson_95": wilson_ci(specificity_correct),
    }


def strata(direction, magnitude, p):
    out = {}
    definitions = {
        "control": direction == "D",
        "weak_positive": (direction != "D") & (magnitude < APPRECIABLE),
        "appreciable": (direction != "D") & (magnitude >= APPRECIABLE),
    }
    for rate in (5e-7, 2.5e-6, 5e-5, 2.5e-4):
        definitions[f"fixed_{rate:.1e}"] = (direction != "D") & (np.abs(magnitude - rate) <= rate * 1e-9)
    for name, use in definitions.items():
        called = p[use] >= 0.5
        out[name] = {
            "n": int(use.sum()),
            "mean_score": round(float(p[use].mean()), 4),
            "fraction_called_at_0.5": round(float(called.mean()), 4),
            "called_count": [int(called.sum()), int(len(called))],
            "called_wilson_95": wilson_ci(called),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="regen_full")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--external", default="regen_extra_round1,regen_extra_round2",
                        help="comma-separated independent datasets; empty string disables")
    parser.add_argument("--out")
    args = parser.parse_args()

    X, direction, groups, magnitude = dnnaic.load_dataset(args.dataset)
    Xrep, rep_direction, rep_magnitude = replicate_table(X, direction, groups, magnitude)
    target = ((rep_direction != "D") & (rep_magnitude >= APPRECIABLE)).astype(int)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    prob = repeated_oof(Xrep, target, seeds)
    result = {
        "dataset": args.dataset,
        "unit": "simulation replicate",
        "feature_representation": "27 features at 8 log-spaced depths plus curve mean (243-D)",
        "target": "positive migration with rate >= 2.5e-4 versus weak positive rates plus control",
        "n_replicates": int(len(target)),
        "class_counts": {"appreciable": int(target.sum()), "other": int((target == 0).sum())},
        "seeds": list(seeds),
        "metrics": metrics(target, prob),
        "strata": strata(rep_direction, rep_magnitude, prob),
    }
    external = {}
    names = [name for name in args.external.split(",") if name]
    scale = StandardScaler().fit(Xrep)
    model = LogisticRegression(max_iter=3000, C=1.0).fit(
        scale.transform(Xrep), target)
    if names:
        pooled_y, pooled_p, pooled_d, pooled_m = [], [], [], []
        for name in names:
            Xe, de, ge, me = dnnaic.load_dataset(name)
            Xerep, derep, merep = replicate_table(Xe, de, ge, me)
            ye = ((derep != "D") & (merep >= APPRECIABLE)).astype(int)
            pe = model.predict_proba(scale.transform(Xerep))[:, 1]
            external[name] = {
                "n_replicates": int(len(ye)),
                "metrics": metrics(ye, pe),
                "strata": strata(derep, merep, pe),
            }
            pooled_y.append(ye); pooled_p.append(pe); pooled_d.append(derep); pooled_m.append(merep)
        py, pp = np.concatenate(pooled_y), np.concatenate(pooled_p)
        pd, pm = np.concatenate(pooled_d), np.concatenate(pooled_m)
        external["pooled"] = {
            "n_replicates": int(len(py)),
            "metrics": metrics(py, pp),
            "strata": strata(pd, pm, pp),
        }
    result["independent_batch_evaluation"] = external
    result["final_canonical_fit"] = {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 3000,
        "class_weight": None,
        "classes": model.classes_.tolist(),
        "coefficient_serialization_decimals": 8,
        "scaler_mean": np.round(scale.mean_, 8).tolist(),
        "scaler_scale": np.round(scale.scale_, 8).tolist(),
        "coef": np.round(model.coef_, 8).tolist(),
        "intercept": np.round(model.intercept_, 8).tolist(),
    }
    result["provenance"] = provenance((args.dataset, *names))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
