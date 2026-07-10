#!/usr/bin/env python3
"""Replicate-level linear direction benchmark with explicit rate masks.

This is the apples-to-apples replacement for the historical rate table, which
mixed exact fixed-rate atoms for one model with broad rate windows for another.
One 54-D rarefaction-curve vector (mean and standard deviation across depths)
represents each replicate.  All cross-validation
predictions hold out whole replicates, and every reported stratum includes its
sample size, mask definition, and Wilson interval.
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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import dnnaic
from appreciable_gate import APPRECIABLE, folds

CLASSES = np.array(["A", "B", "C"])
FIXED = (5e-7, 2.5e-6, 5e-5, 2.5e-4)


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


def replicate_table(X, direction, groups, magnitude):
    """Represent each replicate by mean and SD of each 27-D curve coordinate."""
    uniq, first, inv = np.unique(groups, return_index=True, return_inverse=True)
    order = np.lexsort((X[:, 0], inv))
    per = np.bincount(inv)
    if per.min() != per.max():
        raise ValueError("every replicate must carry the same rarefaction-depth grid")
    table = X[order].reshape(len(uniq), int(per[0]), X.shape[1])[:, :, 1:]
    features = np.concatenate([table.mean(axis=1), table.std(axis=1)], axis=1)
    return features, direction[first], magnitude[first]


def repeated_oof(X, y, seeds):
    prob = []
    for seed in seeds:
        ps = np.full((len(y), len(np.unique(y))), np.nan)
        for train, test in folds(len(y), seed):
            scale = StandardScaler().fit(X[train])
            model = LogisticRegression(max_iter=3000, C=1.0).fit(
                scale.transform(X[train]), y[train])
            ps[test] = model.predict_proba(scale.transform(X[test]))
        if np.isnan(ps).any():
            raise RuntimeError("incomplete out-of-fold predictions")
        prob.append(ps)
    return np.mean(prob, axis=0)


def wilson_ci(correct, z=1.959963984540054):
    """Wilson 95% interval for a conditional replicate-level accuracy."""
    n = len(correct)
    p = float(np.sum(correct)) / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [round(float(max(0.0, center - half)), 4),
            round(float(min(1.0, center + half)), 4)]


def paired_bootstrap_difference(first_correct, second_correct, seed=260709, draws=10_000):
    """Conditional paired interval with fitted OOF predictions held fixed."""
    delta = np.asarray(first_correct, float) - np.asarray(second_correct, float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for draw in range(draws):
        estimates[draw] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return {
        "estimate": round(float(delta.mean()), 4),
        "resampling_unit": "simulation replicate", "seed": int(seed), "draws": int(draws),
        "conditional_paired_bootstrap_95": [
            round(float(np.percentile(estimates, 2.5)), 4),
            round(float(np.percentile(estimates, 97.5)), 4),
        ],
    }


def retain_moments(features, retained):
    """Select mean/variance/SE channels from both depth-summary halves of the 54-D vector."""
    width = features.shape[1] // 2
    use = np.isin(np.arange(width) % 3, retained)
    columns = np.concatenate((np.where(use)[0], width + np.where(use)[0]))
    return features[:, columns]


def score_mask(y, pred, use, definition):
    correct = pred[use] == y[use]
    return {
        "definition": definition,
        "n": int(use.sum()),
        "accuracy": round(float(correct.mean()), 4),
        "wilson_95": wilson_ci(correct),
    }


def report(y, prob, magnitude):
    pred = prob.argmax(1)
    result = {
        "overall": score_mask(y, pred, np.ones(len(y), dtype=bool), "all positive-rate replicates"),
        "appreciable": score_mask(y, pred, magnitude >= APPRECIABLE,
                                   "migration rate >= 2.5e-4"),
        "fixed_atoms": {},
        "plus_minus_50pct_bands": {},
        "confusion_all": confusion_matrix(y, pred, labels=list(range(prob.shape[1]))).tolist(),
    }
    for rate in FIXED:
        exact = np.abs(magnitude - rate) <= rate * 1e-9
        band = np.abs(magnitude - rate) <= 0.5 * rate
        key = f"{rate:.1e}"
        result["fixed_atoms"][key] = score_mask(
            y, pred, exact, f"abs(m-rate) <= 1e-9*rate; rate={rate:.8g}")
        result["plus_minus_50pct_bands"][key] = score_mask(
            y, pred, band, f"abs(m-rate) <= 0.5*rate; center={rate:.8g}")
    return result


def fit_full_model(X, y):
    scale = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=3000, C=1.0).fit(scale.transform(X), y)
    return scale, model


def fitted_model_record(scale, model):
    """Serializable final full-data fit for direct coefficient/provenance audit."""
    def serial(values):
        return np.round(np.asarray(values, dtype=float), 8).tolist()
    return {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 3000,
        "class_weight": None,
        "classes": model.classes_.tolist(),
        "coefficient_serialization_decimals": 8,
        "scaler_mean": serial(scale.mean_), "scaler_scale": serial(scale.scale_),
        "coef": serial(model.coef_), "intercept": serial(model.intercept_),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="regen_full")
    parser.add_argument("--external", default="regen_extra_round1,regen_extra_round2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out")
    args = parser.parse_args()

    X, direction, groups, magnitude = dnnaic.load_dataset(args.dataset)
    Xrep, rep_direction, rep_magnitude = replicate_table(X, direction, groups, magnitude)
    positive = rep_direction != "D"
    Xp, dp, mp = Xrep[positive], rep_direction[positive], rep_magnitude[positive]
    y = np.searchsorted(CLASSES, dp)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    prob = repeated_oof(Xp, y, seeds)

    # Binary B-versus-C is the pure donor/recipient reversal on one fixed pair.
    bc = np.isin(dp, ["B", "C"])
    ybc = (dp[bc] == "C").astype(int)
    pbc = repeated_oof(Xp[bc], ybc, seeds)[:, 1]
    full_scale, full_model = fit_full_model(Xp, y)
    bc_scale, bc_model = fit_full_model(Xp[bc], ybc)
    bc_result = report(ybc, np.column_stack([1.0 - pbc, pbc]), mp[bc])
    bc_result["interpretation"] = "B=P2->P3 versus C=P3->P2; same population pair and exposure duration"

    result = {
        "dataset": args.dataset,
        "unit": "simulation replicate",
        "feature_representation": "mean and SD of each 27-D rarefaction-curve coordinate (54-D)",
        "model": "standardized multinomial logistic regression; repeated five-fold OOF",
        "seeds": list(seeds),
        "three_scenario": report(y, prob, mp),
        "binary_reversed_pair": bc_result,
    }

    primary_correct = prob.argmax(axis=1) == y
    ablation = {}
    ablation_correctness = {}
    for name, retained in (("across_locus_mean", (0,)),
                           ("across_locus_variance", (1,)),
                           ("mean_plus_variance", (0, 1))):
        Xa = retain_moments(Xp, retained)
        pa = repeated_oof(Xa, y, seeds)
        ablation_correct = pa.argmax(axis=1) == y
        ablation_correctness[name] = ablation_correct
        ablation[name] = {
            "dimension": int(Xa.shape[1]),
            "retained_moment_indices": list(retained),
            "report": report(y, pa, mp),
            "full_minus_ablation_accuracy": paired_bootstrap_difference(
                primary_correct, ablation_correct),
        }
    ablation["mean_variance_se"] = {
        "dimension": int(Xp.shape[1]),
        "retained_moment_indices": [0, 1, 2],
        "report": result["three_scenario"],
    }
    result["matched_moment_ablation"] = ablation
    result["matched_summary_comparisons"] = {
        "mean_plus_variance_minus_mean": paired_bootstrap_difference(
            ablation_correctness["mean_plus_variance"],
            ablation_correctness["across_locus_mean"]),
        "mean_plus_variance_minus_variance": paired_bootstrap_difference(
            ablation_correctness["mean_plus_variance"],
            ablation_correctness["across_locus_variance"]),
        "full_minus_mean_plus_variance": paired_bootstrap_difference(
            primary_correct, ablation_correctness["mean_plus_variance"]),
    }
    result["final_canonical_fit"] = {
        "three_scenario": fitted_model_record(full_scale, full_model),
        "binary_reversed_pair": fitted_model_record(bc_scale, bc_model),
    }

    external = {}
    pooled_y, pooled_prob, pooled_m = [], [], []
    pooled_ybc, pooled_pbc, pooled_mbc = [], [], []
    for name in [n for n in args.external.split(",") if n]:
        Xe, de, ge, me = dnnaic.load_dataset(name)
        Xerep, derep, merep = replicate_table(Xe, de, ge, me)
        keep = derep != "D"
        ye = np.searchsorted(CLASSES, derep[keep])
        pe = full_model.predict_proba(full_scale.transform(Xerep[keep]))
        bce = np.isin(derep, ["B", "C"])
        ybce = (derep[bce] == "C").astype(int)
        pbce = bc_model.predict_proba(bc_scale.transform(Xerep[bce]))
        external[name] = {
            "three_scenario": report(ye, pe, merep[keep]),
            "binary_reversed_pair": report(ybce, pbce, merep[bce]),
        }
        pooled_y.append(ye); pooled_prob.append(pe); pooled_m.append(merep[keep])
        pooled_ybc.append(ybce); pooled_pbc.append(pbce); pooled_mbc.append(merep[bce])
    if pooled_y:
        external["pooled"] = {
            "three_scenario": report(np.concatenate(pooled_y), np.vstack(pooled_prob),
                                     np.concatenate(pooled_m)),
            "binary_reversed_pair": report(np.concatenate(pooled_ybc), np.vstack(pooled_pbc),
                                             np.concatenate(pooled_mbc)),
        }
    result["independent_batch_evaluation"] = external
    result["provenance"] = provenance((args.dataset, *[n for n in args.external.split(",") if n]))

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
