#!/usr/bin/env python3
"""Moment ablation: does the across-locus VARIANCE add orientation signal beyond the MEAN?

Empirical counterpart of the variance-orients proposition. The rarefaction kernel weights a
private allele linearly in the across-locus mean but quadratically in the across-locus variance,
so two reversed directions can share the same expected private count yet differ in its
concentration. Prediction: mean+variance beats mean-only, and the gain is largest for the
exchangeable reversed pair (B, C).

Leakage-free protocol: GroupKFold(5) by true replicate, replicate-aggregated, logistic
(Fisher-optimal) head, three seeds.

Usage:  DNNAIC_DATA=/path/to/simulation_data  python scripts/moment_ablation.py
"""
import argparse, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import dnnaic

APPRECIABLE = 2.5e-4


def run_subset(X, d, g, m, cols, seeds=(0, 1, 2)):
    uniq, first, inv = np.unique(g, return_index=True, return_inverse=True)
    R = len(uniq); rep_dir = d[first]; rep_mag = m[first]
    pos = d != "D"; idx = np.where(pos)[0]; invd = inv[idx]
    Xc = X[:, cols]; yd = np.searchsorted(dnnaic.CLASSES, d[idx])
    truth = np.searchsorted(dnnaic.CLASSES, rep_dir)
    ov, ap = [], []; conf = np.zeros((3, 3))
    for s in seeds:
        rp = np.zeros((R, 3)); seen = np.zeros(R, bool)
        for tr, te in dnnaic.group_folds(invd, s):
            sc = StandardScaler().fit(Xc[idx][tr])
            clf = LogisticRegression(max_iter=400, C=1.0).fit(sc.transform(Xc[idx][tr]), yd[tr])
            p = clf.predict_proba(sc.transform(Xc[idx][te]))
            num = np.zeros((R, 3)); den = np.zeros(R)
            np.add.at(num, invd[te], p); np.add.at(den, invd[te], 1.0)
            mm = den > 0; rp[mm] = num[mm] / den[mm, None]; seen |= mm
        pred = rp.argmax(1); val = seen & (rep_dir != "D")
        ov.append((pred[val] == truth[val]).mean())
        apm = val & (rep_mag >= APPRECIABLE); ap.append((pred[apm] == truth[apm]).mean())
        if s == 0:
            for t, pp in zip(truth[val], pred[val]):
                conf[t, pp] += 1
    return dict(overall=round(float(np.mean(ov)) * 100, 1),
                appreciable=round(float(np.mean(ap)) * 100, 1),
                BC_confusion=int(conf[1, 2] + conf[2, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="regen_full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    X, d, g, m = dnnaic.load_dataset(args.dataset)
    res = {name: run_subset(X, d, g, m, cols) for name, cols in dnnaic.MOMENT_SETS.items()}
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
