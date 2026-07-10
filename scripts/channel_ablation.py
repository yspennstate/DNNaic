#!/usr/bin/env python3
"""Feature-channel ablation: which rarefaction statistics carry the orientation signal?

The nine statistics split into three channels: the symmetric allelic richness (alpha), the
private richness (pi), and the pairwise-private richness (pihat). Theorem (symmetric summaries)
predicts that the symmetric richness alone cannot orient gene flow, while the asymmetric private
and pairwise-private channels can. This scores each channel subset on its own.

Leakage-free protocol matching the moment ablation: GroupKFold(5) by true replicate,
replicate-aggregated class probabilities, three seeds. The learner is HistGradientBoosting,
so the comparison is between feature sets rather than between a linear and a nonlinear model.

Usage:  DNNAIC_DATA=/path/to/simulation_data  python scripts/channel_ablation.py
"""
import argparse, json, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import dnnaic

APPRECIABLE = 2.5e-4


def run_subset(X, d, g, m, cols, seeds=(0, 1, 2)):
    uniq, first, inv = np.unique(g, return_index=True, return_inverse=True)
    R = len(uniq); rep_dir = d[first]; rep_mag = m[first]
    pos = d != "D"; idx = np.where(pos)[0]; invd = inv[idx]
    Xc = X[:, cols]; yd = np.searchsorted(dnnaic.CLASSES, d[idx])
    truth = np.searchsorted(dnnaic.CLASSES, rep_dir)
    ov, ap = [], []
    for s in seeds:
        rp = np.zeros((R, 3)); seen = np.zeros(R, bool)
        for tr, te in dnnaic.group_folds(invd, s):
            clf = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                early_stopping=True, n_iter_no_change=15, validation_fraction=0.1,
                random_state=s).fit(Xc[idx][tr], yd[tr])
            p = clf.predict_proba(Xc[idx][te])
            num = np.zeros((R, 3)); den = np.zeros(R)
            np.add.at(num, invd[te], p); np.add.at(den, invd[te], 1.0)
            mm = den > 0; rp[mm] = num[mm] / den[mm, None]; seen |= mm
        pred = rp.argmax(1); val = seen & (rep_dir != "D")
        ov.append((pred[val] == truth[val]).mean())
        apm = val & (rep_mag >= APPRECIABLE); ap.append((pred[apm] == truth[apm]).mean())
    return dict(overall=round(float(np.mean(ov)) * 100, 1),
                appreciable=round(float(np.mean(ap)) * 100, 1),
                n_features=len(cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="regen_full")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    X, d, g, m = dnnaic.load_dataset(args.dataset)
    order = ["richness", "private", "pairwise", "asymmetric", "full"]
    res = {name: run_subset(X, d, g, m, dnnaic.CHANNELS[name], seeds) for name in order}
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
