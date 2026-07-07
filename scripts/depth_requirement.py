#!/usr/bin/env python3
"""How does orientation degrade as the rarefaction depth is capped?

Archaic data offers only a few high-coverage genomes, so the common rarefaction depth is very
shallow (two Neanderthals give g<=4). This restricts the simulation rows to g<=g_max and measures,
under the leakage-free protocol, the appreciable-band direction accuracy and the detection-gate
ROC-AUC as a function of g_max. It uses the closed-form Fisher linear discriminant -- the
linear-optimal rule of the theory -- so it is exact and fast. The result quantifies why the
frozen model's gate abstains on the four-gene-copy Neanderthal trios: at g<=4 orientation is
degraded (but not destroyed), which combined with the archaic lineage being out of distribution
puts the real archaic features beyond the model's reliable range.

Usage:  DNNAIC_DATA=/path/to/simulation_data  python scripts/depth_requirement.py
"""
import argparse, json, numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score
import dnnaic

APPRECIABLE = 2.5e-4


def eval_depth(X, d, g, m, g_max, seeds=(0, 1, 2)):
    depth = X[:, 0]
    uniq, first, inv = np.unique(g, return_index=True, return_inverse=True)
    R = len(uniq); rep_dir = d[first]; rep_mag = m[first]
    rowmask = depth <= g_max
    pos = (d != "D") & rowmask; idx = np.where(pos)[0]; invd = inv[idx]
    yd = np.searchsorted(dnnaic.CLASSES, d[idx]); truth = np.searchsorted(dnnaic.CLASSES, rep_dir)
    ap, auc = [], []
    for s in seeds:
        rp = np.zeros((R, 3)); seen = np.zeros(R, bool)
        for tr, te in dnnaic.group_folds(invd, s):
            clf = LDA().fit(X[idx][tr], yd[tr]); p = clf.predict_proba(X[idx][te])
            num = np.zeros((R, 3)); den = np.zeros(R)
            np.add.at(num, invd[te], p); np.add.at(den, invd[te], 1.0)
            mm = den > 0; rp[mm] = num[mm] / den[mm, None]; seen |= mm
        pred = rp.argmax(1); apm = seen & (rep_dir != "D") & (rep_mag >= APPRECIABLE)
        ap.append((pred[apm] == truth[apm]).mean())
        keep = rowmask & ((d == "D") | (m >= APPRECIABLE)); gi = np.where(keep)[0]; gv = inv[gi]
        yb = ((d[gi] != "D") & (m[gi] >= APPRECIABLE)).astype(int); gp = np.zeros(R); gs = np.zeros(R, bool)
        for tr, te in dnnaic.group_folds(gv, s):
            clf = LDA().fit(X[gi][tr], yb[tr]); p = clf.predict_proba(X[gi][te])[:, 1]
            num = np.zeros(R); den = np.zeros(R); np.add.at(num, gv[te], p); np.add.at(den, gv[te], 1.0)
            mm = den > 0; gp[mm] = num[mm] / den[mm]; gs |= mm
        gy = ((rep_dir != "D") & (rep_mag >= APPRECIABLE)).astype(int)
        auc.append(roc_auc_score(gy[gs], gp[gs]))
    return dict(g_max=int(g_max), appreciable_accuracy=round(float(np.mean(ap)) * 100, 1),
                gate_auc=round(float(np.mean(auc)), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="regen_full")
    ap.add_argument("--depths", default="4,6,10,199")
    args = ap.parse_args()
    X, d, g, m = dnnaic.load_dataset(args.dataset)
    res = [eval_depth(X, d, g, m, int(gm)) for gm in args.depths.split(",")]
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
