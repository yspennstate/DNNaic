#!/usr/bin/env python3
"""Direction and detection of appreciable gene flow, under the replicate-aware protocol.

Reproduces the paper's main table: three-way direction accuracy per migration rate, in the
appreciable band, and rate-averaged, plus the detection-gate ROC-AUC and expected calibration
error. Evaluation is GroupKFold(5) by true simulation replicate (no replicate shared between
train and test), replicate-aggregated over rarefaction depths. The linear (logistic) head is the
Fisher-optimal one of the linear-optimality proposition; a gradient-boosted head is reported
alongside as an independent, different-software-stack estimator.

Usage:  DNNAIC_DATA=/path/to/simulation_data  python scripts/direction_detection.py
"""
import argparse, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
import dnnaic

FIXED = [5e-7, 2.5e-6, 5e-5, 2.5e-4]
APPRECIABLE = 2.5e-4


def direction_curve(X, d, groups, m, head="logit", seeds=(0, 1, 2)):
    uniq, first, inv = np.unique(groups, return_index=True, return_inverse=True)
    R = len(uniq); rep_dir = d[first]; rep_mag = m[first]
    pos = d != "D"; idx = np.where(pos)[0]; invd = inv[idx]
    yd = np.searchsorted(dnnaic.CLASSES, d[idx]); truth = np.searchsorted(dnnaic.CLASSES, rep_dir)
    per_seed = []
    for s in seeds:
        rp = np.zeros((R, 3)); seen = np.zeros(R, bool)
        for tr, te in dnnaic.group_folds(invd, s):
            if head == "logit":
                sc = StandardScaler().fit(X[idx][tr])
                clf = LogisticRegression(max_iter=400, C=1.0).fit(sc.transform(X[idx][tr]), yd[tr])
                p = clf.predict_proba(sc.transform(X[idx][te]))
            else:
                clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=s)
                clf.fit(X[idx][tr], yd[tr]); p = clf.predict_proba(X[idx][te])
            num = np.zeros((R, 3)); den = np.zeros(R)
            np.add.at(num, invd[te], p); np.add.at(den, invd[te], 1.0)
            mm = den > 0; rp[mm] = num[mm] / den[mm, None]; seen |= mm
        per_seed.append((rp.argmax(1), seen, truth, rep_dir, rep_mag))

    def acc(mask):
        vals = []
        for pred, seen, truth, rd, rm in per_seed:
            v = seen & (rd != "D") & mask
            if v.sum(): vals.append((pred[v] == truth[v]).mean())
        return round(float(np.mean(vals)) * 100, 1)

    rep_mag0 = per_seed[0][4]
    out = {f"{r:.1e}": acc(np.abs(rep_mag0 - r) <= r * 1e-3) for r in FIXED}
    out["appreciable"] = acc(rep_mag0 >= APPRECIABLE)
    out["overall"] = acc(np.ones(len(rep_mag0), bool))
    return out


def detection_gate(X, d, groups, m, seeds=(0, 1, 2)):
    uniq, first, inv = np.unique(groups, return_index=True, return_inverse=True)
    R = len(uniq); rep_dir = d[first]; rep_mag = m[first]
    keep = (d == "D") | (m >= APPRECIABLE); gidx = np.where(keep)[0]; ginv = inv[gidx]
    yb = ((d[gidx] != "D") & (m[gidx] >= APPRECIABLE)).astype(int)
    aucs_all = []; aucs_appr = []; eces = []; briers = []
    for s in seeds:
        gp = np.zeros(R); gs = np.zeros(R, bool)
        for tr, te in dnnaic.group_folds(ginv, s):
            sc = StandardScaler().fit(X[gidx][tr])
            clf = LogisticRegression(max_iter=400).fit(sc.transform(X[gidx][tr]), yb[tr])
            p = clf.predict_proba(sc.transform(X[gidx][te]))[:, 1]
            num = np.zeros(R); den = np.zeros(R); np.add.at(num, ginv[te], p); np.add.at(den, ginv[te], 1.0)
            mm = den > 0; gp[mm] = num[mm] / den[mm]; gs |= mm
        gy = ((rep_dir != "D") & (rep_mag >= APPRECIABLE)).astype(int)
        vv = gs & ((rep_dir == "D") | (rep_mag >= APPRECIABLE))
        aucs_all.append(roc_auc_score(gy[gs], gp[gs]))
        appr = gs & ((rep_dir == "D") | (rep_mag >= APPRECIABLE))
        aucs_appr.append(roc_auc_score(gy[appr], gp[appr]))
        eces.append(dnnaic.expected_calibration_error(gy[gs], gp[gs]))
        briers.append(brier_score_loss(gy[gs], gp[gs]))
    return dict(roc_auc_all=round(float(np.mean(aucs_all)), 3),
                roc_auc_appreciable=round(float(np.mean(aucs_appr)), 3),
                ece=round(float(np.mean(eces)), 3), brier=round(float(np.mean(briers)), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="regen_full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    X, d, g, m = dnnaic.load_dataset(args.dataset)
    res = dict(dataset=args.dataset,
               direction_linear=direction_curve(X, d, g, m, "logit"),
               direction_boosted=direction_curve(X, d, g, m, "hgb"),
               detection_gate=detection_gate(X, d, g, m))
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
