#!/usr/bin/env python3
"""Does a deep representation with a kernel correction beat the linear head?

This script tests whether the strongest modern regression-and-classification recipe --- a
neural network that learns a representation, followed by a kernel-ridge correction on the
network's own features (the "neural mean + kernel correction" method used for physics
emulators) --- improves DNNaic's direction and magnitude inference over the Fisher-optimal
linear head of Proposition~\ref{prop:linear-optimal}.

It does not. On introgression direction the network underperforms the linear head and the
kernel correction changes no predictions; on magnitude the network matches the linear head in
the recoverable (appreciable) band and the kernel correction does not improve it. A positive
control on synthetic data with a genuinely nonlinear boundary confirms the same code recovers
nonlinear structure and residual signal when they exist --- so the null on the rarefaction
features reflects the information limit (Theorem~\ref{thm:fisher}), not an inert pipeline.

Evaluation is leakage-free: replicates are grouped and split with GroupKFold(5), so no
simulated genealogy is shared between train and test. Each replicate is summarized by its
rarefaction curve (the nine statistics sampled across a set of depths), and every model is
fit only on the training fold, including the kernel bandwidth and ridge. The kernel correction
uses out-of-fold network predictions for its residual targets so the correction is honest.

Usage:  DNNAIC_DATA=/path/to/simulation_data  python scripts/deep_kernel_comparison.py
        [--dataset regen_full] [--seeds 0,1] [--device cpu|cuda] [--out results.json]
"""
import os, sys, json, time, argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:                                    # torch is required for the network
    sys.stderr.write("this comparison needs pytorch: pip install torch\n")
    raise

import dnnaic
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr

CLASSES = np.array(["A", "B", "C"])
FIXED = [5e-7, 2.5e-6, 5e-5, 2.5e-4]
APPRECIABLE = 2.5e-4
DEV = torch.device("cpu")


# --------------------------------------------------------------------------- data
def replicate_curves(X, groups, n_depths_expected=None):
    """Group the per-row 28-D features by replicate and order by depth (column 0).

    Returns (curves, order) where curves has shape (R, n_depths, 27): the nine rarefaction
    statistics (columns 1..27; column 0 is the depth g) as a function of depth, one page per
    replicate. Assumes every replicate carries the same depth grid.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.lexsort((X[:, 0], inv))
    Xo, invo = X[order], inv[order]
    per = np.bincount(invo)
    if per.min() != per.max():
        raise ValueError("replicates have unequal depth grids; this script assumes a shared grid")
    nd = int(per[0])
    T = Xo.reshape(len(uniq), nd, X.shape[1])
    return T[:, :, 1:], uniq, inv


def curve_features(curves, k=8):
    """Replicate feature vector: the statistics at k log-spaced depths, plus the curve mean."""
    nd = curves.shape[1]
    idx = np.unique(np.round(np.geomspace(1, nd - 1, k)).astype(int))
    sub = curves[:, idx, :].reshape(len(curves), -1)
    return np.concatenate([sub, curves.mean(1)], 1)


def group_folds(n, seed, n_splits=5):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold = np.empty(n, int)
    for f in range(n_splits):
        fold[perm[f::n_splits]] = f
    for f in range(n_splits):
        yield np.where(fold != f)[0], np.where(fold == f)[0]


# ------------------------------------------------------------------ network + kernel
class MLP(nn.Module):
    def __init__(s, di, w=256, depth=3, out=3):
        super().__init__()
        s.inp = nn.Linear(di, w)
        s.hid = nn.ModuleList([nn.Linear(w, w) for _ in range(depth - 1)])
        s.out = nn.Linear(w, out)

    def feat(s, x):
        h = F.silu(s.inp(x))
        for l in s.hid:
            h = h + F.silu(l(h))
        return h

    def forward(s, x):
        return s.out(s.feat(x))


def train_mlp(X, y, out, task, epochs=150, seed=0, w=256, depth=3):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y, dtype=torch.long if task == "cls" else torch.float32, device=DEV)
    m = MLP(X.shape[1], w, depth, out).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    n, bs = len(Xt), 256
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            pred = m(Xt[j])
            loss = F.cross_entropy(pred, yt[j]) if task == "cls" else F.mse_loss(pred.squeeze(-1), yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval()
    return m


@torch.no_grad()
def mlp_out(m, X, task):
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    logits = m(Xt)
    p = F.softmax(logits, 1).cpu().numpy() if task == "cls" else logits.squeeze(-1).cpu().numpy()
    return p, m.feat(Xt).cpu().numpy()


def _sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def _matern52(D2, s):
    r2 = D2 / (s * s); r = np.sqrt(np.maximum(r2, 0)); a = np.sqrt(5.0) * r
    return (1 + a + (5.0 / 3.0) * r2) * np.exp(-a)


def kernel_correct(Ftr, Rtr, Fte, seed, nb=1500):
    """Matern-5/2 kernel-ridge correction Ftr->Rtr, predicted at Fte. Subset-of-regressors
    basis for speed; bandwidth (median heuristic) and ridge tuned on an inner split of the
    training fold only."""
    rng = np.random.default_rng(seed)
    mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-9
    Zt = (Ftr - mu) / sd; Ze = (Fte - mu) / sd
    bas = rng.choice(len(Zt), min(nb, len(Zt)), replace=False)
    Zb, Rb = Zt[bas], Rtr[bas]
    med = np.sqrt(np.median(_sqd(Zb, Zb)[np.triu_indices(len(Zb), 1)]) + 1e-12)
    io = rng.permutation(len(Zb)); cut = int(0.8 * len(Zb)); itr, iva = io[:cut], io[cut:]
    best = (np.inf, 1.0, 1e-2)
    for smult in (0.5, 1.0, 2.0):
        s = smult * med
        Kk = _matern52(_sqd(Zb[itr], Zb[itr]), s); Kv = _matern52(_sqd(Zb[iva], Zb[itr]), s)
        for lam in (1e-4, 1e-3, 1e-2, 1e-1):
            c = cho_factor(Kk + lam * len(itr) * np.eye(len(itr)), lower=True, check_finite=False)
            a = cho_solve(c, Rb[itr], check_finite=False)
            e = np.mean((Rb[iva] - Kv @ a) ** 2)
            if e < best[0]:
                best = (e, smult, lam)
    _, smult, lam = best; s = smult * med
    Kk = _matern52(_sqd(Zb, Zb), s)
    c = cho_factor(Kk + lam * len(Zb) * np.eye(len(Zb)), lower=True, check_finite=False)
    a = cho_solve(c, Rb, check_finite=False)
    return _matern52(_sqd(Ze, Zb), s) @ a


# ------------------------------------------------------------------ stacked OOF drivers
def oof_linear_cls(X, y, seed, k=3):
    P = np.zeros((len(X), k))
    for tr, te in group_folds(len(X), seed):
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
        clf = LogisticRegression(max_iter=500).fit((X[tr] - mu) / sd, y[tr])
        tmp = np.zeros((len(te), k)); tmp[:, clf.classes_] = clf.predict_proba((X[te] - mu) / sd)
        P[te] = tmp
    return P


def oof_mlp(X, y, seed, out, task, epochs):
    P = np.zeros(len(X)) if task == "reg" else np.zeros((len(X), out))
    Fte_all = None
    for tr, te in group_folds(len(X), seed):
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
        if task == "reg":
            ym, ys = y[tr].mean(), y[tr].std() + 1e-9
            m = train_mlp((X[tr] - mu) / sd, (y[tr] - ym) / ys, out, task, epochs, seed)
            p, f = mlp_out(m, (X[te] - mu) / sd, task); P[te] = p * ys + ym
        else:
            m = train_mlp((X[tr] - mu) / sd, y[tr], out, task, epochs, seed)
            p, f = mlp_out(m, (X[te] - mu) / sd, task); P[te] = p
        if Fte_all is None:
            Fte_all = np.zeros((len(X), f.shape[1]))
        Fte_all[te] = f
    return P, Fte_all


def oof_kernel_on_base(F, base, target, seed, task, out=3):
    """Correct `base` predictions with a kernel on features F. Residual target is
    (onehot(y) - base) for classification or (y - base) for regression."""
    resid = (np.eye(out)[target] - base) if task == "cls" else (target - base)
    corrected = base.copy()
    for tr, te in group_folds(len(F), seed + 137):
        corr = kernel_correct(F[tr], resid[tr], F[te], seed + 7)
        corrected[te] = base[te] + corr
    return corrected


# ------------------------------------------------------------------ metrics
def direction_table(pred, y, mag):
    bands = {f"{r:.1e}": np.abs(mag - r) <= r * 1e-3 for r in FIXED}
    bands["appreciable"] = mag >= APPRECIABLE
    bands["overall"] = np.ones(len(mag), bool)
    return {b: round(100 * float((pred[m] == y[m]).mean()), 1) for b, m in bands.items() if m.sum()}


def magnitude_metrics(pred_log, true_log, mag, band):
    pr, tr = 10 ** pred_log[band], 10 ** true_log[band]
    sp = float(spearmanr(pred_log[band], true_log[band]).correlation)
    return dict(MRE=round(float(np.mean(np.abs(pr - tr) / tr)), 3),
                medRE=round(float(np.median(np.abs(pr - tr) / tr)), 3),
                spearman=round(sp, 3), n=int(band.sum()))


# ------------------------------------------------------------------ experiments
def run_direction(Xrep, rep_dir, rep_mag, seeds, epochs):
    pos = rep_dir != "D"; idx = np.where(pos)[0]
    y = np.searchsorted(CLASSES, rep_dir[idx]); X = Xrep[idx]; mag = rep_mag[idx]
    acc = {n: [] for n in ("linear", "network", "network+kernel")}
    for s in seeds:
        Pl = oof_linear_cls(X, y, s)
        P0, F0 = oof_mlp(X, y, s, 3, "cls", epochs)
        Ph = oof_kernel_on_base(F0, P0, y, s, "cls")
        for n, P in (("linear", Pl), ("network", P0), ("network+kernel", Ph)):
            acc[n].append(direction_table(P.argmax(1), y, mag))
    return {n: {b: round(float(np.mean([a[b] for a in lst])), 1) for b in lst[0]} for n, lst in acc.items()}


def run_magnitude(Xrep, rep_dir, rep_mag, seeds, epochs):
    sel = (rep_dir != "D") & (rep_mag > 0); idx = np.where(sel)[0]
    y = np.log10(rep_mag[idx]); X = Xrep[idx]; mag = rep_mag[idx]; appr = mag >= APPRECIABLE
    preds = {n: np.zeros((len(seeds), len(idx))) for n in ("linear", "network", "network+kernel")}
    for si, s in enumerate(seeds):
        Pr = np.zeros(len(idx))
        for tr, te in group_folds(len(idx), s):
            mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9; ym, ys = y[tr].mean(), y[tr].std() + 1e-9
            Pr[te] = Ridge(alpha=1.0).fit((X[tr] - mu) / sd, (y[tr] - ym) / ys).predict((X[te] - mu) / sd) * ys + ym
        P0, F0 = oof_mlp(X, y, s, 1, "reg", epochs)
        Ph = oof_kernel_on_base(F0, P0, y, s, "reg")
        preds["linear"][si] = Pr; preds["network"][si] = P0; preds["network+kernel"][si] = Ph
    out = {}
    for n, P in preds.items():
        pavg = P.mean(0)
        out[n] = {"appreciable": magnitude_metrics(pavg, y, mag, appr),
                  "overall": magnitude_metrics(pavg, y, mag, np.ones(len(idx), bool))}
    return out


def positive_control(seeds, epochs):
    """Concentric-ring classes: label is the radius band in a rotated 2-D subspace, which no
    linear rule separates. Confirms the network captures nonlinearity and the kernel recovers a
    crippled network's residual."""
    rng = np.random.default_rng(0); N, d = 2700, 6
    X = rng.standard_normal((N, d))
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    q1, q2 = np.quantile(r, [1 / 3, 2 / 3])
    y = np.where(r < q1, 0, np.where(r < q2, 1, 2)).astype(int)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d))); X = X @ Q
    out = {n: [] for n in ("linear", "network", "network+kernel", "weak_network", "weak_network+kernel")}
    for s in seeds:
        Pl = oof_linear_cls(X, y, s)
        P0, F0 = oof_mlp(X, y, s, 3, "cls", epochs)
        Ph = oof_kernel_on_base(F0, P0, y, s, "cls")
        Pw, Fw = oof_mlp(X, y, s, 3, "cls", 6)
        Pwh = oof_kernel_on_base(Fw, Pw, y, s, "cls")
        for n, P in (("linear", Pl), ("network", P0), ("network+kernel", Ph),
                     ("weak_network", Pw), ("weak_network+kernel", Pwh)):
            out[n].append(round(100 * float((P.argmax(1) == y).mean()), 1))
    return {n: round(float(np.mean(v)), 1) for n, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="regen_full")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--curve_depths", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global DEV
    DEV = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    seeds = tuple(int(x) for x in args.seeds.split(","))
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    t0 = time.time()
    X, d, g, m = dnnaic.load_dataset(args.dataset)
    curves, uniq, inv = replicate_curves(X, g)
    first = np.unique(g, return_index=True)[1]
    rep_dir, rep_mag = d[first], m[first]
    Xrep = curve_features(curves, args.curve_depths)
    print(f"[{time.time()-t0:.0f}s] {args.dataset}: {len(uniq)} replicates, curve feature dim {Xrep.shape[1]}, device {DEV}")

    res = {"dataset": args.dataset, "seeds": list(seeds), "curve_dim": int(Xrep.shape[1])}
    res["direction"] = run_direction(Xrep, rep_dir, rep_mag, seeds, args.epochs)
    print(f"[{time.time()-t0:.0f}s] direction done")
    res["magnitude"] = run_magnitude(Xrep, rep_dir, rep_mag, seeds, args.epochs)
    print(f"[{time.time()-t0:.0f}s] magnitude done")
    res["positive_control_rings"] = positive_control(seeds, args.epochs)
    print(f"[{time.time()-t0:.0f}s] positive control done")

    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
