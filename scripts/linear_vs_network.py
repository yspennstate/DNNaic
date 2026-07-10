"""Linear versus network on the two halves of DNNaic, reproducibly.

Direction (3-way A/B/C classification) is a first-order-linear problem: a linear (logistic)
head is Bayes-optimal, and no network improves on it. Even a network with an explicit linear
skip, or one initialised exactly at the logistic solution, or a heavily regularised one,
underperforms the linear head -- at this information limit the nonlinear path has only noise
to fit, and training spends capacity on it at a cost to test accuracy.

Magnitude (log10-rate regression) is genuinely nonlinear: a network (or gradient boosting)
substantially beats a linear ridge model. A linear-skip network is a reasonable choice here,
since the nonlinear path has real signal to add.

Each replicate is one 54-D vector (mean and standard deviation across depths of the 27 non-depth
rarefaction coordinates). Leakage-free K-fold by replicate; fold-local scaling; CPU only.
Set DNNAIC_SIM_DATA to the regen_full directory. Reproduces Table (linear vs network).
"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
torch.set_num_threads(4)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import spearmanr

DATA = os.environ.get("DNNAIC_SIM_DATA") or os.path.join(
    os.environ.get("DNNAIC_DATA", "data/simulation_data"), "regen_full")


def load_54d(root):
    X = np.load(os.path.join(root, "X.npy")); d = np.load(os.path.join(root, "direction.npy"))
    g = np.load(os.path.join(root, "groups.npy")); m = np.load(os.path.join(root, "magnitude.npy"))
    uniq, first, inv = np.unique(g, return_index=True, return_inverse=True); G = len(uniq)
    order = np.lexsort((X[:, 0], inv)); per = np.bincount(inv); assert per.min() == per.max()
    tab = X[order].reshape(G, per.min(), X.shape[1]); v = tab[..., 1:]
    return np.concatenate((v.mean(1), v.std(1)), 1), d[first], m[first]


class Net(nn.Module):
    """Residual SiLU MLP for classification (d_out=3) or regression (d_out=1), optional linear skip."""
    def __init__(self, d_in, d_out, width=256, depth=3, dropout=0.0, skip=False):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if skip else None
        self.inp = nn.Linear(d_in, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(width, d_out)
    def forward(self, x):
        h = Fn.silu(self.inp(x))
        for l in self.hidden: h = h + Fn.silu(l(h))
        y = self.out(self.drop(h))
        return y + self.skip(x) if self.skip is not None else y


def train(net, Xtr, ytr, Xva, yva, task, epochs=200, lr=1e-3, wd=1e-4, seed=0, patience=40):
    torch.manual_seed(seed); np.random.seed(seed)
    xt = torch.tensor(Xtr, dtype=torch.float32); xv = torch.tensor(Xva, dtype=torch.float32)
    if task == "cls":
        yt = torch.tensor(ytr, dtype=torch.long); yv = torch.tensor(yva, dtype=torch.long)
        lossf = lambda o, y: Fn.cross_entropy(o, y)
        score = lambda: float((net(xv).argmax(1) == yv).float().mean()); better, best0 = (lambda a, b: a > b), -1.0
    else:
        yt = torch.tensor(ytr, dtype=torch.float32); yv = torch.tensor(yva, dtype=torch.float32)
        lossf = lambda o, y: Fn.mse_loss(o.reshape(-1), y)
        score = lambda: float(Fn.mse_loss(net(xv).reshape(-1), yv)); better, best0 = (lambda a, b: a < b), np.inf
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    n = len(xt); best = (best0, None); bad = 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for k in range(0, n, 256):
            idx = perm[k:k + 256]
            opt.zero_grad(set_to_none=True); lossf(net(xt[idx]), yt[idx]).backward(); opt.step()
        sch.step()
        if (ep + 1) % 5 == 0:
            net.eval()
            with torch.no_grad(): s = score()
            if better(s, best[0]): best = (s, {k2: v.detach().clone() for k2, v in net.state_dict().items()}); bad = 0
            else: bad += 1
            if bad >= patience // 5: break
    if best[1] is not None: net.load_state_dict(best[1])
    net.eval(); return net


def linear_init(net, Xtr, yi_tr, d_out=3):
    """Warm-start the linear skip from ridge one-hot least-squares; start exactly at the linear map."""
    Y = np.eye(d_out)[yi_tr]; Xb = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    A = np.linalg.solve(Xb.T @ Xb + 1e-2 * np.eye(Xb.shape[1]), Xb.T @ Y)
    with torch.no_grad():
        net.skip.weight.copy_(torch.tensor(A[:-1].T, dtype=torch.float32))
        net.skip.bias.copy_(torch.tensor(A[-1], dtype=torch.float32))
        net.out.weight.zero_(); net.out.bias.zero_()
    return net


def cls_pred(net, Z):
    with torch.no_grad():
        return net(torch.tensor(Z, dtype=torch.float32)).argmax(1).numpy()


def run_direction(F, lab, rate):
    pos = lab != "D"; Fp = F[pos]
    yi = np.array([{"A": 0, "B": 1, "C": 2}[c] for c in lab[pos]]); appr = rate[pos] >= 2.5e-4
    models = ["logistic", "mlp", "skip_mlp", "linear_init_mlp", "reg_mlp"]
    pred = {m: np.full(len(yi), -1, int) for m in models}
    t0 = time.time()
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Fp, yi)):
        itr, iva = tr[:int(0.8 * len(tr))], tr[int(0.8 * len(tr)):]
        sc = StandardScaler().fit(Fp[tr]); Z = lambda A: sc.transform(Fp[A]).astype(np.float64)
        pred["logistic"][te] = LogisticRegression(C=1.0, max_iter=3000).fit(Z(tr), yi[tr]).predict(Z(te))
        pred["mlp"][te] = cls_pred(train(Net(54, 3), Z(itr), yi[itr], Z(iva), yi[iva], "cls", wd=1e-4), Z(te))
        pred["skip_mlp"][te] = cls_pred(train(Net(54, 3, skip=True), Z(itr), yi[itr], Z(iva), yi[iva], "cls", wd=1e-4), Z(te))
        n3 = linear_init(Net(54, 3, skip=True), Z(itr), yi[itr])
        pred["linear_init_mlp"][te] = cls_pred(train(n3, Z(itr), yi[itr], Z(iva), yi[iva], "cls", wd=1e-4), Z(te))
        pred["reg_mlp"][te] = cls_pred(train(Net(54, 3, dropout=0.3), Z(itr), yi[itr], Z(iva), yi[iva], "cls", wd=1e-2, patience=60), Z(te))
        print(f"  direction fold {k} ({time.time()-t0:.0f}s)", flush=True)
    acc = lambda p, m: float((p[m] == yi[m]).mean())
    allm = np.ones(len(yi), bool)
    return {m: {"overall": acc(pred[m], allm), "appreciable": acc(pred[m], appr)} for m in models}, int(len(yi))


def run_magnitude(F, lab, rate):
    pos = (lab != "D") & (rate > 0); Fp = F[pos]; y = np.log10(rate[pos]); ratep = rate[pos]; appr = ratep >= 2.5e-4
    models = ["ridge", "mlp", "skip_mlp", "hgb"]
    predlog = {m: np.zeros(len(y)) for m in models}
    t0 = time.time()
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(Fp)):
        itr, iva = tr[:int(0.85 * len(tr))], tr[int(0.85 * len(tr)):]
        sc = StandardScaler().fit(Fp[tr]); Z = lambda A: sc.transform(Fp[A]).astype(np.float64)
        predlog["ridge"][te] = Ridge(alpha=10.0).fit(Z(tr), y[tr]).predict(Z(te))
        predlog["hgb"][te] = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, random_state=0).fit(Z(tr), y[tr]).predict(Z(te))
        ym, ys = y[itr].mean(), y[itr].std() + 1e-9
        for name, skip in (("mlp", False), ("skip_mlp", True)):
            net = train(Net(54, 1, skip=skip), Z(itr), (y[itr] - ym) / ys, Z(iva), (y[iva] - ym) / ys, "reg")
            with torch.no_grad():
                predlog[name][te] = net(torch.tensor(Z(te), dtype=torch.float32)).reshape(-1).numpy() * ys + ym
        print(f"  magnitude fold {k} ({time.time()-t0:.0f}s)", flush=True)

    def metrics(parr, mask):
        pr = 10 ** parr[mask]; tr = ratep[mask]; rel = np.abs(pr - tr) / tr
        return {"MRE": float(rel.mean()), "median": float(np.median(rel)), "spearman": float(spearmanr(parr[mask], y[mask]).statistic)}
    allm = np.ones(len(y), bool)
    return {m: {"appreciable": metrics(predlog[m], appr), "all": metrics(predlog[m], allm)} for m in models}, int(appr.sum())


def main():
    F, lab, rate = load_54d(DATA)
    out = {"direction": None, "magnitude": None}
    out["direction"], n_dir = run_direction(F, lab, rate)
    print("\n== DIRECTION (accuracy) ==")
    for m, v in out["direction"].items(): print(f"  {m:18s} overall {v['overall']:.4f}  appreciable {v['appreciable']:.4f}")
    out["magnitude"], n_appr = run_magnitude(F, lab, rate)
    print("\n== MAGNITUDE (appreciable band) ==")
    for m, v in out["magnitude"].items():
        a = v["appreciable"]; print(f"  {m:18s} MRE {a['MRE']:.3f}  median {a['median']:.3f}  Spearman {a['spearman']:.3f}")
    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(here, "..", "results", "linear_vs_network.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump({**out, "_n_direction": n_dir, "_n_appreciable": n_appr}, open(dst, "w"), indent=2)
    print("\nwrote", os.path.normpath(dst))


if __name__ == "__main__":
    main()
