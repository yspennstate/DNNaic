#!/usr/bin/env python3
"""Schematic and exploratory figures for the manuscript.

Generates the demography, gene-flow-scenario, migration-rate, and PCA figures into
paper/figures/. The per-rate accuracy, confusion, and calibration figures are produced by the
evaluation scripts; these here are the data-independent schematics plus the two exploratory PCA
panels.

Usage:  DNNAIC_DATA=/path/to/simulation_data  python scripts/make_figures.py
"""
import os, argparse, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.8})
NODE_FC = "#eef2f7"; LINE = "#3a4a5a"
EXP_FC = "#dfe9f3"; EXP_EC = "#2b3a4a"; OUT_FC = "#f0ece2"; OUT_EC = "#8a7f6a"
RED = "#b2182b"; BLUE = "#2166ac"; BAND = "#fde2b8"


def fig_demography(out):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    X = {"P1": 0.0, "P2": 1.0, "P3": 2.1, "P4": 3.6}
    T12, T123, T1234 = 500, 1000, 2000
    xm12 = (X["P1"] + X["P2"]) / 2; xm123 = (xm12 + X["P3"]) / 2; xm1234 = (xm123 + X["P4"]) / 2
    v = lambda x, y0, y1: ax.plot([x, x], [y0, y1], color=LINE, lw=1.6, solid_capstyle="round", zorder=1)
    h = lambda x0, x1, y: ax.plot([x0, x1], [y, y], color=LINE, lw=1.6, solid_capstyle="round", zorder=1)
    v(X["P1"], 0, T12); v(X["P2"], 0, T12); h(X["P1"], X["P2"], T12)
    v(xm12, T12, T123); v(X["P3"], 0, T123); h(xm12, X["P3"], T123)
    v(xm123, T123, T1234); v(X["P4"], 0, T1234); h(xm123, X["P4"], T1234); v(xm1234, T1234, T1234 + 320)
    for p in ("P1", "P2", "P3", "P4"):
        fc, ec = (OUT_FC, OUT_EC) if p == "P4" else (EXP_FC, EXP_EC)
        ax.scatter([X[p]], [0], s=1050, c=fc, edgecolors=ec, linewidths=1.4, zorder=3, clip_on=False)
        ax.text(X[p], 0, p, ha="center", va="center", fontsize=10.5, zorder=4, color=ec, fontweight="bold")
    for y in (T12, T123, T1234):
        ax.scatter([{T12: xm12, T123: xm123, T1234: xm1234}[y]], [y], s=26, c=LINE, zorder=3)
    ax.annotate("", xy=(X["P3"] + 0.30, -340), xytext=(X["P1"] - 0.30, -340),
                arrowprops=dict(arrowstyle="-", color="#66788a", lw=1.1), annotation_clip=False)
    ax.text((X["P1"] + X["P3"]) / 2, -470, "experimental populations\n(outgroup-free input to DNNaic)",
            ha="center", va="top", fontsize=8.6, color="#4a5a6a")
    ax.text(X["P4"], -340, "outgroup\n(Patterson's $D$ only)", ha="center", va="top",
            fontsize=8.4, color=OUT_EC, style="italic")
    ax.set_ylim(-760, T1234 + 430); ax.set_xlim(-0.7, 4.15)
    ax.set_yticks([0, T12, T123, T1234]); ax.set_yticklabels(["0", "500", "1,000", "2,000"])
    ax.set_ylabel("generations before present"); ax.set_xticks([])
    for s in ("top", "right", "bottom"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_bounds(0, T1234); ax.tick_params(axis="y", length=3)
    ax.set_title("Simulated four-population divergence history  ($N_e=10{,}000$)")
    fig.savefig(os.path.join(out, "fig_demography.pdf"), bbox_inches="tight"); plt.close(fig)
    print("[fig] fig_demography.pdf")


def fig_scenarios(out):
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.7))
    xs = {"P1": 0, "P2": 1, "P3": 2}
    cases = [("A", "P1", "P2", RED, "P1$\\to$P2"), ("B", "P2", "P3", RED, "P2$\\to$P3"),
             ("C", "P3", "P2", BLUE, "P3$\\to$P2"), ("D", None, None, None, "no migration")]
    for ax, (case, src, dst, col, lab) in zip(axes, cases):
        for p, x in xs.items():
            ax.scatter([x], [0], s=760, c=EXP_FC, edgecolors=EXP_EC, linewidths=1.3, zorder=3)
            ax.text(x, 0, p, ha="center", va="center", fontsize=10, fontweight="bold", color=EXP_EC, zorder=4)
        if src:
            x0, x1 = xs[src], xs[dst]; arc = -0.4 if x1 > x0 else 0.4
            ax.add_patch(FancyArrowPatch((x0, 0.16), (x1, 0.16), connectionstyle=f"arc3,rad={arc}",
                         arrowstyle="-|>", mutation_scale=15, lw=2.1, color=col, zorder=2))
            ax.text((x0 + x1) / 2, 0.62, lab, ha="center", color=col, fontsize=9.5, fontweight="bold")
        else:
            ax.text(1, 0.5, "$\\varnothing$", ha="center", va="center", color="#7a8794", fontsize=15)
            ax.text(1, 0.72, lab, ha="center", color="#7a8794", fontsize=9.5)
        ax.set_title(f"case {case}", fontsize=11)
        ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.55, 0.95); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.text(0.5, -0.06, "cases B and C share the pair $(P2,P3)$ with the flow reversed "
             "— the donor$\\to$recipient orientation a symmetric $D$ cannot make",
             ha="center", fontsize=8.7, color="#4a5a6a")
    fig.suptitle("Directional introgression classes", y=1.05, fontsize=12)
    fig.savefig(os.path.join(out, "fig_introgression_scenarios.pdf"), bbox_inches="tight"); plt.close(fig)
    print("[fig] fig_introgression_scenarios.pdf")


def fig_rate(out, data_dir):
    m = np.load(os.path.join(data_dir, "regen_full", "magnitude.npy"), allow_pickle=True).astype(float).reshape(-1)
    g = np.load(os.path.join(data_dir, "regen_full", "groups.npy"), allow_pickle=True).astype(str).reshape(-1)
    _, first = np.unique(g, return_index=True); rate = m[first]; pos = rate[rate > 0]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    lo, hi = np.log10(pos.min()) - 0.15, np.log10(pos.max()) + 0.15
    ax.hist(np.log10(pos), bins=np.linspace(lo, hi, 46), color="#5b83b0", edgecolor="white", linewidth=0.4, zorder=2)
    appr = np.log10(2.5e-4); ax.axvspan(appr, hi, color=BAND, alpha=0.55, zorder=0)
    ax.text((appr + hi) / 2, ax.get_ylim()[1] * 0.93, "appreciable\nband", ha="center", va="top", fontsize=8.8, color="#9a6a1a")
    for fr in [5e-7, 2.5e-6, 5e-5, 2.5e-4]:
        ax.axvline(np.log10(fr), color=RED, lw=1.3, ls=(0, (4, 2)), zorder=3)
    ax.text(np.log10(2.5e-4), ax.get_ylim()[1] * 0.5, "  Fisher threshold $m_\\star$\n  (direction resolves)",
            color=RED, fontsize=8.3, va="center", ha="left")
    ticks = [-7, -6, -5, -4, -3]; ax.set_xticks(ticks); ax.set_xticklabels([f"$10^{{{t}}}$" for t in ticks])
    ax.set_xlabel("per-generation migration rate"); ax.set_ylabel("simulation replicates")
    ax.set_title("Sampled migration rates (4 fixed $+$ exponential continuum)")
    ax.legend(handles=[Line2D([0], [0], color=RED, ls=(0, (4, 2)), lw=1.3, label="four fixed rates"),
                       Rectangle((0, 0), 1, 1, fc=BAND, alpha=0.7, label="appreciable ($\\geq 2.5\\times10^{-4}$)")],
              fontsize=8.4, frameon=False, loc="upper left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(out, "fig_migration_rate_distribution.pdf"), bbox_inches="tight"); plt.close(fig)
    print("[fig] fig_migration_rate_distribution.pdf")


def fig_pca(out, data_dir, max_rows=120000):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    d = os.path.join(data_dir, "regen_full")
    X = np.load(os.path.join(d, "X.npy")).astype(float)
    cls = np.load(os.path.join(d, "direction.npy"), allow_pickle=True).astype(str).reshape(-1)
    rate = np.load(os.path.join(d, "magnitude.npy"), allow_pickle=True).astype(float).reshape(-1)
    if len(X) > max_rows:
        rng = np.random.default_rng(42); idx = rng.choice(len(X), max_rows, replace=False)
        X, cls, rate = X[idx], cls[idx], rate[idx]
    z = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in sorted(set(cls)):
        m = cls == c; ax.scatter(z[m, 0], z[m, 1], s=5, alpha=0.35, label=c)
    ax.legend(title="class", markerscale=2); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA of rarefaction features, by class")
    fig.savefig(os.path.join(out, "fig_pca_by_class.pdf"), bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    lr = np.log10(np.maximum(rate, 1e-12))
    sc = ax.scatter(z[:, 0], z[:, 1], s=5, alpha=0.35, c=lr, cmap="viridis")
    fig.colorbar(sc, label="log10 migration rate"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA of rarefaction features, by migration rate")
    fig.savefig(os.path.join(out, "fig_pca_by_rate.pdf"), bbox_inches="tight"); plt.close(fig)
    print("[fig] fig_pca_by_class.pdf, fig_pca_by_rate.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--data-dir", default=os.environ.get("DNNAIC_DATA", "data/simulation_data"))
    ap.add_argument("--skip-pca", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_demography(args.out); fig_scenarios(args.out)
    try:
        fig_rate(args.out, args.data_dir)
        if not args.skip_pca:
            fig_pca(args.out, args.data_dir)
    except FileNotFoundError:
        print("[skip] data-driven figures (set DNNAIC_DATA to the simulation arrays)")


if __name__ == "__main__":
    main()
