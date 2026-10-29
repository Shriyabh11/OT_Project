"""
Objective 2: evaluate Uniform / static importance / Adaptive v1 / Adaptive v2 / Phase-aware SMPM
on elastic-net logistic regression over real datasets, with measured prox cost.

Usage:  python3 run_objective2.py            (full run, ~10 min on one core)
        python3 run_objective2.py --quick    (smoke test)
Outputs: results/obj2.json, fig/obj2_*.pdf
"""

import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_utils import build, gamma0, evaluate, summarize, paired_speedup, cost_to_tol
from smpm_general import Uniform, StaticImportance, AdaptiveV1, AdaptiveV2, PhaseAware

QUICK = "--quick" in sys.argv
os.makedirs("results", exist_ok=True)
os.makedirs("fig", exist_ok=True)

GRID = [0.25, 0.5, 1, 2, 4, 8, 16] if not QUICK else [1, 4]
T_OF = {'breast_cancer': 3500, 'digits_even_odd': 6000}
T = 3500 if not QUICK else 800
MU = 1e-3
TUNE_SEEDS = [100, 101, 102] if not QUICK else [100]
EVAL_SEEDS = list(range(10)) if not QUICK else [0, 1]

DATASETS = [("breast_cancer", 20), ("digits_even_odd", 40)]
COL = {"Uniform": "#5B7FDE", "Static importance": "#7A7A7A", "Adaptive v1 (Obj.1)": "#B58CD9",
       "Adaptive v2 (ours)": "#E0633D", "Phase-aware (ours)": "#2E9E6B"}


def makers(P):
    cost_model = 0.2 + 0.8 * P.L_h / P.L_h.max()          # Obj.-1 style *modeled* cost
    imp = StaticImportance(P.L_h, P.mu, P.n).p
    return {
        "Uniform": lambda: Uniform(P.n),
        "Static importance": lambda: StaticImportance(P.L_h, P.mu, P.n),
        "Adaptive v1 (Obj.1)": lambda: AdaptiveV1(P.n, cost_model),
        "Adaptive v2 (ours)": lambda: AdaptiveV2(P.n, kappa=1.0),
        "Phase-aware (ours)": lambda: PhaseAware(P.n, imp),
    }


def geo_band(runs, key, xkey):
    """median curve (+ IQR) of log error against a common x grid."""
    xs = np.linspace(0, min(r[xkey][-1] for r in runs), 200)
    Y = np.array([np.interp(xs, r[xkey], np.log10(r[key])) for r in runs])
    return xs, np.median(Y, 0), np.percentile(Y, 25, 0), np.percentile(Y, 75, 0)


summary = {}
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
for row, (name, n) in enumerate(DATASETS):
    P, xs, Fs = build(name, n, mu=MU)
    T = T_OF[name] if not QUICK else 800
    print(f"[{name}] N={P.N} d={P.d} n={n} shard sizes {P.m.min()}..{P.m.max()}  nnz(x*)={(xs!=0).sum()}", flush=True)
    res = evaluate(P, xs, Fs, makers(P), GRID, T, TUNE_SEEDS, EVAL_SEEDS)
    rows = summarize(res)
    for k, r in rows.items():
        if k != "Uniform":
            r["paired_speedup_flops_med_q25_q75"] = paired_speedup(res, k)
    summary[name] = dict(N=int(P.N), d=int(P.d), n=int(n), m_min=int(P.m.min()), m_max=int(P.m.max()),
                         nnz_star=int((xs != 0).sum()), rows=rows)
    for k, r in rows.items():
        print(f"  {k:22s} gamma x{r['gamma_mult']:>4.0f}  Mflop-to-1e-8 {r['flops_med']/1e3:8.1f}  "
              f"iters {r['iters_med']:6.0f}  time {r['time_med']*1e3:7.0f} ms  "
              f"speedup(flops) {r['speedup_flops']:.2f}x  speedup(time) {r['speedup_time']:.2f}x  "
              f"reached {r['reached']}/{r['n_runs']}", flush=True)
    for col, xkey, lab in [(0, "it", "Iteration $t$"), (1, "flops", "Cumulative prox cost (Mflop)")]:
        ax = axes[row, col]
        for k, d in res.items():
            x, m, lo, hi = geo_band(d["runs"], "err", xkey)
            sc = 1e3 if xkey == "flops" else 1
            ax.plot(x / sc, m, color=COL[k], lw=1.4, label=k)
            ax.fill_between(x / sc, lo, hi, color=COL[k], alpha=0.12, lw=0)
        ax.set_xlabel(lab, fontsize=8)
        ax.set_ylabel(r"$\log_{10}\|\hat x^t-x^\star\|^2$", fontsize=8)
        ax.set_title(f"{name.replace('_', ' ')} ($N$={P.N}, $d$={P.d}, $n$={n})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.set_ylim(-12, 1)
        if col == 1:
            ax.set_xlim(0, np.median([r["flops"][-1] for r in res["Uniform"]["runs"]]) / 1e3 * 0.55)
axes[0, 0].legend(fontsize=6, loc="upper right")
fig.tight_layout()
fig.savefig("fig/obj2_convergence.pdf")
plt.close(fig)

# ---------------- ablation on breast_cancer: kappa / epsilon / beta ----------------
T = 3500 if not QUICK else 800
P, xs, Fs = build("breast_cancer", 20, mu=MU)
abl = {}
variants = {
    "benefit only ($\\kappa$=0)": lambda: AdaptiveV2(P.n, kappa=0.0),
    "$\\kappa$=0.5": lambda: AdaptiveV2(P.n, kappa=0.5),
    "$\\kappa$=1 (default)": lambda: AdaptiveV2(P.n, kappa=1.0),
    "$\\varepsilon$=0.02": lambda: AdaptiveV2(P.n, kappa=1.0, eps=0.02),
    "$\\varepsilon$=0.30": lambda: AdaptiveV2(P.n, kappa=1.0, eps=0.30),
    "$\\beta$=0.1": lambda: AdaptiveV2(P.n, kappa=1.0, beta=0.1),
    "$\\beta$=0.6": lambda: AdaptiveV2(P.n, kappa=1.0, beta=0.6),
}
allm = {"Uniform": lambda: Uniform(P.n), **variants}
res = evaluate(P, xs, Fs, allm, GRID, T, TUNE_SEEDS, EVAL_SEEDS)
rows = summarize(res)
for k, r in rows.items():
    if k != "Uniform":
        r["paired_speedup_flops_med_q25_q75"] = paired_speedup(res, k)
    print(f"  [ablation] {k:32s} speedup(flops) {r['speedup_flops']:.2f}x", flush=True)
summary["ablation_breast_cancer"] = rows

# ---------------- when does cost-awareness pay? shard-size heterogeneity sweep ----------------
sweep = {}
SP = [0.2, 0.7, 1.2, 1.7] if not QUICK else [0.2, 1.4]
for sp in SP:
    P, xs, Fs = build("breast_cancer", 20, spread=sp, mu=MU)
    mk = makers(P)
    sub = {k: mk[k] for k in ["Uniform", "Static importance", "Adaptive v2 (ours)", "Phase-aware (ours)"]}
    res = evaluate(P, xs, Fs, sub, GRID, T, TUNE_SEEDS, EVAL_SEEDS[:6])
    rows = summarize(res)
    sweep[str(sp)] = dict(size_ratio=float(P.m.max() / P.m.min()), rows=rows)
    print(f"  [sweep spread={sp}] size ratio {P.m.max()/P.m.min():.1f}  " +
          "  ".join(f"{k.split(' ')[0]}:{r['speedup_flops']:.2f}x" for k, r in rows.items()), flush=True)
summary["sweep_spread_breast_cancer"] = sweep

fig, ax = plt.subplots(figsize=(3.4, 2.4))
for k in ["Static importance", "Adaptive v2 (ours)", "Phase-aware (ours)"]:
    ax.plot([sweep[s]["size_ratio"] for s in sweep], [sweep[s]["rows"][k]["speedup_flops"] for s in sweep],
            "o-", ms=3, lw=1.2, color=COL[k], label=k)
ax.axhline(1.0, color=COL["Uniform"], ls="--", lw=1, label="Uniform")
ax.set_xscale("log")
ax.set_xlabel("shard-size ratio $m_{\\max}/m_{\\min}$", fontsize=8)
ax.set_ylabel("cost speedup vs. uniform", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(alpha=0.25)
ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig("fig/obj2_sweep.pdf")

json.dump(summary, open("results/obj2.json", "w"), indent=1, default=float)
print("done")
