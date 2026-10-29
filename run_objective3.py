"""
Objective 3: how does each component's contribution evolve across optimization phases,
and does adapting the sampling frequency to it help?

Part A  tracking: log the true dual errors e_i(t) = ||u_i^t - grad h_i(x*)|| (the components of
        SMPM's Lyapunov function) and the sampling probabilities p_i(t); split each run into
        three phases by the error level and quantify how the importance ranking shifts.
Part B  ablation of the phase-aware policy: confidence-weighted prior, decaying exploration.

Usage: python3 run_objective3.py [--quick]      Outputs: results/obj3.json, fig/obj3_*.pdf
"""

import json, sys, os
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_utils import build, gamma0, evaluate, summarize, paired_speedup, run_one, tune_gamma, cost_to_tol
from smpm_general import Uniform, StaticImportance, AdaptiveV2, PhaseAware

QUICK = "--quick" in sys.argv
os.makedirs("results", exist_ok=True)
os.makedirs("fig", exist_ok=True)
GRID = [0.25, 0.5, 1, 2, 4, 8, 16] if not QUICK else [1, 4]
T_OF = {'breast_cancer': 3500, 'digits_even_odd': 6000}
T = 3500 if not QUICK else 800
MU = 1e-3
TUNE_SEEDS = [100, 101, 102] if not QUICK else [100]
EVAL_SEEDS = list(range(10)) if not QUICK else [0, 1]
TRACK_SEEDS = list(range(6)) if not QUICK else [0, 1]
DATASETS = [("breast_cancer", 20), ("digits_even_odd", 40)]
PHASES = [("I", 1e-2, np.inf), ("II", 1e-6, 1e-2), ("III", 0.0, 1e-6)]   # err ranges (lo, hi]

summary = {}
fig_h, axs_h = plt.subplots(len(DATASETS), 2, figsize=(7.2, 5.2))
for row, (name, n) in enumerate(DATASETS):
    P, xs, Fs = build(name, n, mu=MU)
    T = T_OF[name] if not QUICK else 800
    g0 = gamma0(P)
    imp = StaticImportance(P.L_h, P.mu, n).p
    order = np.argsort(P.m)                      # components sorted by shard size
    # ---------------- Part A: tracking ----------------
    mk_phase = lambda: PhaseAware(n, imp)
    gam = g0 * tune_gamma(P, xs, Fs, mk_phase, [g0 * m for m in GRID], T, TUNE_SEEDS) / g0
    share_phase = {ph[0]: [] for ph in PHASES}
    top3_mass = {ph[0]: [] for ph in PHASES}
    freq_phase = {ph[0]: [] for ph in PHASES}
    for s in TRACK_SEEDS:
        r = run_one(P, xs, Fs, mk_phase, gam, T, s, track_dual_every=5, log_p_every=5)
        err_snap = np.interp(np.arange(0, T, 5), r["it"], r["err"])       # error at the snapshot times
        D = r["dual"] ** 2
        D = D / D.sum(1, keepdims=True)                                  # share of Lyapunov dual mass
        k = min(len(D), len(err_snap))
        for ph, lo, hi in PHASES:
            sel = (err_snap[:k] > lo) & (err_snap[:k] <= hi)
            if sel.sum() >= 3:
                share_phase[ph].append(D[:k][sel].mean(0))
                top3_mass[ph].append(np.sort(D[:k][sel].mean(0))[-3:].sum())
                tt = np.arange(0, T, 5)[:k][sel]
                lo_t, hi_t = tt.min(), tt.max() + 5
                freq_phase[ph].append(np.bincount(r["samples"][lo_t:hi_t], minlength=n) / (hi_t - lo_t))
        if s == TRACK_SEEDS[0]:
            Dm, Pm, Tshow = D, r["p_log"], int(min(T, (np.where(r["err"] < 1e-10)[0][:1].tolist() or [len(r["it"]) - 1])[0] * 20 * 1.15 + 50))
            ax = axs_h[row, 0]
            im = ax.imshow(np.log10(Dm[: Tshow // 5, order].T + 1e-6), aspect="auto", origin="lower",
                           extent=[0, Tshow, 0, n], cmap="viridis", vmin=-4, vmax=0)
            ax.set_title(f"{name.replace('_',' ')}: dual-error share $e_i^2/\\sum e_j^2$", fontsize=7)
            ax.set_ylabel("component (sorted by $m_i$)", fontsize=7)
            ax.tick_params(labelsize=6)
            plt.colorbar(im, ax=ax, pad=0.01).ax.tick_params(labelsize=6)
            ax = axs_h[row, 1]
            im = ax.imshow(Pm[: Tshow // 5, order].T * n, aspect="auto", origin="lower",
                           extent=[0, Tshow, 0, n], cmap="magma", vmin=0, vmax=3)
            ax.set_title("phase-aware sampling prob. $n\\,p_i(t)$ (1 = uniform)", fontsize=7)
            ax.tick_params(labelsize=6)
            plt.colorbar(im, ax=ax, pad=0.01).ax.tick_params(labelsize=6)
            for a in axs_h[row]:
                a.set_xlabel("iteration $t$", fontsize=7)
    A = {}
    mean_share = {ph: np.mean(v, 0) for ph, v in share_phase.items() if len(v)}
    A["phases_observed"] = {ph: len(v) for ph, v in share_phase.items()}
    phs = list(mean_share)
    for a_i in range(len(phs)):
        for b_i in range(a_i + 1, len(phs)):
            A[f"spearman_{phs[a_i]}_{phs[b_i]}"] = float(spearmanr(mean_share[phs[a_i]], mean_share[phs[b_i]])[0])
    for ph in phs:
        A[f"spearman_share_vs_L_{ph}"] = float(spearmanr(mean_share[ph], P.L_h)[0])
        A[f"top3_mass_{ph}"] = float(np.mean(top3_mass[ph]))
        f = np.mean(freq_phase[ph], 0)
        A[f"sampling_corr_with_L_{ph}"] = float(spearmanr(f, P.L_h)[0])
        A[f"sampling_share_of_top_quartile_by_L_{ph}"] = float(f[P.L_h >= np.percentile(P.L_h, 75)].sum())
    A["uniform_share_of_top_quartile_by_L"] = float((P.L_h >= np.percentile(P.L_h, 75)).mean())
    print(f"[{name}] Part A:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in A.items()}), flush=True)

    # ---------------- Part B: ablation ----------------
    variants = {
        "Uniform": lambda: Uniform(n),
        "Static importance": lambda: StaticImportance(P.L_h, P.mu, n),
        "Adaptive v2": lambda: AdaptiveV2(n, kappa=1.0),
        "Phase-aware (full)": lambda: PhaseAware(n, imp),
        "  - no prior shrinkage": lambda: PhaseAware(n, imp, k=0.0),
        "  - no $\\varepsilon$ decay (0.05)": lambda: PhaseAware(n, imp, eps0=0.05, eps_min=0.05),
        "  - const. $\\varepsilon$=0.30": lambda: PhaseAware(n, imp, eps0=0.30, eps_min=0.30),
        "  - $\\kappa$=0 (benefit only)": lambda: PhaseAware(n, imp, kappa=0.0),
    }
    res = evaluate(P, xs, Fs, variants, GRID, T, TUNE_SEEDS, EVAL_SEEDS)
    rows = summarize(res)
    for k, r in rows.items():
        if k != "Uniform":
            r["paired_speedup_flops_med_q25_q75"] = paired_speedup(res, k)
        print(f"  {k:34s} gamma x{r['gamma_mult']:>3.0f} speedup(flops) {r['speedup_flops']:.2f}x "
              f"speedup(iters) {r['speedup_iters']:.2f}x speedup(time) {r['speedup_time']:.2f}x reached {r['reached']}/{r['n_runs']}", flush=True)
    summary[name] = dict(partA=A, partB=rows)

fig_h.tight_layout()
fig_h.savefig("fig/obj3_heatmaps.pdf")
plt.close(fig_h)

# ---------------- ablation bar chart ----------------
fig, axes = plt.subplots(1, len(DATASETS), figsize=(7.2, 2.5), sharey=True)
for ax, (name, _) in zip(axes, DATASETS):
    rows = summary[name]["partB"]
    ks = [k for k in rows if k != "Uniform"]
    med = [rows[k]["paired_speedup_flops_med_q25_q75"][0] for k in ks]
    lo = [rows[k]["paired_speedup_flops_med_q25_q75"][0] - rows[k]["paired_speedup_flops_med_q25_q75"][1] for k in ks]
    hi = [rows[k]["paired_speedup_flops_med_q25_q75"][2] - rows[k]["paired_speedup_flops_med_q25_q75"][0] for k in ks]
    ax.barh(range(len(ks)), med, xerr=[lo, hi], color="#E0633D", alpha=0.8, error_kw=dict(lw=0.8))
    ax.axvline(1.0, color="#5B7FDE", ls="--", lw=1)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks, fontsize=6)
    ax.set_xlabel("cost speedup vs. uniform (paired median, IQR)", fontsize=7)
    ax.set_title(name.replace("_", " "), fontsize=8)
    ax.tick_params(labelsize=6)
    ax.invert_yaxis()
fig.tight_layout()
fig.savefig("fig/obj3_ablation.pdf")

json.dump(summary, open("results/obj3.json", "w"), indent=1, default=float)
print("done")
