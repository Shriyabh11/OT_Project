"""Shared helpers for the Objective 2 / 3 experiments: stepsize tuning,
multi-seed evaluation and cost-to-accuracy summaries."""

import numpy as np
from ml_problem import load_dataset, make_shards, ShardedElasticNetLogReg
from smpm_general import SMPMGeneral

TOL = 1e-8  # accuracy target on ||xhat - x*||^2


def build(name, n, seed=0, spread=1.2, lam=0.01, mu=0.01):
    X, y = load_dataset(name)
    rng = np.random.default_rng(seed)
    P = ShardedElasticNetLogReg(X, y, make_shards(len(y), n, rng, spread=spread), lam=lam, mu=mu)
    xs, resid = P.full_x_star()
    assert resid / mu < 1e-7, resid
    return P, xs, P.F(xs)


def gamma0(P):
    """Theory-scale stepsize of Cor. C.3 (s = 1)."""
    return float(np.sqrt(1.0 / (P.n * P.L_h.max() * P.mu)))


def cost_to_tol(rec, tol=TOL, key="flops"):
    ix = np.where(rec["err"] < tol)[0]
    return float(rec[key][ix[0]]) if len(ix) else np.inf


def iters_to_tol(rec, tol=TOL):
    ix = np.where(rec["err"] < tol)[0]
    return float(rec["it"][ix[0]]) if len(ix) else np.inf


def run_one(P, xs, Fs, make_policy, gamma, T, seed, **kw):
    kw.setdefault('stop_tol', 1e-11)
    return SMPMGeneral(P, make_policy(), gamma, seed=seed).run(T, xs, Fs, **kw)


def tune_gamma(P, xs, Fs, make_policy, grid, T, seeds):
    best, best_c = None, np.inf
    for g in grid:
        cs = [cost_to_tol(run_one(P, xs, Fs, make_policy, g, T, s, stop_tol=TOL)) for s in seeds]
        c = np.median(cs)
        if c < best_c:
            best, best_c = g, c
    return best if best is not None else grid[0]


def evaluate(P, xs, Fs, policy_makers, grid_mult, T, tune_seeds, eval_seeds, **kw):
    """policy_makers: dict name -> zero-arg factory. Returns dict name -> dict(gamma, runs)."""
    g0 = gamma0(P)
    out = {}
    for name, mk in policy_makers.items():
        g = g0 * tune_gamma(P, xs, Fs, mk, [g0 * m for m in grid_mult], T, tune_seeds) / g0
        runs = [run_one(P, xs, Fs, mk, g, T, s, **kw) for s in eval_seeds]
        out[name] = dict(gamma=g, gamma_mult=g / g0, runs=runs)
    return out


def summarize(res, base="Uniform"):
    rows = {}
    for name, d in res.items():
        fl = np.array([cost_to_tol(r) for r in d["runs"]])
        tm = np.array([cost_to_tol(r, key="time") for r in d["runs"]])
        it = np.array([iters_to_tol(r) for r in d["runs"]])
        rows[name] = dict(
            gamma_mult=d["gamma_mult"],
            flops_med=float(np.median(fl)), flops_q25=float(np.percentile(fl, 25)), flops_q75=float(np.percentile(fl, 75)),
            time_med=float(np.median(tm)), iters_med=float(np.median(it)),
            reached=int(np.isfinite(fl).sum()), n_runs=len(fl),
        )
    b = rows[base]
    for name, r in rows.items():
        r["speedup_flops"] = b["flops_med"] / r["flops_med"]
        r["speedup_time"] = b["time_med"] / r["time_med"]
        r["speedup_iters"] = b["iters_med"] / r["iters_med"]
        r["speedup_flops_ci"] = boot_speedup(res, name, base)[1:]
    return rows


def boot_speedup(res, name, base="Uniform", B=2000, seed=0):
    """Ratio of median costs base/name with a 95% bootstrap CI over seeds."""
    rng = np.random.default_rng(seed)
    a = np.array([cost_to_tol(r) for r in res[base]["runs"]])
    b = np.array([cost_to_tol(r) for r in res[name]["runs"]])
    est = np.median(a) / np.median(b)
    bs = []
    for _ in range(B):
        ia, ib = rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b))
        bs.append(np.median(a[ia]) / np.median(b[ib]))
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_speedup(res, name, base="Uniform"):
    """Per-seed paired ratio base/name of flops-to-tol (same seeds -> same RNG stream start)."""
    a = np.array([cost_to_tol(r) for r in res[base]["runs"]])
    b = np.array([cost_to_tol(r) for r in res[name]["runs"]])
    ratio = a / b
    return float(np.median(ratio)), float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))
