"""Objective 3 (evolving component importance), standalone single file.
Merged from ml_problem.py, smpm_general.py, experiment_utils.py, run_objective3.py.
Needs: numpy scipy scikit-learn matplotlib.
Run:   python obj3_standalone.py          (full)
       python obj3_standalone.py --quick  (smoke test)
Outputs go to results/ and fig/ next to this file.
"""
import json, sys, os, time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===== ml_problem.py =====
"""
Objective 2: practical ML problem for SMPM -- elastic-net logistic regression
on real datasets, with heterogeneous-size data shards as the components h_i.

    min_x  F(x) = f(x) + g(x) + (1/n) sum_i h_i(x)

    f(x)   = (mu/2) ||x||^2                         (smooth, strongly convex)
    g(x)   = lam ||x||_1                            (nonsmooth, prox = soft-threshold)
    h_i(x) = (n/N) sum_{j in shard i} log(1 + exp(-y_j a_j^T x))

so that (1/n) sum_i h_i = (1/N) sum_j loss_j : the usual empirical risk.

Why shards: each prox_{c h_i} is a d-dimensional strictly convex problem with no
closed form. It is solved here by damped Newton, and its cost genuinely
depends on the shard size m_i and on the number of Newton iterations. That
gives *measured*, heterogeneous prox cost instead of the synthetic cost model
used in Objective 1.
"""



# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_dataset(name):
    from sklearn.datasets import load_breast_cancer, load_digits

    if name == "breast_cancer":
        ds = load_breast_cancer()
        X, y = ds.data, np.where(ds.target == 1, 1.0, -1.0)
    elif name == "digits_even_odd":
        ds = load_digits()
        X, y = ds.data, np.where(ds.target % 2 == 0, 1.0, -1.0)
    else:
        raise ValueError(name)
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-8  # drop constant pixels
    X = (X[:, keep] - mu[keep]) / sd[keep]
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # bias column
    return X, y


def make_shards(N, n, rng, spread=1.2, min_size=4):
    """Random partition of N samples into n shards with log-normal sizes
    (federated-style heterogeneity). Returns list of index arrays."""
    w = rng.lognormal(0.0, spread, size=n)
    sizes = np.maximum(min_size, np.round(w / w.sum() * N)).astype(int)
    while sizes.sum() > N:
        sizes[np.argmax(sizes)] -= 1
    while sizes.sum() < N:
        sizes[np.argmax(sizes)] += 1
    perm = rng.permutation(N)
    out, s = [], 0
    for m in sizes:
        out.append(perm[s:s + m])
        s += m
    return out


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _log1pexp(z):  # log(1+e^z), stable
    return np.logaddexp(0.0, z)


# ----------------------------------------------------------------------------
# problem
# ----------------------------------------------------------------------------
class ShardedElasticNetLogReg:
    def __init__(self, X, y, shards, lam=0.01, mu=0.01):
        self.X, self.y = X, y
        self.N, self.d = X.shape
        self.shards = shards
        self.n = len(shards)
        self.lam, self.mu = lam, mu
        self.scale = self.n / self.N  # h_i = scale * sum of losses in shard i
        self.A = [X[idx] * y[idx][:, None] for idx in shards]  # rows y_j a_j
        self.m = np.array([len(idx) for idx in shards])
        # smoothness of each h_i:  scale * lambda_max(A_i^T A_i) / 4
        self.L_h = np.array(
            [self.scale * np.linalg.eigvalsh(Ai.T @ Ai)[-1] / 4.0 for Ai in self.A]
        )
        self.Lf, self.muf = mu, mu

    # --- pieces of the objective ---
    def grad_f(self, x):
        return self.mu * x

    def prox_g(self, v, gamma):
        t = gamma * self.lam
        return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)

    def risk(self, x):
        return np.mean(_log1pexp(-(self.X * self.y[:, None]) @ x))

    def F(self, x):
        return self.risk(x) + 0.5 * self.mu * x @ x + self.lam * np.abs(x).sum()

    def grad_h(self, i, x):
        return -self.scale * (self.A[i].T @ _sigmoid(-(self.A[i] @ x)))

    # --- prox of c*h_i by damped Newton; returns (y, cost info) ---
    def prox_h(self, i, c, v, tol=1e-13, max_iter=60):
        Ai, s = self.A[i], self.scale
        m, d = Ai.shape
        cs = c * s

        def obj(y):
            return cs * _log1pexp(-(Ai @ y)).sum() + 0.5 * np.sum((y - v) ** 2)

        y = v.copy()
        fy = obj(y)
        iters = 0
        for _ in range(max_iter):
            z = Ai @ y
            sg = _sigmoid(-z)                       # sigma(-z)
            g = -cs * (Ai.T @ sg) + (y - v)
            w = sg * (1.0 - sg)
            H = cs * (Ai.T * w) @ Ai
            H[np.diag_indices(d)] += 1.0
            step = np.linalg.solve(H, g)
            dec = g @ step
            iters += 1
            if dec < tol:
                break
            t = 1.0
            while t > 1e-10:
                fn = obj(y - t * step)
                if fn <= fy - 0.25 * t * dec:
                    break
                t *= 0.5
            y = y - t * step
            fy = fn
        # deterministic flop-count proxy of the work done (units: 1e3 flops)
        flops = iters * (2.0 * m * d * d + d ** 3 / 3.0 + 4.0 * m * d) / 1e3
        return y, flops, iters

    def full_x_star(self):
        """High-accuracy reference solution (L-BFGS-B on x = p - q split, then
        prox-gradient fixed-point residual is checked by the caller)."""
        d, lam, mu = self.d, self.lam, self.mu
        Xy = self.X * self.y[:, None]

        def fun(pq):
            p, q = pq[:d], pq[d:]
            x = p - q
            z = Xy @ x
            val = np.mean(_log1pexp(-z)) + 0.5 * mu * x @ x + lam * (p.sum() + q.sum())
            gr = -(Xy.T @ _sigmoid(-z)) / self.N + mu * x
            return val, np.concatenate([gr + lam, -gr + lam])

        r = minimize(fun, np.zeros(2 * d), jac=True, method="L-BFGS-B",
                     bounds=[(0, None)] * (2 * d),
                     options=dict(maxiter=100000, maxfun=200000, ftol=1e-30, gtol=1e-14, maxcor=50))
        x = r.x[:d] - r.x[d:]
        # polish with proximal-gradient steps
        L = np.linalg.eigvalsh(Xy.T @ Xy)[-1] / (4 * self.N) + mu
        for _ in range(20000):
            gr = -(Xy.T @ _sigmoid(-(Xy @ x))) / self.N + mu * x
            xn = self.prox_g(x - gr / L, 1.0 / L)
            if np.linalg.norm(xn - x) < 1e-16:
                x = xn
                break
            x = xn
        gr = -(Xy.T @ _sigmoid(-(Xy @ x))) / self.N + mu * x
        resid = np.linalg.norm(x - self.prox_g(x - gr / L, 1.0 / L))
        return x, resid


# ===== smpm_general.py =====
"""
General singleton-sampling SMPM runner (|Omega^t| = 1, p_empty = 0) with a
non-trivial g (prox) and expensive, cost-measured prox_h oracles, plus the
sampling policies used in Objectives 2 and 3.

Kept separate from smpm.py (Objective 1) so the original experiment is
untouched. Policy interface:  probs() -> p,  update(i, info) -> None
with info = dict(resid=||grad h_i(y_i) - u_i||, step=||xhat - y_i||, cost=measured cost).
"""



# ----------------------------------------------------------------------------
# sampling policies
# ----------------------------------------------------------------------------
class Policy:
    def probs(self):
        raise NotImplementedError

    def update(self, i, info):
        pass


class Uniform(Policy):
    def __init__(self, n):
        self.p = np.full(n, 1.0 / n)

    def probs(self):
        return self.p


class StaticImportance(Policy):
    """Paper's Corollary C.4 importance sampling: p_i ~ b_i = max(1, sqrt(L_i / (n (mu_f+mu_g))))."""

    def __init__(self, L_h, mu, n):
        b = np.maximum(1.0, np.sqrt(L_h / (n * mu)))
        self.p = b / b.sum()

    def probs(self):
        return self.p


class AdaptiveV1(Policy):
    """Objective-1 policy: benefit = EMA of raw step ||xhat - y_i||, cost = *modeled* (given)."""

    def __init__(self, n, cost_model, beta=0.2, eps=0.15):
        self.n, self.c, self.beta, self.eps = n, cost_model, beta, eps
        self.b = np.ones(n)
        self._rc()

    def _rc(self):
        pr = self.b / self.c
        pr /= pr.sum()
        self.p = (1 - self.eps) * pr + self.eps / self.n

    def probs(self):
        return self.p

    def update(self, i, info):
        self.b[i] = (1 - self.beta) * self.b[i] + self.beta * info["step"]
        self._rc()


class AdaptiveV2(Policy):
    """
    Objective-2 policy ("AC-SMPM v2").
      * benefit_i  = EMA of the dual residual  r_i = ||grad h_i(y_i) - u_i||
                     (= ||xhat - y_i|| / (gamma*eta_i); dividing out eta_i removes the
                     feedback of the sampling probability itself from the signal)
      * cost_i     = EMA of the *measured* prox cost (learned online, no cost model)
      * priority_i = benefit_i / cost_i^kappa,  eps-uniform floor keeps p_i > 0.
    kappa = 0 -> benefit only; kappa = 1 -> full benefit/cost rule.
    Optimistic initialisation (benefit = max seen so far on first visit) forces
    every component to be explored.
    """

    def __init__(self, n, kappa=1.0, beta=0.3, eps=0.10, cost_prior=None):
        self.n, self.kappa, self.beta, self.eps = n, kappa, beta, eps
        self.b = np.full(n, np.nan)
        self.c = np.full(n, np.nan) if cost_prior is None else np.array(cost_prior, float)
        self.seen = np.zeros(n, bool)
        self.p = np.full(n, 1.0 / n)

    def probs(self):
        return self.p

    def _rc(self):
        b = np.where(self.seen, self.b, np.nanmax(self.b) if self.seen.any() else 1.0)
        c = np.where(np.isnan(self.c), np.nanmean(self.c) if (~np.isnan(self.c)).any() else 1.0, self.c)
        pr = b / c ** self.kappa
        pr = pr / pr.sum()
        self.p = (1 - self.eps) * pr + self.eps / self.n

    def update(self, i, info):
        r, cost = info["resid"], info["cost"]
        if not self.seen[i]:
            self.b[i], self.seen[i] = r, True
        else:
            self.b[i] = (1 - self.beta) * self.b[i] + self.beta * r
        self.c[i] = cost if np.isnan(self.c[i]) else (1 - self.beta) * self.c[i] + self.beta * cost
        self._rc()


class PhaseAware(Policy):
    """
    Objective-3 policy ("phase-aware AC-SMPM").
    Same benefit/cost priority as AdaptiveV2, but
      (i)  the empirical priority is shrunk toward the static importance prior
           with a confidence weight w_i = n_i / (n_i + k)  (early phase: few
           observations per component -> trust the L_h-based prior; late phase:
           trust the measured residuals),
      (ii) the exploration floor decays with the global progress, eps_t = eps_min +
           (eps_0 - eps_min) * exp(-t / tau).
    """

    def __init__(self, n, prior_p, kappa=1.0, beta=0.3, eps0=0.30, eps_min=0.05, tau=500.0, k=3.0):
        self.n, self.kappa, self.beta = n, kappa, beta
        self.prior = prior_p / prior_p.sum()
        self.eps0, self.eps_min, self.tau, self.k = eps0, eps_min, tau, k
        self.b = np.zeros(n)
        self.c = np.full(n, np.nan)
        self.cnt = np.zeros(n)
        self.t = 0
        self.p = self.prior.copy()

    def probs(self):
        return self.p

    def _rc(self):
        seen = self.cnt > 0
        if not seen.any():
            self.p = self.prior.copy()
            return
        bmax = self.b[seen].max()
        c_fill = np.nanmean(self.c)
        c = np.where(np.isnan(self.c), c_fill, self.c)
        emp = np.where(seen, self.b, bmax) / c ** self.kappa
        emp /= emp.sum()
        den = self.cnt + self.k
        w = np.where(den > 0, self.cnt / np.where(den > 0, den, 1.0), 1.0)
        pr = w * emp + (1 - w) * self.prior
        pr /= pr.sum()
        eps = self.eps_min + (self.eps0 - self.eps_min) * np.exp(-self.t / self.tau)
        self.p = (1 - eps) * pr + eps / self.n

    def update(self, i, info):
        self.t += 1
        r, cost = info["resid"], info["cost"]
        self.b[i] = r if self.cnt[i] == 0 else (1 - self.beta) * self.b[i] + self.beta * r
        self.c[i] = cost if np.isnan(self.c[i]) else (1 - self.beta) * self.c[i] + self.beta * cost
        self.cnt[i] += 1
        self._rc()


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------
class SMPMGeneral:
    """x^{t+1} = y_i, xhat^t = prox_{gamma g}(x^t - gamma (grad f(x^t) + u_avg)),
    eta_i = 1/(n p_i(t)), as in Corollary C.1 (p_i possibly time-varying)."""

    def __init__(self, problem, policy, gamma, seed=0):
        self.P, self.policy, self.gamma = problem, policy, gamma
        self.rng = np.random.default_rng(seed)
        n, d = problem.n, problem.d
        self.x = np.zeros(d)
        self.u = np.zeros((n, d))
        self.u_avg = np.zeros(d)
        self.cum_flops = 0.0
        self.cum_time = 0.0

    def step(self):
        P, g = self.P, self.gamma
        xhat = P.prox_g(self.x - g * (P.grad_f(self.x) + self.u_avg), g)
        p = self.policy.probs()
        i = int(self.rng.choice(P.n, p=p))
        eta = 1.0 / (P.n * p[i])
        t0 = time.perf_counter()
        y, flops, iters = P.prox_h(i, g * eta, xhat + g * eta * self.u[i])
        self.cum_time += time.perf_counter() - t0
        u_new = self.u[i] + (xhat - y) / (g * eta)
        step = np.linalg.norm(xhat - y)
        self.u_avg += (u_new - self.u[i]) / P.n
        self.u[i] = u_new
        self.x = y
        self.cum_flops += flops
        self.policy.update(i, dict(step=step, resid=step / (g * eta), cost=flops))
        self.last_resid = step / (g * eta)
        return xhat, i

    def run(self, T, x_star, F_star, eval_every=20, log_p_every=0, track_dual_every=0, stop_tol=None):
        """track_dual_every > 0 additionally logs the true dual errors
        e_i = ||u_i - grad h_i(x*)|| (the Lyapunov components) -- needs P.grad_h."""
        rec = dict(it=[], err=[], gap=[], flops=[], time=[], nnz=[])
        counts, p_log, sample_log, resid_log, dual_log = np.zeros(self.P.n), [], [], [], []
        if track_dual_every:
            gstar = np.array([self.P.grad_h(j, x_star) for j in range(self.P.n)])
        for t in range(T):
            xhat, i = self.step()
            counts[i] += 1
            sample_log.append(i)
            resid_log.append(self.last_resid)
            if track_dual_every and t % track_dual_every == 0:
                dual_log.append(np.linalg.norm(self.u - gstar, axis=1))
            if log_p_every and t % log_p_every == 0:
                p_log.append(self.policy.probs().copy())
            if t % eval_every == 0 or t == T - 1:
                rec["it"].append(t)
                rec["err"].append(np.sum((xhat - x_star) ** 2))
                rec["gap"].append(max(self.P.F(xhat) - F_star, 1e-18))
                rec["flops"].append(self.cum_flops)
                rec["time"].append(self.cum_time)
                rec["nnz"].append(int((xhat != 0).sum()))
                e = rec["err"][-1]
                if (stop_tol is not None and e < stop_tol) or not np.isfinite(e) or e > 1e8:
                    break
        out = {k: np.array(v) for k, v in rec.items()}
        out["counts"] = counts
        out["samples"] = np.array(sample_log)
        out["p_log"] = np.array(p_log)
        out["resid"] = np.array(resid_log)
        out["dual"] = np.array(dual_log)
        return out


# ===== experiment_utils.py =====
"""Shared helpers for the Objective 2 / 3 experiments: stepsize tuning,
multi-seed evaluation and cost-to-accuracy summaries."""


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


# ===== run_objective3.py =====
"""
Objective 3: how does each component's contribution evolve across optimization phases,
and does adapting the sampling frequency to it help?

Part A  tracking: log the true dual errors e_i(t) = ||u_i^t - grad h_i(x*)|| (the components of
        SMPM's Lyapunov function) and the sampling probabilities p_i(t); split each run into
        three phases by the error level and quantify how the importance ranking shifts.
Part B  ablation of the phase-aware policy: confidence-weighted prior, decaying exploration.

Usage: python3 run_objective3.py [--quick]      Outputs: results/obj3.json, fig/obj3_*.pdf
"""

matplotlib.use("Agg")


QUICK = "--quick" in sys.argv
os.chdir(os.path.dirname(os.path.abspath(__file__)))
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
