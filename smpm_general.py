"""
General singleton-sampling SMPM runner (|Omega^t| = 1, p_empty = 0) with a
non-trivial g (prox) and expensive, cost-measured prox_h oracles, plus the
sampling policies used in Objectives 2 and 3.

Kept separate from smpm.py (Objective 1) so the original experiment is
untouched. Policy interface:  probs() -> p,  update(i, info) -> None
with info = dict(resid=||grad h_i(y_i) - u_i||, step=||xhat - y_i||, cost=measured cost).
"""

import time
import numpy as np


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
