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

import time
import numpy as np
from scipy.optimize import minimize


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
