"""
Reusable SMPM (Stochastic Multi-Proximal Method) implementation.

Base algorithm follows Condat, Gasanov & Richtarik (arXiv:2505.12409, 2025),
specialized to the singleton-sampling case (|Omega^t| = 1 at every iteration,
p_empty = 0), which is the setting used for both uniform sampling
(Corollary C.3) and importance sampling (Corollary C.4) in the paper.

On top of the base algorithm, this module adds an ADAPTIVE COST-AWARE
sampling policy: the sampling distribution over components is updated online
using a priority score = (estimated benefit) / (estimated cost), where cost
models the (possibly heterogeneous) computational expense of each component's
proximal operator. This implements the "Objective 1" extension proposed in
the project proposal (Adaptive Cost-Aware SMPM).

Kept intentionally modular so later objectives (real-data benchmarks,
evolving-importance tracking, distributed cost-awareness, ...) can reuse
`Problem`, `SMPMRunner`, and the sampling-policy interface without rewriting
the base algorithm.
"""

import numpy as np


class Problem:
    """
    Diagonal quadratic composite problem used for controlled experiments:

        min_x  f(x) + (1/n) sum_i h_i(x)

    with f(x) = 0.5 x^T B x   (diagonal B, so f is smooth & strongly convex)
    and  h_i(x) = 0.5 x^T diag(a_i) x - b_i^T x   (diagonal, closed-form prox)

    g == 0 throughout (matches the paper's Section 6.1 setup).
    """

    def __init__(self, A, b, B_diag):
        # A: (n, d) array, row i = diagonal of A_i (>= 0)
        # b: (n, d) array, row i = b_i
        # B_diag: (d,) diagonal of B (> 0, gives strong convexity of f)
        self.A = A
        self.b = b
        self.B_diag = B_diag
        self.n, self.d = A.shape
        self.L_h = A.max(axis=1)          # L_{h_i} per component (diagonal case)
        self.Lf = B_diag.max()
        self.muf = B_diag.min()

    def x_star(self):
        mean_a = self.A.mean(axis=0)
        mean_b = self.b.mean(axis=0)
        return mean_b / (self.B_diag + mean_a)

    def grad_f(self, x):
        return self.B_diag * x

    def prox_h(self, i, c, v):
        # prox_{c * h_i}(v), elementwise closed form for diagonal h_i
        return (c * self.b[i] + v) / (c * self.A[i] + 1.0)


class SamplingPolicy:
    """Base interface: returns current per-component probabilities pi (sum=1, all>0)."""

    def probs(self):
        raise NotImplementedError

    def update(self, sampled_i, step_norm):
        """Called after each iteration with the index sampled and the observed
        update magnitude ||xhat - y_i|| (the 'benefit' signal)."""
        pass


class UniformPolicy(SamplingPolicy):
    def __init__(self, n):
        self.n = n
        self._p = np.full(n, 1.0 / n)

    def probs(self):
        return self._p

    def update(self, sampled_i, step_norm):
        pass  # static


class AdaptiveCostAwarePolicy(SamplingPolicy):
    """
    Priority_i propto  benefit_i / cost_i,  benefit_i tracked via an EMA of the
    observed step norm each time component i is sampled. An epsilon-uniform
    floor keeps the distribution proper (p_i > 0 for all i, required by SMPM's
    arbitrary-sampling theory) and guarantees exploration of every component.
    """

    def __init__(self, n, cost, ema_beta=0.2, epsilon=0.05, init_benefit=1.0):
        self.n = n
        self.cost = cost
        self.ema_beta = ema_beta
        self.epsilon = epsilon
        self.benefit = np.full(n, init_benefit, dtype=float)
        self._p = np.full(n, 1.0 / n)
        self._recompute()

    def _recompute(self):
        priority = self.benefit / self.cost
        priority = priority / priority.sum()
        self._p = (1 - self.epsilon) * priority + self.epsilon * (1.0 / self.n)
        self._p = self._p / self._p.sum()  # numerical safety

    def probs(self):
        return self._p

    def update(self, sampled_i, step_norm):
        self.benefit[sampled_i] = (
            (1 - self.ema_beta) * self.benefit[sampled_i] + self.ema_beta * step_norm
        )
        self._recompute()


class SMPMRunner:
    """
    Runs singleton-sampling SMPM (|Omega^t| = 1, p_empty = 0) on a Problem,
    using eta_i = 1 / (n * p_i(t)) as in Corollary C.1/C.4 (with p_i now
    possibly time-varying under the adaptive policy -- an empirical extension
    of the paper's static-sampling theory, evaluated here experimentally).
    """

    def __init__(self, problem: Problem, policy: SamplingPolicy, gamma, cost=None, seed=0):
        self.problem = problem
        self.policy = policy
        self.gamma = gamma
        self.cost = cost  # optional (n,) array; tracked cumulatively for cost-domain plots
        self.rng = np.random.default_rng(seed)
        n, d = problem.n, problem.d
        self.x = np.zeros(d)
        self.u = np.zeros((n, d))   # dual variables u_i
        self.u_avg = np.zeros(d)
        self.cum_cost = 0.0

    def step(self):
        p = self.problem
        gamma = self.gamma
        xhat = self.x - gamma * (p.grad_f(self.x) + self.u_avg)  # g = 0

        probs = self.policy.probs()
        i = self.rng.choice(p.n, p=probs)
        eta_i = 1.0 / (p.n * probs[i])

        y_i = p.prox_h(i, gamma * eta_i, xhat + gamma * eta_i * self.u[i])
        u_i_new = self.u[i] + (1.0 / (gamma * eta_i)) * (xhat - y_i)

        step_norm = np.linalg.norm(xhat - y_i)
        self.u_avg += (u_i_new - self.u[i]) / p.n
        self.u[i] = u_i_new
        self.x = y_i  # |Omega^t| = 1 -> x^{t+1} is the single y_i

        if self.cost is not None:
            self.cum_cost += self.cost[i]

        self.policy.update(i, step_norm)

    def run(self, T, eval_every, x_star):
        errors = []
        iters = []
        cum_costs = []
        for t in range(T):
            self.step()
            if t % eval_every == 0 or t == T - 1:
                errors.append(np.sum((self.x - x_star) ** 2))
                iters.append(t)
                cum_costs.append(self.cum_cost)
        return np.array(iters), np.array(errors), np.array(cum_costs)