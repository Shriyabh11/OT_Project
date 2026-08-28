import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smpm import Problem, UniformPolicy, AdaptiveCostAwarePolicy, SMPMRunner

rng = np.random.default_rng(42)
n, d = 60, 60
Lmax = 100.0
n_high = int(0.2 * n)   # 20% of components are "hard" (near Lmax)

A = np.zeros((n, d))
A[:n_high] = rng.uniform(0, Lmax, size=(n_high, d))
A[n_high:] = rng.uniform(0, 0.05 * Lmax, size=(n - n_high, d))  # alpha = 0.05 disparity
perm = rng.permutation(n)
A = A[perm]

b = rng.uniform(-1, 1, size=(n, d))
B_diag = rng.uniform(0.1, 10.0, size=d)  # gives mu_f > 0

problem = Problem(A, b, B_diag)
x_star = problem.x_star()

# Cost model: cost correlated with L_hi (harder components cost more to evaluate)
L_h = problem.L_h
cost = 0.2 + 0.8 * (L_h / L_h.max())

gamma = 1.0 / problem.Lf
T = 4000
eval_every = 10

# ---- Run uniform baseline ----
uniform_runner = SMPMRunner(problem, UniformPolicy(n), gamma, cost=cost, seed=1)
it_u, err_u, cc_u = uniform_runner.run(T, eval_every, x_star)

# ---- Run adaptive cost-aware policy ----
adaptive_policy = AdaptiveCostAwarePolicy(n, cost, ema_beta=0.2, epsilon=0.05)
adaptive_runner = SMPMRunner(problem, adaptive_policy, gamma, cost=cost, seed=1)
it_a, err_a, cc_a = adaptive_runner.run(T, eval_every, x_star)

# ---- Plot: iteration-domain (left) and cumulative-cost-domain (right) ----
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

ax = axes[0]
ax.semilogy(it_u, err_u, label="Uniform SMPM", color="#5B7FDE", linewidth=2)
ax.semilogy(it_a, err_a, label="Adaptive Cost-Aware SMPM", color="#E0633D", linewidth=2)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel(r"$\|x^t - x^\star\|^2$", fontsize=12)
ax.set_title("(a) Error vs. Iteration", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.25)

ax = axes[1]
ax.semilogy(cc_u, err_u, label="Uniform SMPM", color="#5B7FDE", linewidth=2)
ax.semilogy(cc_a, err_a, label="Adaptive Cost-Aware SMPM", color="#E0633D", linewidth=2)
ax.set_xlabel("Cumulative sampling cost", fontsize=12)
ax.set_ylabel(r"$\|x^t - x^\star\|^2$", fontsize=12)
ax.set_title("(b) Error vs. Cumulative Cost", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.25)

fig.suptitle("Objective 1: Uniform vs. Adaptive Cost-Aware SMPM", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("convergence_plot.png", dpi=200, bbox_inches="tight")

# ---- Summary numbers ----
print(f"n={n}, d={d}, T={T}")
print(f"Final error (uniform):  {err_u[-1]:.3e}")
print(f"Final error (adaptive): {err_a[-1]:.3e}")
print(f"Total cost incurred (uniform):  {cc_u[-1]:.1f}")
print(f"Total cost incurred (adaptive): {cc_a[-1]:.1f}")

threshold = 1e-6
def first_below(x_axis, errs, thr):
    idx = np.argmax(errs < thr) if np.any(errs < thr) else None
    return x_axis[idx] if idx is not None else None

t_u_iter = first_below(it_u, err_u, threshold)
t_a_iter = first_below(it_a, err_a, threshold)
c_u_cost = first_below(cc_u, err_u, threshold)
c_a_cost = first_below(cc_a, err_a, threshold)
print(f"Iterations to reach error < {threshold}: uniform={t_u_iter}, adaptive={t_a_iter}")
print(f"Cumulative cost to reach error < {threshold}: uniform={c_u_cost:.1f}, adaptive={c_a_cost:.1f}")
if c_u_cost and c_a_cost:
    print(f"Cost-efficiency gain at threshold: {c_u_cost / c_a_cost:.2f}x")