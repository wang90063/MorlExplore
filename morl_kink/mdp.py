"""Tabular two-objective MDPs with exact solutions.

Preferences are parametrised as w(t) = (1 - t, t), t in [0, 1].
Under linear scalarisation V*(t) is convex piecewise-linear in t; the optimal
policy is piecewise constant and jumps at "corner weights".
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class MOMDP:
    P: np.ndarray      # [S, A, S] transition probabilities
    R: np.ndarray      # [S, A, 2] vector rewards
    gamma: float
    mu: np.ndarray     # [S] initial state distribution

    @property
    def S(self):
        return self.P.shape[0]

    @property
    def A(self):
        return self.P.shape[1]

    def scalar_reward(self, t):
        return (1.0 - t) * self.R[..., 0] + t * self.R[..., 1]


def random_momdp(S=30, A=4, branching=3, gamma=0.9, corr=0.0, seed=0):
    """Sparse random MDP with uniform rewards.

    `corr` in [0, 1) mixes objective 0 into objective 1; higher values mean less
    conflict and therefore fewer corner weights.
    """
    rng = np.random.default_rng(seed)
    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            nxt = rng.choice(S, size=branching, replace=False)
            P[s, a, nxt] = rng.dirichlet(np.ones(branching))
    R = rng.uniform(0.0, 1.0, size=(S, A, 2))
    R[..., 1] = corr * R[..., 0] + (1.0 - corr) * R[..., 1]
    mu = np.zeros(S)
    mu[0] = 1.0
    return MOMDP(P, R, gamma, mu)


def value_iteration(mdp, t, tol=1e-10, max_iter=100_000):
    """Optimal scalarised Q for preference t."""
    r = mdp.scalar_reward(t)
    Q = np.zeros((mdp.S, mdp.A))
    for _ in range(max_iter):
        Q_new = r + mdp.gamma * mdp.P @ Q.max(1)
        if np.abs(Q_new - Q).max() < tol:
            return Q_new
        Q = Q_new
    return Q


def policy_eval_vec(mdp, pi):
    """Vector-valued V^pi [S, 2] and Q^pi [S, A, 2] for deterministic pi [S]."""
    idx = np.arange(mdp.S)
    P_pi = mdp.P[idx, pi]                      # [S, S]
    R_pi = mdp.R[idx, pi]                      # [S, 2]
    V = np.linalg.solve(np.eye(mdp.S) - mdp.gamma * P_pi, R_pi)
    Q = mdp.R + mdp.gamma * np.einsum("sat,tk->sak", mdp.P, V)
    return V, Q


def solve(mdp, t):
    Q = value_iteration(mdp, t)
    pi = Q.argmax(1)
    V, _ = policy_eval_vec(mdp, pi)
    return pi, mdp.mu @ V


def corner_weights(mdp, eps=1e-9):
    """Exact corner weights of the convex coverage set (2 objectives).

    Recursive bisection on the upper envelope of lines (1-t) V0 + t V1, which is
    exact because V*(t) is convex piecewise-linear.
    """
    corners = []

    def scal(V, t):
        return (1 - t) * V[0] + t * V[1]

    def rec(a, Va, b, Vb, depth=0):
        if np.allclose(Va, Vb, atol=1e-9) or depth > 60:
            return
        # intersection of the two lines
        da, db = Va[1] - Va[0], Vb[1] - Vb[0]
        if abs(da - db) < 1e-14:
            return
        tc = (Vb[0] - Va[0]) / (da - db)
        if not (a < tc < b):
            return
        _, Vc = solve(mdp, tc)
        if scal(Vc, tc) <= scal(Va, tc) + eps:
            corners.append(tc)
            return
        rec(a, Va, tc, Vc, depth + 1)
        rec(tc, Vc, b, Vb, depth + 1)

    _, V0 = solve(mdp, 0.0)
    _, V1 = solve(mdp, 1.0)
    rec(0.0, V0, 1.0, V1)
    return np.array(sorted(corners))


def exact_tables(mdp, ts):
    """Q*[T, S, A], optimal greedy policy pi*[T, S], V*(mu)[T, 2] on a grid of t."""
    Qs, pis, Vs = [], [], []
    for t in ts:
        Q = value_iteration(mdp, t)
        pi = Q.argmax(1)
        V, _ = policy_eval_vec(mdp, pi)
        Qs.append(Q)
        pis.append(pi)
        Vs.append(mdp.mu @ V)
    return np.stack(Qs), np.stack(pis), np.stack(Vs)


def local_switch_points(ts, pis):
    """Per-state t values where the optimal greedy action changes (grid midpoints)."""
    switches = []
    for s in range(pis.shape[1]):
        ch = np.nonzero(pis[1:, s] != pis[:-1, s])[0]
        switches.append(0.5 * (ts[ch] + ts[ch + 1]))
    return switches


def dist_to_nearest(x, points):
    if len(points) == 0:
        return np.full_like(x, np.inf, dtype=float)
    return np.abs(x[:, None] - np.asarray(points)[None, :]).min(1)
