"""A+B kill test K2: can a few pairwise comparisons pin down a kinked utility well enough for ESR?

Same random stochastic MOMDPs as K1 (Dirichlet 1.0 transitions). A simulated user with a true utility
answers Bradley-Terry comparisons between two episode returns. The learner models u as a non-negative
combination of monotone features (linear terms, single-objective steps 1[R_i >= a], joint steps
1[R_1 >= a, R_2 >= b]), so every fitted u is monotone. The posterior is approximated by a bootstrap
ensemble of L1-regularised MAP fits with randomised per-feature penalties.

After each batch of queries the learner plans the exact ESR-optimal (augmented-state) policy for the
ensemble-mean utility; we report its true normalised regret (same scale as K1).

Query strategies
  random    returns of two uniform-random-policy episodes
  thompson  two ensemble members -> their ESR-optimal policies -> one episode each (GUTS-like)
Baseline
  lin       linear weight fitted on the random queries, then linear-optimal policy (K1 lin_fit)

Usage: PYTHONPATH=esr python esr/k2.py [--n 20]
"""
import argparse
import json
import os

import numpy as np

from k1 import A, H, M, S, evaluate, linear_policy, make_mdp, sample_returns

BUDGETS = [10, 20, 50, 100]
GRID = np.stack(np.meshgrid(np.arange(M), np.arange(M), indexing="ij"), -1)


def features(R):
    R = np.asarray(R)
    r1, r2 = R[..., 0, None], R[..., 1, None]
    k = np.arange(1, M)
    lin = np.stack([R[..., 0], R[..., 1]], -1) / (M - 1)
    s1, s2 = (r1 >= k).astype(float), (r2 >= k).astype(float)
    kk = np.arange(2, M, 2)                                  # joint steps on a coarser lattice
    j = ((R[..., 0, None, None] >= kk[:, None]) & (R[..., 1, None, None] >= kk[None])).astype(float)
    return np.concatenate([lin, s1, s2, j.reshape(*R.shape[:-1], -1)], -1)


FG = features(GRID)                                          # (M, M, F)
NF = FG.shape[-1]


def fit(X, y, rng, n_ens=8, iters=400, lam=0.02):
    """X: feature differences (n, F); y: 1 if first preferred. Returns (n_ens, F) non-negative weights."""
    W = []
    for _ in range(n_ens):
        idx = rng.integers(0, len(y), len(y))
        Xb, yb = X[idx], y[idx]
        pen = lam * rng.exponential(1.0, NF)
        w = rng.exponential(0.01, NF)
        lr = 1.0
        for _ in range(iters):
            z = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
            g = Xb.T @ (yb - z) / len(yb) - pen
            w = np.maximum(w + lr * g, 0)
        W.append(w)
    return np.array(W)


def esr_policy(P, r, U):
    """Exact ESR optimum; returns (value, pol[t, s, R1, R2] -> action)."""
    V = np.broadcast_to(U, (S, M, M)).copy()
    pol = np.zeros((H, S, M, M), int)
    for t in reversed(range(H)):
        EV = np.einsum("sat,tij->saij", P, V)
        Q = np.full((S, A, M, M), -np.inf)
        for s in range(S):
            for a in range(A):
                r1, r2 = r[s, a]
                Q[s, a, : M - r1, : M - r2] = EV[s, a, r1:, r2:]
        pol[t] = Q.argmax(1)
        V = Q.max(1)
    return V[0, 0, 0], pol


def evaluate_aug(P, r, pol, U):
    D = np.zeros((S, M, M))
    D[0, 0, 0] = 1
    for t in range(H):
        N = np.zeros_like(D)
        for s in range(S):
            for a in range(A):
                m = D[s] * (pol[t, s] == a)
                if not m.any():
                    continue
                moved = np.zeros((M, M))
                moved[r[s, a, 0]:, r[s, a, 1]:] = m[: M - r[s, a, 0], : M - r[s, a, 1]]
                N += P[s, a][:, None, None] * moved[None]
        D = N
    return float((D.sum(0) * U).sum())


def rollout_aug(P, r, pol, rng):
    s, R = 0, np.zeros(2, int)
    for t in range(H):
        a = pol[t, s, R[0], R[1]]
        R += r[s, a]
        s = rng.choice(S, p=P[s, a])
    return R


def true_utilities(m1, m2):
    n = lambda R: (np.asarray(R, float)[..., 0] / m1, np.asarray(R, float)[..., 1] / m2)
    return {
        "min": lambda R: np.minimum(*n(R)),
        "thr0.6": lambda R: n(R)[0] + 1.0 * (n(R)[1] >= 0.6),
        "thr0.9": lambda R: n(R)[0] + 1.0 * (n(R)[1] >= 0.9),
    }


def run(seed, uname, strategy, temp=0.05):
    rng = np.random.default_rng(seed)
    P, r = make_mdp(rng, 1.0)
    lin = lambda w: GRID[..., 0] * w[0] + GRID[..., 1] * w[1]
    m1 = evaluate(P, r, linear_policy(P, r, np.array([1.0, 0])), lin([1, 0]))[0]
    m2 = evaluate(P, r, linear_policy(P, r, np.array([0, 1.0])), lin([0, 1]))[0]
    if min(m1, m2) <= 0:
        return None
    u = true_utilities(m1, m2)[uname]
    U = u(GRID)
    scale = np.ptp(U)
    v_esr, _ = esr_policy(P, r, U)
    v_rand = evaluate(P, r, np.full((H, S, A), 1 / A), U)[0]
    den = max(v_esr - v_rand, 1e-9)
    qrng = np.random.default_rng(1000 + seed)

    def answer(Ra, Rb):
        p = 1 / (1 + np.exp(-(u(Ra) - u(Rb)) / (temp * scale)))
        return float(qrng.random() < p)

    pairs, ys = [], []
    for _ in range(5):                                       # warm start: random pairs
        Ra, Rb = sample_returns(P, r, qrng, 2)
        pairs.append((Ra, Rb)); ys.append(answer(Ra, Rb))
    out, W = {}, None
    while len(ys) < BUDGETS[-1]:
        W = fit(np.array([features(a) - features(b) for a, b in pairs]), np.array(ys), qrng)
        batch = 5 if len(ys) < 50 else 10
        if strategy == "thompson":
            pols = {}
            for _ in range(batch):
                i, j = qrng.choice(len(W), 2, replace=False)
                for k in (i, j):
                    if k not in pols:
                        pols[k] = esr_policy(P, r, FG @ W[k])[1]
                Ra, Rb = rollout_aug(P, r, pols[i], qrng), rollout_aug(P, r, pols[j], qrng)
                pairs.append((Ra, Rb)); ys.append(answer(Ra, Rb))
        else:
            for _ in range(batch):
                Ra, Rb = sample_returns(P, r, qrng, 2)
                pairs.append((Ra, Rb)); ys.append(answer(Ra, Rb))
        if len(ys) in BUDGETS:
            Wn = fit(np.array([features(a) - features(b) for a, b in pairs]), np.array(ys), qrng)
            pol = esr_policy(P, r, FG @ Wn.mean(0))[1]
            res = {"reg": (v_esr - evaluate_aug(P, r, pol, U)) / den}
            if strategy == "random":                         # linear fit on the same data
                X = np.array([a - b for a, b in pairs], float)
                y, w = np.array(ys), np.zeros(2)
                for _ in range(2000):
                    z = 1 / (1 + np.exp(-X @ w))
                    w += 0.05 * X.T @ (y - z) / len(y) - 1e-3 * w
                w = np.clip(w, 0, None)
                w = w / w.sum() if w.sum() > 0 else np.array([0.5, 0.5])
                res["lin_reg"] = (v_esr - evaluate(P, r, linear_policy(P, r, w), U)[0]) / den
            out[len(ys)] = res
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()
    from multiprocessing import Pool
    unames = ["min", "thr0.6", "thr0.9"]
    jobs = [(s, u, st) for u in unames for st in ("random", "thompson") for s in range(args.n)]
    with Pool(8) as pool:
        res = pool.starmap(run, jobs)
    allres = {}
    for (s, u, st), o in zip(jobs, res):
        if o is not None:
            allres.setdefault(u, {}).setdefault(st, []).append(o)
    os.makedirs("esr/out", exist_ok=True)
    json.dump(allres, open("esr/out/k2.json", "w"))
    print("normalised regret of ESR policy planned on learned utility: median [p25, p75]")
    for u in unames:
        print(f"== {u}")
        for st in ("random", "thompson"):
            L = allres[u][st]
            row = []
            for b in BUDGETS:
                x = np.array([o[b]["reg"] for o in L])
                row.append(f"{b:3d}: {np.median(x):.3f} [{np.percentile(x, 25):.3f}, {np.percentile(x, 75):.3f}]")
            print(f"   {st:9s} " + "  ".join(row))
        L = allres[u]["random"]
        row = [f"{b:3d}: {np.median([o[b]['lin_reg'] for o in L]):.3f}" for b in BUDGETS]
        print(f"   {'linear':9s} " + "  ".join(row))


if __name__ == "__main__":
    main()
