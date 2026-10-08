"""A+B kill test K1: how much utility do linear / Markov surrogates lose against the exact ESR optimum?

Finite-horizon two-objective MOMDPs, integer rewards, undiscounted. The user's utility u is applied
to the episode's return vector (ESR: maximise E[u(R)]). Everything is computed exactly.

  esr      exact optimum: DP over the augmented state (s, t, accrued return)
  lin_or   oracle linear: best (by true ESR utility) of the 101 linear-scalarisation optimal policies.
           Upper bound for any method that elicits a linear weight and then optimises it.
  lin_fit  linear weight fitted by logistic regression on 300 pairwise trajectory comparisons
           (Bradley-Terry on u), then optimised.
  markov   best time-dependent Markov policy found by coordinate ascent on the exact ESR utility,
           started from lin_or (a lower bound on the best non-augmented policy).
  ser      Markov policy found the same way but maximising the SER objective u(E[R]) (what existing
           non-linear SER methods optimise), then scored by its true ESR utility.
  rand     uniform random policy (for normalisation)

Reported: normalised regret (U_esr - U_b) / (U_esr - U_rand).
Kill if lin_or regret is small (< 5 %) across utilities in the stochastic settings.

Usage: python esr/k1.py [--n 30]
"""
import argparse
import json
import os

import numpy as np

S, A, H = 8, 3, 10
M = 2 * H + 1                                   # accrued return per objective lies in 0..2H


def make_mdp(rng, stoch):
    """stoch: 'det' (one successor), or a Dirichlet concentration (smaller = sparser)."""
    if stoch == "det":
        P = np.zeros((S, A, S))
        P[np.arange(S)[:, None], np.arange(A)[None], rng.integers(0, S, (S, A))] = 1
    else:
        P = rng.dirichlet(np.full(S, float(stoch)), (S, A))
    k = rng.integers(0, 3, (S, A))
    r = np.stack([k, 2 - k], -1)                 # conflicting objectives, (k, 2-k)
    r[rng.random((S, A)) < 0.2] = 0              # some zero-reward actions
    return P, r


def shift(V, r):
    """V[..., R1, R2] -> V[..., R1 + r1, R2 + r2] (values beyond the grid are never reached)."""
    out = np.zeros_like(V)
    out[..., : M - r[0], : M - r[1]] = V[..., r[0]:, r[1]:]
    return out


def esr_opt(P, r, U):
    """Exact ESR optimum; U[R1, R2] utility table. Returns value at (s0, t0, R=0)."""
    V = np.broadcast_to(U, (S, M, M)).copy()
    for _ in range(H):
        EV = np.einsum("sat,tij->saij", P, V)    # expectation over s' at fixed accrued R
        Q = np.stack([[shift(EV[s, a], r[s, a]) for a in range(A)] for s in range(S)])
        V = Q.max(1)
    return V[0, 0, 0]


def evaluate(P, r, pi, U):
    """ESR utility of a Markov policy pi[t, s, a] (stochastic allowed) by forward distribution."""
    D = np.zeros((S, M, M))
    D[0, 0, 0] = 1
    for t in range(H):
        N = np.zeros_like(D)
        for s in range(S):
            for a in range(A):
                if pi[t, s, a] == 0:
                    continue
                moved = np.zeros((M, M))
                moved[r[s, a, 0]:, r[s, a, 1]:] = D[s, : M - r[s, a, 0], : M - r[s, a, 1]]
                N += pi[t, s, a] * P[s, a][:, None, None] * moved[None]
        D = N
    return float((D.sum(0) * U).sum()), D.sum(0)


def linear_policy(P, r, w):
    pi = np.zeros((H, S, A))
    V = np.zeros(S)
    for t in reversed(range(H)):
        Q = r @ w + P @ V
        a = Q.argmax(1)
        pi[t, np.arange(S), a] = 1
        V = Q.max(1)
    return pi


def coord_ascent(P, r, pi, U, sweeps=4, obj=None):
    """Greedy single-entry improvement of a deterministic Markov policy. obj(D) -> score; default ESR."""
    score = obj or (lambda D: float((D * U).sum()))
    best = score(evaluate(P, r, pi, U)[1])
    for _ in range(sweeps):
        improved = False
        for t in range(H):
            for s in range(S):
                for a in range(A):
                    if pi[t, s, a] == 1:
                        continue
                    cand = pi.copy()
                    cand[t, s] = 0
                    cand[t, s, a] = 1
                    v = score(evaluate(P, r, cand, U)[1])
                    if v > best + 1e-12:
                        best, pi, improved = v, cand, True
        if not improved:
            break
    return pi


def sample_returns(P, r, rng, n):
    out = np.zeros((n, 2), int)
    for i in range(n):
        s = 0
        for _ in range(H):
            a = rng.integers(A)
            out[i] += r[s, a]
            s = rng.choice(S, p=P[s, a])
    return out


def fit_linear(P, r, u, rng, n_pairs=300, temp=0.05):
    R = sample_returns(P, r, rng, 2 * n_pairs)
    Ra, Rb = R[:n_pairs], R[n_pairs:]
    ua, ub = u(Ra), u(Rb)
    scale = max(np.ptp(np.concatenate([ua, ub])), 1e-9)
    p = 1 / (1 + np.exp(-(ua - ub) / (temp * scale)))
    y = (rng.random(n_pairs) < p).astype(float)
    X = (Ra - Rb).astype(float)
    w = np.zeros(2)
    for _ in range(2000):                        # logistic regression, no intercept, plain GD
        z = 1 / (1 + np.exp(-X @ w))
        w += 0.05 * X.T @ (y - z) / n_pairs - 1e-3 * w
    w = np.clip(w, 0, None)
    return w / w.sum() if w.sum() > 0 else np.array([0.5, 0.5])


def utilities(m1, m2):
    """Monotone non-linear utilities, scaled by the single-objective optimal expected returns m_i."""
    n = lambda R: (np.asarray(R, float)[..., 0] / m1, np.asarray(R, float)[..., 1] / m2)
    return {
        "min": lambda R: np.minimum(*n(R)),
        "concave": lambda R: np.sqrt(n(R)[0]) + np.sqrt(n(R)[1]),
        "threshold": lambda R: n(R)[0] + 1.0 * (n(R)[1] >= 0.6),
        "convex": lambda R: n(R)[0] ** 2 + n(R)[1] ** 2,
    }


def run_instance(seed, stoch):
    rng = np.random.default_rng(seed)
    P, r = make_mdp(rng, stoch)
    grid = np.stack(np.meshgrid(np.arange(M), np.arange(M), indexing="ij"), -1)
    lin = lambda R, w: R[..., 0] * w[0] + R[..., 1] * w[1]
    m1 = evaluate(P, r, linear_policy(P, r, np.array([1.0, 0])), lin(grid, [1, 0]))[0]
    m2 = evaluate(P, r, linear_policy(P, r, np.array([0, 1.0])), lin(grid, [0, 1]))[0]
    if min(m1, m2) <= 0:
        return None
    ws = np.stack([np.linspace(0, 1, 101), 1 - np.linspace(0, 1, 101)], 1)
    lin_pols = [linear_policy(P, r, w) for w in ws]
    out = {}
    for name, u in utilities(m1, m2).items():
        U = u(grid)
        v_esr = esr_opt(P, r, U)
        vals = [evaluate(P, r, p, U)[0] for p in lin_pols]
        i = int(np.argmax(vals))
        v_lin = vals[i]
        v_fit = evaluate(P, r, linear_policy(P, r, fit_linear(P, r, u, rng)), U)[0]
        v_mk = max(evaluate(P, r, coord_ascent(P, r, lin_pols[i].copy(), U), U)[0], v_lin)
        ser = lambda D: float(u((D[..., None] * grid).sum((0, 1))))
        j = int(np.argmax([ser(evaluate(P, r, p, U)[1]) for p in lin_pols]))
        v_ser = evaluate(P, r, coord_ascent(P, r, lin_pols[j].copy(), U, obj=ser), U)[0]
        v_rand = evaluate(P, r, np.full((H, S, A), 1 / A), U)[0]
        den = max(v_esr - v_rand, 1e-9)
        out[name] = dict(esr=v_esr, lin_or=v_lin, lin_fit=v_fit, markov=v_mk, ser=v_ser, rand=v_rand,
                         reg_lin_or=(v_esr - v_lin) / den, reg_lin_fit=(v_esr - v_fit) / den,
                         reg_markov=(v_esr - v_mk) / den, reg_ser=(v_esr - v_ser) / den)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()
    from multiprocessing import Pool
    settings = ["det", 0.1, 1.0]
    jobs = [(seed, st) for st in settings for seed in range(args.n)]
    with Pool(8) as pool:
        res = pool.starmap(run_instance, jobs)
    allres = {}
    for (seed, st), o in zip(jobs, res):
        if o is not None:
            allres.setdefault(str(st), []).append(o)
    os.makedirs("esr/out", exist_ok=True)
    json.dump(allres, open("esr/out/k1.json", "w"))
    print("normalised regret vs exact ESR optimum: median [p75, max]   (n instances)")
    for st, L in allres.items():
        print(f"== transitions: {st}  (n={len(L)})")
        for u in L[0]:
            row = []
            for b in ("lin_or", "lin_fit", "ser", "markov"):
                x = np.array([o[u]["reg_" + b] for o in L])
                row.append(f"{b} {np.median(x):.3f} [{np.percentile(x, 75):.3f}, {x.max():.3f}]")
            frac = np.mean([o[u]["reg_lin_or"] > 0.05 for o in L])
            print(f"   {u:9s} " + "   ".join(row) + f"   lin_or>5%: {frac:.0%}")


if __name__ == "__main__":
    main()
