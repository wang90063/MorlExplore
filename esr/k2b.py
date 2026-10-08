"""K2b: fix the rare-threshold failure of K2 (thr0.9).

Diagnosis from K2: with an L1 prior every ensemble member sets weights on unseen step features to 0, so no
posterior sample values high R2 and Thompson queries never go near the threshold (0-3 % of queried returns).

Fix: anchored ensemble (randomised prior functions). Member k draws an anchor w0_k from a sparse monotone
prior (linear terms + a few random steps) and is fitted with an L2 pull towards w0_k instead of towards 0.
Where the data says nothing, members keep their prior guesses, so Thompson queries explore them.

Variants (all with Thompson queries):
  l1        K2 fit (reference)
  anchor    anchored ensemble, plan on the ensemble-mean utility
  anchor_lo anchored ensemble, plan on the pointwise 25 % quantile utility (conservative; still monotone)
  anchor_qbc anchored ensemble, query pair chosen by maximum member disagreement from a pool of rollouts
             of member policies; plan on the mean

Usage: PYTHONPATH=esr python esr/k2b.py [--n 30]
"""
import argparse
import json
import os

import numpy as np

from k1 import A, H, S, evaluate, linear_policy, make_mdp, sample_returns
from k2 import BUDGETS, FG, GRID, NF, esr_policy, evaluate_aug, features, fit, rollout_aug, true_utilities

N_LIN, N_STEP = 2, 40                          # feature layout: [lin(2), R1 steps(20), R2 steps(20), joint]


def prior_draw(rng, scale=5.0, p_step=0.15, p_joint=0.02):
    w = np.zeros(NF)
    w[:N_LIN] = rng.exponential(scale, N_LIN)
    m = rng.random(N_STEP) < p_step
    w[N_LIN:N_LIN + N_STEP][m] = rng.exponential(scale, m.sum())
    m = rng.random(NF - N_LIN - N_STEP) < p_joint
    w[N_LIN + N_STEP:][m] = rng.exponential(scale, m.sum())
    return w


def fit_anchor(X, y, anchors, rng, iters=400, lam=1.0):
    W = []
    for w0 in anchors:
        idx = rng.integers(0, len(y), len(y))
        Xb, yb = X[idx], y[idx]
        w = w0.copy()
        lr = 1.0 / (len(yb) + lam * 10)
        for _ in range(iters):
            z = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
            w = np.maximum(w + lr * (Xb.T @ (yb - z) - lam * (w - w0)), 0)
        W.append(w)
    return np.array(W)


def run(seed, uname, variant, temp=0.05, n_ens=8):
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
    q = np.random.default_rng(1000 + seed)
    anchors = [prior_draw(q) for _ in range(n_ens)]

    def answer(Ra, Rb):
        return float(q.random() < 1 / (1 + np.exp(-(u(Ra) - u(Rb)) / (temp * scale))))

    def fitW(pairs, ys):
        X, y = np.array([features(a) - features(b) for a, b in pairs]), np.array(ys)
        return fit(X, y, q, n_ens) if variant == "l1" else fit_anchor(X, y, anchors, q)

    pairs, ys = [], []
    for _ in range(5):
        Ra, Rb = sample_returns(P, r, q, 2)
        pairs.append((Ra, Rb)); ys.append(answer(Ra, Rb))
    out = {}
    while len(ys) < BUDGETS[-1]:
        W = fitW(pairs, ys)
        pols = [esr_policy(P, r, FG @ w)[1] for w in W]
        batch = 5 if len(ys) < 50 else 10
        for _ in range(batch):
            if variant == "anchor_qbc":
                pool = np.array([rollout_aug(P, r, pols[k], q) for k in range(len(W)) for _ in range(3)])
                F = features(pool) @ W.T                                  # (pool, members)
                pref = (F[:, None, :] > F[None, :, :]).mean(-1)          # fraction of members preferring i over j
                a, b = np.unravel_index(np.argmax(pref * (1 - pref)), pref.shape)
                Ra, Rb = pool[a], pool[b]
            else:
                i, j = q.choice(len(W), 2, replace=False)
                Ra, Rb = rollout_aug(P, r, pols[i], q), rollout_aug(P, r, pols[j], q)
            pairs.append((Ra, Rb)); ys.append(answer(Ra, Rb))
        if len(ys) in BUDGETS:
            W = fitW(pairs, ys)
            Uh = FG @ W.T                                                 # (M, M, members)
            Up = np.quantile(Uh, 0.25, -1) if variant == "anchor_lo" else Uh.mean(-1)
            pol = esr_policy(P, r, Up)[1]
            out[len(ys)] = (v_esr - evaluate_aug(P, r, pol, U)) / den
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()
    from multiprocessing import Pool
    unames = ["min", "thr0.6", "thr0.9"]
    variants = ["l1", "anchor", "anchor_lo", "anchor_qbc"]
    jobs = [(s, u, v) for u in unames for v in variants for s in range(args.n)]
    with Pool(8) as pool:
        res = pool.starmap(run, jobs)
    allres = {}
    for (s, u, v), o in zip(jobs, res):
        if o is not None:
            allres.setdefault(u, {}).setdefault(v, []).append(o)
    os.makedirs("esr/out", exist_ok=True)
    json.dump(allres, open("esr/out/k2b.json", "w"))
    print("normalised regret, Thompson-style queries: median [p25, p75]")
    for u in unames:
        print(f"== {u}")
        for v in variants:
            L = allres[u][v]
            row = [f"{b:3d}: {np.median([o[b] for o in L]):.3f} [{np.percentile([o[b] for o in L], 25):.3f}, "
                   f"{np.percentile([o[b] for o in L], 75):.3f}]" for b in BUDGETS]
            print(f"   {v:10s} " + "  ".join(row))


if __name__ == "__main__":
    main()
