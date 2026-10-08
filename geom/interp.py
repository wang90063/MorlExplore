"""Candidate 2 kill test: interpolation barriers between per-preference SAC actors,
raw and after permutation alignment (Git Re-Basin weight matching).

For a pair (a, b), theta(l) = (1 - l) theta_a + l theta_b. With J the vector return and
chord(l) = (1 - l) J_a + l J_b, the barrier is

    max_l  w(l) . (chord(l) - J(l))      in range-normalised objective units,

where w(l) interpolates the two preferences, each mapped into normalised units and
rescaled to sum to 1. This is a Rewarded-Soups-style LMC check: barrier <= 0 means the
interpolated policy is at least as good as the chord for the interpolated preference.

Pairs fall into three groups: same c / different init (independent-run baseline),
different c / shared init, different c / different init.

Usage: python geom/interp.py [--runs geom/runs] [--episodes 3]
"""
import argparse
import glob
import itertools
import json
import os
from multiprocessing import Pool

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from sac_mo import Actor, evaluate

LAMBDAS = np.linspace(0, 1, 11)


def weight_match(sa, sb, iters=50):
    """Permute hidden units of actor b to match actor a. Returns permuted copy of sb."""
    W1a, W1b, b1a, b1b = sa["l1.weight"], sb["l1.weight"], sa["l1.bias"], sb["l1.bias"]
    W2a, W2b, b2a, b2b = sa["l2.weight"], sb["l2.weight"], sa["l2.bias"], sb["l2.bias"]
    Oa = torch.cat([sa["mu.weight"], sa["ls.weight"]], 0)
    Ob = torch.cat([sb["mu.weight"], sb["ls.weight"]], 0)
    h = W1a.shape[0]
    p1, p2 = torch.arange(h), torch.arange(h)
    for _ in range(iters):
        old = (p1.clone(), p2.clone())
        # layer-1 units: incoming weights + bias + outgoing weights into (already permuted) layer 2
        S1 = W1a @ W1b.T + b1a[:, None] * b1b[None] + W2a.T @ W2b[p2]
        p1 = torch.as_tensor(linear_sum_assignment(S1.numpy(), maximize=True)[1])
        S2 = W2a @ W2b[:, p1].T + b2a[:, None] * b2b[None] + Oa.T @ Ob
        p2 = torch.as_tensor(linear_sum_assignment(S2.numpy(), maximize=True)[1])
        if torch.equal(p1, old[0]) and torch.equal(p2, old[1]):
            break
    out = dict(sb)
    out["l1.weight"], out["l1.bias"] = W1b[p1], b1b[p1]
    out["l2.weight"], out["l2.bias"] = W2b[p2][:, p1], b2b[p2]
    out["mu.weight"], out["ls.weight"] = sb["mu.weight"][:, p2], sb["ls.weight"][:, p2]
    return out


def interp_curve(job):
    sa, sb, od, ad, env, episodes = job
    actor = Actor(od, ad)
    J = []
    for l in LAMBDAS:
        actor.load_state_dict({k: (1 - l) * sa[k] + l * sb[k] for k in sa})
        J.append(evaluate(actor, env, episodes))
    return np.array(J)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="geom/runs")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--procs", type=int, default=8)
    args = p.parse_args()

    R = [torch.load(f) for f in sorted(glob.glob(os.path.join(args.runs, "*.pt")))]
    env, od, ad = R[0]["env"], R[0]["od"], R[0]["ad"]
    pairs = list(itertools.combinations(range(len(R)), 2))
    jobs, meta = [], []
    for i, j in pairs:
        a, b = R[i], R[j]
        for mode in ("raw", "aligned"):
            sb = b["actor"] if mode == "raw" else weight_match(a["actor"], b["actor"])
            jobs.append((a["actor"], sb, od, ad, env, args.episodes))
            meta.append(dict(i=i, j=j, mode=mode, ca=a["c"], cb=b["c"],
                             shared_init=a["init_seed"] == b["init_seed"]))
    torch.set_num_threads(1)
    with Pool(args.procs) as pool:
        curves = pool.map(interp_curve, jobs)

    E = np.array([r["eval"] for r in R])
    lo, rngv = E.min(0), np.maximum(E.max(0) - E.min(0), 1e-8)

    def wn(c):                                     # preference in normalised units, sum 1
        v = np.array([1.0, c]) * rngv
        return v / v.sum()

    out = []
    for m, C in zip(meta, curves):
        Cn = (C - lo) / rngv
        chord = (1 - LAMBDAS[:, None]) * Cn[0] + LAMBDAS[:, None] * Cn[-1]
        W = (1 - LAMBDAS[:, None]) * wn(m["ca"]) + LAMBDAS[:, None] * wn(m["cb"])
        deficit = ((chord - Cn) * W).sum(1)
        group = ("same_c" if m["ca"] == m["cb"] else
                 "diff_c_shared_init" if m["shared_init"] else "diff_c_diff_init")
        out.append(dict(m, group=group, barrier=float(deficit.max()), curve=C.tolist()))
    json.dump(dict(endpoints=[dict(c=r["c"], seed=r["seed"], init=r["init_seed"], eval=r["eval"]) for r in R],
                   pairs=out), open(os.path.join(args.runs, "interp.json"), "w"))

    print("endpoints (c, seed, [speed, -energy]):")
    for r in R:
        print(f"   c={r['c']:<5} s={r['seed']}  {np.round(r['eval'], 0)}")
    print("\nbarrier (normalised units; chord - interpolated, max over lambda):")
    for g in ("same_c", "diff_c_shared_init", "diff_c_diff_init"):
        for mode in ("raw", "aligned"):
            b = [o["barrier"] for o in out if o["group"] == g and o["mode"] == mode]
            if b:
                print(f"   {g:20s} {mode:8s} n={len(b):2d}  median {np.median(b):.3f}  "
                      f"[{np.min(b):.3f}, {np.max(b):.3f}]")
    print("\nper pair (aligned):")
    for o in out:
        if o["mode"] == "aligned":
            raw = next(x["barrier"] for x in out if x["i"] == o["i"] and x["j"] == o["j"] and x["mode"] == "raw")
            print(f"   c {o['ca']:<4} vs {o['cb']:<4} {o['group']:20s} raw {raw:.3f}  aligned {o['barrier']:.3f}")


if __name__ == "__main__":
    main()
