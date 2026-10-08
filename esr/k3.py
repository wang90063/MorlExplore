"""A+B kill test K3: can ONE utility-conditioned network learn ESR-optimal (augmented) policies for a whole
family of kinked utilities, and generalise to unseen utility parameters?

Same random stochastic MOMDPs as K1/K2 (Dirichlet 1.0), so the exact ESR optimum per utility is available.
Utility families (normalised returns n_i = R_i / m_i):
  thr  u = n1 + 1[n2 >= theta],          theta in [0.3, 1.0]
  min  u = min(alpha n1, (1 - alpha) n2), alpha in [0.2, 0.8]
Q(s, t, R, utility, a) is trained by double DQN on terminal reward u(R_H). The transition data do not depend
on the utility, so every sampled batch is relabelled with fresh random utilities (utility relabelling).

  aug     network sees (s, t, R1, R2, utility)
  markov  ablation: network sees (s, t, utility) only

Test: 12 parameter values per family on a fixed grid (never exactly seen in training, which samples
continuously); policy table is read off the network and evaluated exactly. Reported: normalised regret vs
the exact ESR optimum for that utility (0 = optimal, 1 = uniform random), plus the oracle-linear baseline.

Usage: PYTHONPATH=esr python esr/k3.py [--n 6] [--steps 40000]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from k1 import A, H, M, S, evaluate, linear_policy, make_mdp
from k2 import GRID, esr_policy, evaluate_aug

torch.set_num_threads(1)
TEST = {"thr": np.linspace(0.32, 0.98, 12), "min": np.linspace(0.22, 0.78, 12)}


def util_table(fam, p, m1, m2):
    n1, n2 = GRID[..., 0] / m1, GRID[..., 1] / m2
    if fam == "thr":
        return n1 + 1.0 * (n2 >= p)
    return np.minimum(p * n1, (1 - p) * n2)


def util_batch(fam_is_thr, p, R, m1, m2):
    n1, n2 = R[:, 0] / m1, R[:, 1] / m2
    thr = n1 + (n2 >= p).float()
    mn = torch.minimum(p * n1, (1 - p) * n2)
    return torch.where(fam_is_thr, thr, mn)


def sample_utils(rng, n):
    is_thr = rng.random(n) < 0.5
    p = np.where(is_thr, rng.uniform(0.3, 1.0, n), rng.uniform(0.2, 0.8, n))
    return torch.as_tensor(is_thr), torch.as_tensor(p, dtype=torch.float32)


class QNet(nn.Module):
    def __init__(self, aug, h=256):
        super().__init__()
        self.aug = aug
        d = S + 1 + (2 if aug else 0) + 3
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                               nn.Linear(h, A))

    def forward(self, s, t, R, is_thr, p):
        x = [nn.functional.one_hot(s, S).float(), (t.float() / H)[:, None]]
        if self.aug:
            x.append(R.float() / (M - 1))
        x += [is_thr.float()[:, None], (~is_thr).float()[:, None], p[:, None]]
        return self.f(torch.cat(x, 1))


def train(seed, aug, steps, log_every=10_000):
    rng = np.random.default_rng(seed)
    P, r = make_mdp(rng, 1.0)
    lin = lambda w: GRID[..., 0] * w[0] + GRID[..., 1] * w[1]
    m1 = evaluate(P, r, linear_policy(P, r, np.array([1.0, 0])), lin([1, 0]))[0]
    m2 = evaluate(P, r, linear_policy(P, r, np.array([0, 1.0])), lin([0, 1]))[0]
    if min(m1, m2) <= 0:
        return None
    torch.manual_seed(seed)
    q, qt = QNet(aug), QNet(aug)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=3e-4)
    cum = P.cumsum(-1)
    rt = torch.as_tensor(r)

    cap, n_env = 200_000, 32
    B = {k: np.zeros(cap, int) for k in ("s", "t", "a", "s2")}
    B["R"], B["R2"] = np.zeros((cap, 2), int), np.zeros((cap, 2), int)
    ptr, size = 0, 0
    hist = []
    for it in range(steps):
        if it % 4 == 0:                                            # collect n_env episodes every 4 updates
            eps = max(0.1, 1 - it / (0.5 * steps))
            s, R = np.zeros(n_env, int), np.zeros((n_env, 2), int)
            ut, up = sample_utils(rng, n_env)
            for t in range(H):
                with torch.no_grad():
                    a = q(torch.as_tensor(s), torch.full((n_env,), t), torch.as_tensor(R), ut, up).argmax(1).numpy()
                a = np.where(rng.random(n_env) < eps, rng.integers(0, A, n_env), a)
                Rn = R + r[s, a]
                s2 = (rng.random(n_env)[:, None] < cum[s, a]).argmax(1)
                idx = (ptr + np.arange(n_env)) % cap
                B["s"][idx], B["t"][idx], B["a"][idx], B["s2"][idx], B["R"][idx], B["R2"][idx] = s, t, a, s2, R, Rn
                ptr, size = (ptr + n_env) % cap, min(size + n_env, cap)
                s, R = s2, Rn
        i = rng.integers(0, size, 256)
        s, t, a, s2 = (torch.as_tensor(B[k][i]) for k in ("s", "t", "a", "s2"))
        R, R2 = torch.as_tensor(B["R"][i]), torch.as_tensor(B["R2"][i])
        ut, up = sample_utils(rng, 256)                            # utility relabelling
        with torch.no_grad():
            last = t + 1 == H
            a2 = q(s2, t + 1, R2, ut, up).argmax(1, keepdim=True)
            boot = qt(s2, t + 1, R2, ut, up).gather(1, a2).squeeze(1)
            y = torch.where(last, util_batch(ut, up, R2.float(), m1, m2), boot)
        pred = q(s, t, R, ut, up).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 1000 == 0:
            qt.load_state_dict(q.state_dict())
        if (it + 1) % log_every == 0:
            hist.append(float(loss))

    # exact evaluation on the test grid
    ss, tt, R1, R2 = np.meshgrid(np.arange(S), np.arange(H), np.arange(M), np.arange(M), indexing="ij")
    ss, tt = torch.as_tensor(ss.ravel()), torch.as_tensor(tt.ravel())
    RR = torch.as_tensor(np.stack([R1.ravel(), R2.ravel()], 1))
    ws = np.stack([np.linspace(0, 1, 101), 1 - np.linspace(0, 1, 101)], 1)
    lin_pols = [linear_policy(P, r, w) for w in ws]
    res = {}
    for fam, ps in TEST.items():
        for p in ps:
            U = util_table(fam, p, m1, m2)
            v_esr, _ = esr_policy(P, r, U)
            v_rand = evaluate(P, r, np.full((H, S, A), 1 / A), U)[0]
            den = max(v_esr - v_rand, 1e-9)
            n = len(ss)
            with torch.no_grad():
                a = q(ss, tt, RR, torch.full((n,), fam == "thr"), torch.full((n,), float(p))).argmax(1).numpy()
            pol = a.reshape(S, H, M, M).transpose(1, 0, 2, 3)
            v_net = evaluate_aug(P, r, pol, U)
            v_lin = max(evaluate(P, r, lp, U)[0] for lp in lin_pols)
            res[f"{fam}{p:.2f}"] = dict(fam=fam, p=float(p), reg=(v_esr - v_net) / den, lin=(v_esr - v_lin) / den)
    return dict(res=res, loss=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=40_000)
    args = ap.parse_args()
    from multiprocessing import Pool
    jobs = [(s, aug, args.steps) for aug in (True, False) for s in range(args.n)]
    with Pool(8) as pool:
        out = pool.starmap(train, jobs)
    allres = {}
    for (s, aug, _), o in zip(jobs, out):
        if o is not None:
            allres.setdefault("aug" if aug else "markov", []).append(o)
    os.makedirs("esr/out", exist_ok=True)
    json.dump(allres, open("esr/out/k3.json", "w"))
    print(f"normalised regret vs exact ESR optimum on unseen utility parameters ({args.steps} updates)")
    for fam in ("thr", "min"):
        print(f"== {fam}")
        keys = [k for k in allres["aug"][0]["res"] if k.startswith(fam)]
        for k in keys:
            row = [f"{v} {np.median([o['res'][k]['reg'] for o in allres[v]]):.3f}" for v in ("aug", "markov")]
            print(f"   {k:8s} " + "   ".join(row) + f"   linear {np.median([o['res'][k]['lin'] for o in allres['aug']]):.3f}")
        for v in ("aug", "markov"):
            x = [o["res"][k]["reg"] for o in allres[v] for k in keys]
            print(f"   {fam} overall {v:6s} median {np.median(x):.3f}  mean {np.mean(x):.3f}  p90 {np.percentile(x, 90):.3f}")


if __name__ == "__main__":
    main()
