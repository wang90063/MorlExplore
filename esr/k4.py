"""A+B kill test K4: the full interactive loop on a continuous-state stochastic environment.

Environment ("two patches"): position in [0, 1]^2, start near (0.5, 0.5). Actions stay / left / right /
up / down, step 0.08 plus Gaussian noise (sd 0.03). Inside patch A (centre (0.35, 0.5), radius 0.12) each
step pays objective 1 with prob 0.6; inside patch B (centre (0.65, 0.5)) it pays objective 2 with prob 0.5.
(A first version with patches at 0.2 / 0.8 made switching too slow to matter: linear was optimal.)
Horizon 20, so returns are integers on the K2 grid and the K2 monotone features apply unchanged.

1. Offline: one Q-network conditioned on (pos, t, accrued R, utility weights w) is trained by double DQN
   with utility relabelling. w is drawn from the K2b sparse monotone prior: single draws, means of 2-8
   draws (posterior-mean-like), random joint-step staircases (complementarity) and pure linear weights. The test utilities are never used in training.
2. Online, per true utility and seed: anchored-ensemble posterior from Bradley-Terry comparisons; Thompson
   queries are rollouts of the network conditioned on two members' w (no re-planning); the decision policy
   is the network conditioned on the ensemble-mean w.
Baselines: same network conditioned on a fitted linear weight (from random-policy comparisons, as in K2);
reference: a DQN trained on the true utility alone (also the network conditioned on the true w).
Metric: Monte-Carlo true utility, normalised regret vs the best reference (0) and uniform random (1).

Usage: PYTHONPATH=esr python esr/k4.py train      # trains conditioned nets + references, saves esr/out/k4_*.pt
       PYTHONPATH=esr python esr/k4.py loop [--n 20]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from k2 import BUDGETS, M, NF, features
from k2b import fit_anchor, prior_draw

torch.set_num_threads(1)
H, NA = 20, 5
CA, CB, RAD = np.array([0.35, 0.5]), np.array([0.65, 0.5]), 0.12
PA, PB = 0.6, 0.5
MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]]) * 0.08
OUT = "esr/out"


def env_reset(rng, n):
    return np.clip(0.5 + rng.normal(0, 0.02, (n, 2)), 0, 1)


def env_step(rng, pos, a):
    pos = np.clip(pos + MOVES[a] + rng.normal(0, 0.03, pos.shape), 0, 1)
    r1 = (np.linalg.norm(pos - CA, axis=1) < RAD) & (rng.random(len(pos)) < PA)
    r2 = (np.linalg.norm(pos - CB, axis=1) < RAD) & (rng.random(len(pos)) < PB)
    return pos, np.stack([r1, r2], 1).astype(int)


# ---------- utilities as feature weights ----------
KJ = torch.arange(2, M, 2)
K1 = torch.arange(1, M)


def feat_t(R):
    """Torch version of k2.features for integer returns R (n, 2)."""
    R = R.float()
    lin = R / (M - 1)
    s1 = (R[:, :1] >= K1).float()
    s2 = (R[:, 1:] >= K1).float()
    j = ((R[:, 0, None, None] >= KJ[:, None]) & (R[:, 1, None, None] >= KJ[None])).float().reshape(len(R), -1)
    return torch.cat([lin, s1, s2, j], 1)


CORNER = features(np.array([M - 1, M - 1]))


def normalise(w):
    """Scale so that u(max corner) = 1 (u is monotone, so u in [0, 1] on the grid)."""
    w = np.asarray(w, float)
    return w / max(CORNER @ w, 1e-9)


def staircase(rng):
    """Random increasing chain of joint steps along a random ray, plus a small random prior draw."""
    w = 0.3 * prior_draw(rng)
    kk = np.arange(2, M, 2)
    rho = np.exp(rng.uniform(-1, 1))
    for x in np.linspace(rng.uniform(1, 3), rng.uniform(8, 20), rng.integers(3, 10)):
        a, b = np.searchsorted(kk, x), np.searchsorted(kk, x * rho)
        if a < len(kk) and b < len(kk):
            w[2 + 40 + a * len(kk) + b] += rng.exponential(5.0)
    return w


def sample_w(rng):
    k = rng.random()
    if k < 0.3:
        w = prior_draw(rng)
    elif k < 0.6:
        w = np.mean([prior_draw(rng) for _ in range(rng.integers(2, 9))], 0)
    elif k < 0.85:
        w = staircase(rng) if rng.random() < 0.7 else np.mean([staircase(rng), prior_draw(rng)], 0)
    else:
        w = np.zeros(NF)
        w[:2] = rng.dirichlet([1, 1])
    return normalise(w)


def true_ws(m1, m2):
    """Test utilities written in the feature basis (exact for thresholds; min via joint steps)."""
    out = {}
    for th in (0.3, 0.5):
        w = np.zeros(NF)
        w[0] = (M - 1) / m1                                      # n1 = R1 / m1
        T = int(np.ceil(th * m2))
        w[2 + 20 + T - 1] = 1.0                                  # 1[R2 >= T]
        out[f"thr{th}"] = w
    # u = min(n1, n2) on the coarse joint lattice: sum over k of 1[R1 >= a_k, R2 >= b_k] steps
    w = np.zeros(NF)
    kk = np.arange(2, M, 2)
    for x in np.linspace(0.1, 1.5, 15):
        a, b = np.searchsorted(kk, x * m1), np.searchsorted(kk, x * m2)
        if a < len(kk) and b < len(kk):
            w[2 + 40 + a * len(kk) + b] += 0.1
    out["min"] = w
    return {k: normalise(v) for k, v in out.items()}


# ---------- network ----------
class QNet(nn.Module):
    def __init__(self, cond=True, h=256):
        super().__init__()
        self.cond = cond
        d = 2 + 1 + 2 + (NF if cond else 0)
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                               nn.Linear(h, NA))

    def forward(self, pos, t, R, w):
        x = [pos, (t.float() / H)[:, None], R.float() / (M - 1)]
        if self.cond:
            x.append(w * 10)
        return self.f(torch.cat(x, 1))


def act(q, pos, t, R, w):
    with torch.no_grad():
        return q(torch.as_tensor(pos, dtype=torch.float32), torch.full((len(pos),), t),
                 torch.as_tensor(R), torch.as_tensor(w, dtype=torch.float32)).argmax(1).numpy()


def rollout(q, w, rng, n):
    """n episodes of the greedy policy conditioned on w (one w for all, or (n, NF))."""
    W = np.broadcast_to(w, (n, NF)) if np.ndim(w) == 1 else w
    pos, R = env_reset(rng, n), np.zeros((n, 2), int)
    for t in range(H):
        pos, r = env_step(rng, pos, act(q, pos, t, R, W))
        R = R + r
    return R


def random_returns(rng, n):
    pos, R = env_reset(rng, n), np.zeros((n, 2), int)
    for t in range(H):
        pos, r = env_step(rng, pos, rng.integers(0, NA, n))
        R = R + r
    return R


def train(seed, fixed_w=None, steps=300_000, tag="cond"):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    cond = fixed_w is None
    q, qt = QNet(cond), QNet(cond)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=3e-4)
    cap, n_env = 400_000, 32
    B = dict(p=np.zeros((cap, 2), np.float32), p2=np.zeros((cap, 2), np.float32), t=np.zeros(cap, int),
             a=np.zeros(cap, int), R=np.zeros((cap, 2), int), R2=np.zeros((cap, 2), int))
    ptr = size = 0
    wpool = np.array([sample_w(rng) for _ in range(20_000)]) if cond else None
    fw = torch.as_tensor(fixed_w, dtype=torch.float32) if not cond else None
    for it in range(steps):
        if it % 8 == 0:
            eps = max(0.05, 1 - it / (0.4 * steps))
            W = wpool[rng.integers(0, len(wpool), n_env)] if cond else np.broadcast_to(fixed_w, (n_env, NF))
            pos, R = env_reset(rng, n_env), np.zeros((n_env, 2), int)
            for t in range(H):
                a = act(q, pos, t, R, W)
                a = np.where(rng.random(n_env) < eps, rng.integers(0, NA, n_env), a)
                p2, r = env_step(rng, pos, a)
                idx = (ptr + np.arange(n_env)) % cap
                B["p"][idx], B["p2"][idx], B["t"][idx], B["a"][idx], B["R"][idx], B["R2"][idx] = pos, p2, t, a, R, R + r
                ptr, size = (ptr + n_env) % cap, min(size + n_env, cap)
                pos, R = p2, R + r
        i = rng.integers(0, size, 256)
        p, p2, t, a = (torch.as_tensor(B[k][i]) for k in ("p", "p2", "t", "a"))
        R, R2 = torch.as_tensor(B["R"][i]), torch.as_tensor(B["R2"][i])
        w = torch.as_tensor(wpool[rng.integers(0, len(wpool), 256)], dtype=torch.float32) if cond else fw.expand(256, -1)
        with torch.no_grad():
            a2 = q(p2, t + 1, R2, w).argmax(1, keepdim=True)
            boot = qt(p2, t + 1, R2, w).gather(1, a2).squeeze(1)
            y = torch.where(t + 1 == H, (feat_t(R2) * w).sum(1), boot)
        loss = nn.functional.smooth_l1_loss(q(p, t, R, w).gather(1, a[:, None]).squeeze(1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 1000 == 0:
            qt.load_state_dict(q.state_dict())
        if (it + 1) % 50_000 == 0:
            print(json.dumps(dict(tag=tag, it=it + 1, loss=float(loss))), flush=True)
    torch.save(dict(state=q.state_dict(), cond=cond), f"{OUT}/k4_{tag}.pt")


def scales():
    """m_i: expected return of the scripted 'go to patch i and stay' policy."""
    rng = np.random.default_rng(0)
    out = []
    for c in (CA, CB):
        pos, R = env_reset(rng, 20_000), np.zeros((20_000, 2), int)
        for t in range(H):
            d = c - pos
            a = np.where(np.abs(d[:, 0]) > 0.04, np.where(d[:, 0] < 0, 1, 2), 0)
            pos, r = env_step(rng, pos, a)
            R = R + r
        out.append(R.mean(0)[len(out)])
    return out


def load(tag):
    d = torch.load(f"{OUT}/k4_{tag}.pt")
    q = QNet(d["cond"])
    q.load_state_dict(d["state"])
    return q


def mc_util(R, w):
    return float((features(R) @ w).mean())


def loop_run(seed, uname, n_eval=4000, temp=0.05, n_ens=8):
    m1, m2 = scales()
    wt = true_ws(m1, m2)[uname]
    u = lambda R: features(R) @ wt
    q = load("cond")
    ev = np.random.default_rng(99)
    refs = {"dqn_true": mc_util(rollout(load(f"ref_{uname}"), wt, ev, n_eval), wt),
            "cond_true": mc_util(rollout(q, wt, ev, n_eval), wt)}
    v_ref = max(refs.values())
    v_rand = mc_util(random_returns(ev, n_eval), wt)
    reg = lambda v: (v_ref - v) / max(v_ref - v_rand, 1e-9)
    rng = np.random.default_rng(1000 + seed)
    anchors = [prior_draw(rng) for _ in range(n_ens)]

    def answer(Ra, Rb):
        return float(rng.random() < 1 / (1 + np.exp(-(u(Ra) - u(Rb)) / temp)))     # u is in [0, 1]

    pairs, ys, lpairs, lys = [], [], [], []
    for _ in range(5):
        Ra, Rb = random_returns(rng, 2)
        pairs.append((Ra, Rb)); ys.append(answer(Ra, Rb))
    out = {}
    while len(ys) < BUDGETS[-1]:
        X = np.array([features(a) - features(b) for a, b in pairs])
        W = fit_anchor(X, np.array(ys), anchors, rng)
        batch = 5 if len(ys) < 50 else 10
        for _ in range(batch):
            i, j = rng.choice(n_ens, 2, replace=False)
            Ra, Rb = rollout(q, normalise(W[i]), rng, 1)[0], rollout(q, normalise(W[j]), rng, 1)[0]
            pairs.append((Ra, Rb)); ys.append(answer(Ra, Rb))
        while len(lys) < len(ys):                                # linear baseline: random-policy comparisons
            Ra, Rb = random_returns(rng, 2)
            lpairs.append((Ra, Rb)); lys.append(answer(Ra, Rb))
        if len(ys) in BUDGETS:
            X = np.array([features(a) - features(b) for a, b in pairs])
            W = fit_anchor(X, np.array(ys), anchors, rng)
            wm = normalise(np.mean([normalise(w) for w in W], 0))
            v = mc_util(rollout(q, wm, ev, n_eval), wt)
            Xl, yl, wl = np.array([a - b for a, b in lpairs], float), np.array(lys), np.zeros(2)
            for _ in range(2000):
                z = 1 / (1 + np.exp(-Xl @ wl))
                wl += 0.05 * Xl.T @ (yl - z) / len(yl) - 1e-3 * wl
            wl = np.clip(wl, 1e-6, None)
            wlin = np.zeros(NF)
            wlin[:2] = wl / wl.sum()
            vl = mc_util(rollout(q, normalise(wlin), ev, n_eval), wt)
            out[len(ys)] = dict(loop=reg(v), lin=reg(vl))
    out["refs"] = {k: reg(v) for k, v in refs.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "loop"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=300_000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from multiprocessing import Pool
    unames = ["thr0.3", "thr0.5", "min"]
    if args.mode == "train":
        m1, m2 = scales()
        print(json.dumps(dict(m1=m1, m2=m2)), flush=True)
        tw = true_ws(m1, m2)
        jobs = [(0, None, args.steps, "cond")] + [(1, tw[u], args.steps // 2, f"ref_{u}") for u in unames]
        with Pool(4) as pool:
            pool.starmap(train, jobs)
        return
    jobs = [(s, u) for u in unames for s in range(args.n)]
    with Pool(8) as pool:
        res = pool.starmap(loop_run, jobs)
    allres = {}
    for (s, u), o in zip(jobs, res):
        allres.setdefault(u, []).append(o)
    json.dump(allres, open(f"{OUT}/k4.json", "w"))
    print("normalised regret (0 = best reference, 1 = uniform random): median [p25, p75]")
    for u in unames:
        L = allres[u]
        print(f"== {u}   refs: " + "  ".join(f"{k} {L[0]['refs'][k]:.3f}" for k in L[0]["refs"]))
        for key in ("loop", "lin"):
            row = [f"{b:3d}: {np.median([o[b][key] for o in L]):.3f} [{np.percentile([o[b][key] for o in L], 25):.3f}, "
                   f"{np.percentile([o[b][key] for o in L], 75):.3f}]" for b in BUDGETS]
            print(f"   {key:5s} " + "  ".join(row))


if __name__ == "__main__":
    main()
