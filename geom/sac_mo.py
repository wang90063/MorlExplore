"""Candidate 2 (Pareto policy-set geometry): train one SAC policy per fixed preference.

Scalarised reward r = r_0 + c * r_1 on an MO-Gymnasium env (default mo-halfcheetah-v5,
r_0 = forward velocity, r_1 = -energy). c = 0.1 is the standard HalfCheetah reward.
`--init_seed` fixes the network initialisation separately from the data seed, so runs with the
same init_seed share their starting weights (Rewarded-Soups-like) even for different c.

Saves the actor state_dict and a short eval to geom/runs/<tag>.pt.
"""
import argparse
import json
import os
import time

import gymnasium as gym
import mo_gymnasium as mog
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
LOG_STD_MIN, LOG_STD_MAX = -5, 2


class Actor(nn.Module):
    def __init__(self, od, ad, h=256):
        super().__init__()
        self.l1, self.l2 = nn.Linear(od, h), nn.Linear(h, h)
        self.mu, self.ls = nn.Linear(h, ad), nn.Linear(h, ad)

    def forward(self, o):
        x = F.relu(self.l2(F.relu(self.l1(o))))
        return self.mu(x), self.ls(x).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, o):
        mu, ls = self(o)
        std = ls.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp

    def act(self, o):
        return torch.tanh(self(o)[0])


class Critic(nn.Module):
    def __init__(self, od, ad, h=256):
        super().__init__()
        mk = lambda: nn.Sequential(nn.Linear(od + ad, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.q1, self.q2 = mk(), mk()

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def make_env(env_id):
    return gym.wrappers.ClipAction(mog.make(env_id))


def evaluate(actor, env_id, episodes=3, seed=10_000):
    """Deterministic policy; returns mean undiscounted vector return."""
    env = make_env(env_id)
    rets = []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        R, done = 0.0, False
        while not done:
            with torch.no_grad():
                a = actor.act(torch.as_tensor(o, dtype=torch.float32)).numpy()
            o, r, te, tr, _ = env.step(a)
            R = R + r
            done = te or tr
        rets.append(R)
    return np.mean(rets, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="mo-halfcheetah-v5")
    p.add_argument("--c", type=float, required=True, help="weight on objective 1")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init_seed", type=int, default=None)
    p.add_argument("--out", default="geom/runs")
    args = p.parse_args()
    init_seed = args.seed if args.init_seed is None else args.init_seed

    env = make_env(args.env)
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]
    torch.manual_seed(init_seed)
    actor, critic, critic_t = Actor(od, ad), Critic(od, ad), Critic(od, ad)
    critic_t.load_state_dict(critic.state_dict())
    torch.manual_seed(args.seed + 1000)
    rng = np.random.default_rng(args.seed)
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    opt_al = torch.optim.Adam([log_alpha], lr=3e-4)
    w = torch.tensor([1.0, args.c])
    gamma, tau, batch, start = 0.99, 0.005, 256, 5000

    N = args.steps
    B = dict(o=np.zeros((N, od), np.float32), a=np.zeros((N, ad), np.float32),
             r=np.zeros((N, 2), np.float32), o2=np.zeros((N, od), np.float32), d=np.zeros(N, np.float32))
    o, _ = env.reset(seed=args.seed)
    t0, ep_ret, hist = time.time(), np.zeros(2), []
    for t in range(N):
        if t < start:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = actor.sample(torch.as_tensor(o, dtype=torch.float32)[None])[0][0].numpy()
        o2, r, te, tr, _ = env.step(a)
        B["o"][t], B["a"][t], B["r"][t], B["o2"][t], B["d"][t] = o, a, r, o2, float(te)
        ep_ret += r
        o = o2
        if te or tr:
            hist.append(ep_ret.copy())
            ep_ret[:] = 0
            o, _ = env.reset()
        if t < start:
            continue

        idx = rng.integers(0, t + 1, batch)
        ob, ab, rb, o2b, db = (torch.as_tensor(B[k][idx]) for k in ("o", "a", "r", "o2", "d"))
        rs = rb @ w
        alpha = log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2 = actor.sample(o2b)
            q1t, q2t = critic_t(o2b, a2)
            y = rs + gamma * (1 - db) * (torch.min(q1t, q2t) - alpha * lp2)
        q1, q2 = critic(ob, ab)
        lc = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        opt_c.zero_grad()
        lc.backward()
        opt_c.step()

        an, lp = actor.sample(ob)
        q1p, q2p = critic(ob, an)
        la = (alpha * lp - torch.min(q1p, q2p)).mean()
        opt_a.zero_grad()
        la.backward()
        opt_a.step()
        lal = -(log_alpha * (lp.detach() - ad).mean())          # target entropy = -|A|
        opt_al.zero_grad()
        lal.backward()
        opt_al.step()
        with torch.no_grad():
            for pp, tp in zip(critic.parameters(), critic_t.parameters()):
                tp.mul_(1 - tau).add_(tau * pp)

        if (t + 1) % 20_000 == 0:
            recent = np.mean(hist[-5:], 0) if hist else np.zeros(2)
            print(json.dumps(dict(t=t + 1, ret=recent.round(1).tolist(), scal=float(recent @ w.numpy()),
                                  alpha=float(alpha), sps=round((t + 1) / (time.time() - t0)))), flush=True)

    ev = evaluate(actor, args.env)
    tag = f"{args.env}_c{args.c}_s{args.seed}_i{init_seed}"
    os.makedirs(args.out, exist_ok=True)
    torch.save(dict(actor=actor.state_dict(), c=args.c, seed=args.seed, init_seed=init_seed, env=args.env,
                    eval=ev.tolist(), od=od, ad=ad), os.path.join(args.out, tag + ".pt"))
    print(json.dumps(dict(tag=tag, eval=ev.round(1).tolist(), scal=float(ev @ w.numpy()),
                          minutes=round((time.time() - t0) / 60, 1))))


if __name__ == "__main__":
    main()
