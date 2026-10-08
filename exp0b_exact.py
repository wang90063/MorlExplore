"""Stage 0b (actor only): train the actor against the exact critic of the ArcGoals task.

This removes critic learning and exploration, the continuous-action analogue of the
`sup` mode in exp0a. With deterministic dynamics the optimal scalarised value from
position p is (relaxing the step count to be continuous)

    V*(p, t) = max_k [ gamma^{n_k(p)} w(t) . g_k  -  COST (1 - gamma^{n_k(p)}) / (1 - gamma) ],
    n_k(p)   = max(0, |p - G_k| - GOAL_R) / STEP,

and Q*(p, a, t) = -COST + gamma V*(p + STEP a, t). The actor maximises Q* by
deterministic policy gradient on positions sampled uniformly in the disc.

Actors:
  mlp   : one deterministic MLP
  multi : H heads, pick the head with the largest Q*, train only the winner
  bc    : MLP regressed (MSE) onto the exact optimal direction, as a reference
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from exp0b import COST, GOAL_R, HORIZON, RADIUS, STEP, ArcGoals, mlp, project


class Task:
    def __init__(self, K, gamma):
        env = ArcGoals(K, 1, np.random.default_rng(0))
        self.G = torch.tensor(env.goals, dtype=torch.float32)       # [K, 2]
        self.g = torch.tensor(env.g, dtype=torch.float32)           # [K, 2]
        self.gamma = gamma
        self.env = env

    def per_goal_value(self, p, t):
        d = torch.linalg.norm(p[:, None] - self.G[None], dim=-1)    # [B, K]
        n = (d - GOAL_R).clamp(min=0) / STEP
        disc = self.gamma ** n
        w = torch.stack([1 - t, t], 1)                              # [B, 2]
        return disc * (w @ self.g.T) - COST * (1 - disc) / (1 - self.gamma)

    def Q(self, p, a, t):
        return -COST + self.gamma * self.per_goal_value(p + STEP * project(a), t).max(1).values

    def optimal_dir(self, p, t):
        k = self.per_goal_value(p, t).argmax(1)
        v = self.G[k] - p
        return v / (v.norm(dim=1, keepdim=True) + 1e-8)


def sample_disc(n):
    r = RADIUS * torch.sqrt(torch.rand(n))
    th = 2 * np.pi * torch.rand(n)
    return torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)


def rollout(task, act, ts):
    """Deterministic rollout from the origin; returns scalarised regret and reached goal."""
    n = len(ts)
    t = torch.tensor(ts, dtype=torch.float32)
    pos = torch.zeros(n, 2)
    ret = torch.zeros(n, 2)
    goal = torch.full((n,), -1)
    alive = torch.ones(n, dtype=torch.bool)
    for _ in range(HORIZON):
        with torch.no_grad():
            a = project(act(pos, t))
        pos = torch.where(alive[:, None], pos + STEP * a, pos)
        d = torch.linalg.norm(pos[:, None] - task.G[None], dim=-1)
        hit = alive & (d.min(1).values < GOAL_R)
        k = d.argmin(1)
        ret[alive] -= COST
        ret[hit] += task.g[k[hit]]
        goal[hit] = k[hit]
        alive &= ~hit
    w = np.stack([1 - ts, ts], 1)
    scal = (ret.numpy() * w).sum(1)
    opt = np.array([task.env.optimal_return(x) for x in ts])
    return opt - scal, goal.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--actor", choices=["mlp", "multi", "bc"], default="mlp")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs0b_exact")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    task = Task(args.K, args.gamma)
    H = args.heads if args.actor == "multi" else 1
    net = mlp(3, 2 * H, args.width, depth=args.depth, out_act=nn.Tanh())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def heads(pos, t):
        return net(torch.cat([pos, t[:, None]], 1)).view(-1, H, 2)

    def act(pos, t):
        A = heads(pos, t)
        if H == 1:
            return A[:, 0]
        B = len(pos)
        q = task.Q(pos.repeat_interleave(H, 0), A.reshape(B * H, 2), t.repeat_interleave(H)).view(B, H)
        return A[torch.arange(B), q.argmax(1)]

    for step in range(args.steps):
        pos, t = sample_disc(512), torch.rand(512)
        A = heads(pos, t)
        if args.actor == "bc":
            loss = ((A[:, 0] - task.optimal_dir(pos, t)) ** 2).sum(1).mean()
        else:
            B = len(pos)
            q = task.Q(pos.repeat_interleave(H, 0), A.reshape(B * H, 2), t.repeat_interleave(H)).view(B, H)
            if H > 1:
                loss = -q[torch.arange(B), q.detach().argmax(1)].mean()
            else:
                loss = -q[:, 0].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    ts = np.linspace(0, 1, 2001)
    regret, goal = rollout(task, act, ts)
    corners = task.env.corners()
    d = np.abs(ts[:, None] - corners[None]).min(1)
    # optimal goal for each t, to separate "missed everything" from "picked a wrong goal"
    opt_goal = np.array([(task.env.g @ np.array([1 - x, x])).argmax() for x in ts])
    wrong = (goal != opt_goal) & (goal >= 0)
    tag = f"{args.actor}_K{args.K}_w{args.width}_d{args.depth}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, tag + ".npz"), ts=ts, regret=regret, goal=goal,
                        opt_goal=opt_goal, corners=corners)
    miss = goal < 0
    band = miss.sum() / len(ts)                     # fraction of preference space with a miss
    print(json.dumps(dict(tag=tag, regret=float(regret.mean()), miss_frac=float(band),
                          wrong_frac=float(wrong.mean()),
                          miss_near=float(miss[d < 0.05].mean()),
                          miss_far=float(miss[d >= 0.05].mean()),
                          regret_near=float(regret[d < 0.05].mean()),
                          regret_far=float(regret[d >= 0.05].mean()))))


if __name__ == "__main__":
    main()
