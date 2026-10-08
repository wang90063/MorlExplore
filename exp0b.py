"""Stage 0b: does a preference-conditioned continuous actor "average modes" at corners?

Environment: a 2D point mass starts at the origin. K goals sit on an upper arc of
radius 1.5; reaching goal k ends the episode with vector reward g_k = (cos a_k, sin a_k),
and every step costs 0.01 on both objectives. Under w(t) = (1 - t, t) the optimal policy
is "go to argmax_k w . g_k", so there are exactly K - 1 corner weights at known places.
An actor that interpolates between the two neighbouring modes heads between two goals
and reaches neither.

Agent: preference-conditioned TD3 with a scalar critic Q(s, a, t); vector rewards are
stored so every sampled transition is relabelled with a fresh t. Actors:
  mlp   : one deterministic MLP actor
  multi : H heads; acting picks the head with the largest Q, and only that head is
          trained (winner-take-all), so heads can specialise to different modes
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

STEP, RADIUS, GOAL_R, HORIZON, COST = 0.1, 1.5, 0.15, 40, 0.01


def project(a):
    """Clip actions to the unit disc so every direction has the same top speed."""
    lib = torch if isinstance(a, torch.Tensor) else np
    n = lib.linalg.norm(a, axis=-1) if lib is np else a.norm(dim=-1)
    scale = 1.0 / lib.maximum(n, lib.ones_like(n))
    return a * scale[..., None]


class ArcGoals:
    def __init__(self, K, n_env, rng, random_start=False):
        self.K, self.n, self.rng, self.random_start = K, n_env, rng, random_start
        ang = np.linspace(np.pi * 0.85, np.pi * 0.15, K)           # left to right on the arc
        self.goals = RADIUS * np.stack([np.cos(ang), np.sin(ang)], 1)
        th = np.linspace(0, np.pi / 2, K)                          # objective-space angle
        self.g = np.stack([np.cos(th), np.sin(th)], 1)             # goal k favours obj 0 -> obj 1
        self.reset_all()

    def reset_all(self):
        self.pos = np.zeros((self.n, 2))
        self.t = np.zeros(self.n, dtype=int)
        self.reset(np.ones(self.n, dtype=bool))

    def reset(self, mask):
        """Reset the masked envs; random starts (uniform in the disc) fix exploration."""
        n = int(mask.sum())
        if self.random_start:
            r = RADIUS * np.sqrt(self.rng.uniform(0, 1, n))
            th = self.rng.uniform(0, 2 * np.pi, n)
            self.pos[mask] = np.stack([r * np.cos(th), r * np.sin(th)], 1)
        else:
            self.pos[mask] = 0.0
        self.t[mask] = 0

    def obs(self):
        return np.concatenate([self.pos, (self.t / HORIZON)[:, None]], 1)

    def step(self, a):
        self.pos = self.pos + STEP * project(np.asarray(a))
        self.t += 1
        d = np.linalg.norm(self.pos[:, None] - self.goals[None], axis=-1)   # [n, K]
        hit = d.min(1) < GOAL_R
        k = d.argmin(1)
        r = np.full((self.n, 2), -COST)
        r[hit] += self.g[k[hit]]
        done = hit | (self.t >= HORIZON)
        terminal = hit                                             # time-outs are not terminal
        return r, done, terminal, np.where(hit, k, -1)

    def corners(self):
        """t where neighbouring goals tie: (1-t) g_k0 + t g_k1 = (1-t) g_j0 + t g_j1."""
        out = []
        for k in range(self.K - 1):
            a, b = self.g[k], self.g[k + 1]
            out.append((b[0] - a[0]) / ((b[0] - a[0]) - (b[1] - a[1])))
        return np.array(out)

    def optimal_return(self, t):
        steps = int(np.ceil((RADIUS - GOAL_R) / STEP))
        w = np.array([1 - t, t])
        return (self.g @ w).max() - COST * steps * w.sum()


def mlp(i, o, width, depth=2, out_act=None):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, o))
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, heads, width):
        super().__init__()
        self.heads = heads
        self.net = mlp(obs_dim + 1, 2 * heads, width, out_act=nn.Tanh())

    def all_heads(self, o, t):
        return self.net(torch.cat([o, t[:, None]], 1)).view(-1, self.heads, 2)


class Critic(nn.Module):
    def __init__(self, obs_dim, width):
        super().__init__()
        self.q1 = mlp(obs_dim + 1 + 2, 1, width)
        self.q2 = mlp(obs_dim + 1 + 2, 1, width)

    def forward(self, o, t, a):
        x = torch.cat([o, t[:, None], a], 1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def select(actor, critic, o, t):
    """Per-sample action from the head with the largest Q1; also returns head index."""
    A = actor.all_heads(o, t)                                      # [B, H, 2]
    if actor.heads == 1:
        return A[:, 0], torch.zeros(len(o), dtype=torch.long)
    B, H, _ = A.shape
    q, _ = critic(o.repeat_interleave(H, 0), t.repeat_interleave(H), A.reshape(B * H, 2))
    idx = q.view(B, H).argmax(1)
    return A[torch.arange(B), idx], idx


def evaluate(env_cls, K, actor, critic, ts):
    env = env_cls(K, len(ts), np.random.default_rng(0))
    tt = torch.tensor(ts, dtype=torch.float32)
    ret = np.zeros((len(ts), 2))
    goal = np.full(len(ts), -1)
    alive = np.ones(len(ts), dtype=bool)
    for _ in range(HORIZON):
        with torch.no_grad():
            a, _ = select(actor, critic, torch.tensor(env.obs(), dtype=torch.float32), tt)
        r, done, _, k = env.step(a.numpy())
        ret[alive] += r[alive]
        goal[alive & (k >= 0)] = k[alive & (k >= 0)]
        alive &= ~done
        if not alive.any():
            break
    w = np.stack([1 - ts, ts], 1)
    scal = (ret * w).sum(1)
    opt = np.array([env.optimal_return(t) for t in ts])
    return scal, opt - scal, goal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--actor", choices=["mlp", "multi"], default="mlp")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--steps", type=int, default=60000, help="gradient updates")
    p.add_argument("--n_env", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--random_start", action="store_true", help="train from random starts")
    p.add_argument("--out", default="runs0b")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = ArcGoals(args.K, args.n_env, rng, random_start=args.random_start)
    obs_dim = 3
    H = 1 if args.actor == "mlp" else args.heads
    actor, critic = Actor(obs_dim, H, args.width), Critic(obs_dim, args.width)
    actor_t, critic_t = Actor(obs_dim, H, args.width), Critic(obs_dim, args.width)
    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)

    cap = 500_000
    buf = dict(o=np.zeros((cap, obs_dim), np.float32), a=np.zeros((cap, 2), np.float32),
               r=np.zeros((cap, 2), np.float32), o2=np.zeros((cap, obs_dim), np.float32),
               term=np.zeros(cap, np.float32))
    n_buf, ptr = 0, 0
    t_env = rng.uniform(0, 1, args.n_env)
    warmup, tau, noise, tnoise = 2000, 0.005, 0.2, 0.2

    def push(o, a, r, o2, term):
        nonlocal n_buf, ptr
        for i in range(len(o)):
            for k, v in zip(("o", "a", "r", "o2", "term"), (o[i], a[i], r[i], o2[i], term[i])):
                buf[k][ptr] = v
            ptr = (ptr + 1) % cap
            n_buf = min(n_buf + 1, cap)

    log = []
    for step in range(args.steps):
        o = env.obs()
        if n_buf < warmup:
            a = rng.uniform(-1, 1, (args.n_env, 2))
        else:
            with torch.no_grad():
                a, _ = select(actor, critic, torch.tensor(o, dtype=torch.float32),
                              torch.tensor(t_env, dtype=torch.float32))
            a = np.clip(a.numpy() + noise * rng.standard_normal(a.shape), -1, 1)
        r, done, term, _ = env.step(a)
        push(o, a, r, env.obs(), term.astype(np.float32))
        if done.any():
            env.reset(done)
            t_env[done] = rng.uniform(0, 1, done.sum())
        if n_buf < warmup:
            continue

        idx = rng.integers(0, n_buf, 256)
        ob, ab, rb, o2b, tb = (torch.tensor(buf[k][idx]) for k in ("o", "a", "r", "o2", "term"))
        t = torch.rand(256)                                        # preference relabelling
        rs = (1 - t) * rb[:, 0] + t * rb[:, 1]
        with torch.no_grad():
            a2, _ = select(actor_t, critic_t, o2b, t)
            a2 = (a2 + (tnoise * torch.randn_like(a2)).clamp(-0.5, 0.5)).clamp(-1, 1)
            q1t, q2t = critic_t(o2b, t, a2)
            y = rs + args.gamma * (1 - tb) * torch.min(q1t, q2t)
        q1, q2 = critic(ob, t, ab)
        lc = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        opt_c.zero_grad()
        lc.backward()
        opt_c.step()

        if step % 2 == 0:
            A = actor.all_heads(ob, t)                             # [B, H, 2]
            B = len(ob)
            q, _ = critic(ob.repeat_interleave(H, 0), t.repeat_interleave(H), A.reshape(B * H, 2))
            q = q.view(B, H)
            if H > 1:                                              # winner-take-all
                win = q.detach().argmax(1)
                la = -q[torch.arange(B), win].mean()
            else:
                la = -q.mean()
            opt_a.zero_grad()
            la.backward()
            opt_a.step()
            with torch.no_grad():
                for net, tgt in ((actor, actor_t), (critic, critic_t)):
                    for pp, tp in zip(net.parameters(), tgt.parameters()):
                        tp.mul_(1 - tau).add_(tau * pp)

        if (step + 1) % 10000 == 0:
            ts = np.linspace(0, 1, 101)
            _, reg, goal = evaluate(ArcGoals, args.K, actor, critic, ts)
            log.append(dict(step=step + 1, regret=float(reg.mean()), miss=float((goal < 0).mean())))
            print(json.dumps(log[-1]), flush=True)

    ts = np.linspace(0, 1, 1001)
    scal, reg, goal = evaluate(ArcGoals, args.K, actor, critic, ts)
    corners = ArcGoals(args.K, 1, rng).corners()
    tag = f"{args.actor}_K{args.K}_w{args.width}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, tag + ".npz"), ts=ts, regret=reg, goal=goal,
                        corners=corners, log=json.dumps(log))
    d = np.abs(ts[:, None] - corners[None]).min(1) if len(corners) else np.full(len(ts), np.inf)
    print(json.dumps(dict(tag=tag, regret=float(reg.mean()), miss=float((goal < 0).mean()),
                          miss_near=float((goal[d < 0.05] < 0).mean()),
                          miss_far=float((goal[d >= 0.05] < 0).mean()))))


if __name__ == "__main__":
    main()
