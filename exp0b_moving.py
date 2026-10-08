"""Stage 0b control: same task difficulty as exp0b, but no corners.

There is a single goal whose position slides continuously along the same arc as the
preference changes: t -> angle pi(0.85 - 0.7 t), reward g(t) = (cos(pi t / 2), sin(pi t / 2)).
So omega -> pi* is continuous, while the target is just as narrow (GOAL_R) and the reward
just as sparse as in the K-goal task. If the multi-head actor still beats the MLP here, its
advantage in exp0b is an optimisation effect, not removal of mode averaging.

Relabelling: reward and termination depend on t, so both are recomputed from the stored
next position for every relabelled t (hindsight-style).
"""
import argparse
import json
import os

import numpy as np
import torch

from exp0b import COST, GOAL_R, HORIZON, RADIUS, STEP, Actor, Critic, project, select


def goal_pos(t):
    lib = torch if isinstance(t, torch.Tensor) else np
    ang = np.pi * (0.85 - 0.7 * t)
    return RADIUS * lib.stack([lib.cos(ang), lib.sin(ang)], -1)


def goal_vec(t):
    lib = torch if isinstance(t, torch.Tensor) else np
    th = np.pi / 2 * t
    return lib.stack([lib.cos(th), lib.sin(th)], -1)


class MovingGoal:
    def __init__(self, n_env, rng, random_start=False):
        self.n, self.rng, self.random_start = n_env, rng, random_start
        self.pref = np.zeros(n_env)
        self.pos = np.zeros((n_env, 2))
        self.t = np.zeros(n_env, dtype=int)

    def reset(self, mask, pref):
        n = int(mask.sum())
        if self.random_start:
            r = RADIUS * np.sqrt(self.rng.uniform(0, 1, n))
            th = self.rng.uniform(0, 2 * np.pi, n)
            self.pos[mask] = np.stack([r * np.cos(th), r * np.sin(th)], 1)
        else:
            self.pos[mask] = 0.0
        self.t[mask] = 0
        self.pref[mask] = pref

    def obs(self):
        return np.concatenate([self.pos, (self.t / HORIZON)[:, None]], 1)

    def step(self, a):
        self.pos = self.pos + STEP * project(np.asarray(a))
        self.t += 1
        hit = np.linalg.norm(self.pos - goal_pos(self.pref), axis=-1) < GOAL_R
        r = np.full((self.n, 2), -COST)
        r[hit] += goal_vec(self.pref[hit])
        done = hit | (self.t >= HORIZON)
        return r, done, hit


def optimal_return(t):
    steps = int(np.ceil((RADIUS - GOAL_R) / STEP))
    return float(goal_vec(np.array(t)) @ np.array([1 - t, t])) - COST * steps


def evaluate(actor, critic, ts):
    env = MovingGoal(len(ts), np.random.default_rng(0))
    env.reset(np.ones(len(ts), dtype=bool), ts)
    tt = torch.tensor(ts, dtype=torch.float32)
    ret = np.zeros((len(ts), 2))
    reached = np.zeros(len(ts), dtype=bool)
    alive = np.ones(len(ts), dtype=bool)
    for _ in range(HORIZON):
        with torch.no_grad():
            a, _ = select(actor, critic, torch.tensor(env.obs(), dtype=torch.float32), tt)
        r, done, hit = env.step(a.numpy())
        ret[alive] += r[alive]
        reached |= alive & hit
        alive &= ~done
        if not alive.any():
            break
    scal = (ret * np.stack([1 - ts, ts], 1)).sum(1)
    opt = np.array([optimal_return(t) for t in ts])
    return opt - scal, reached


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", choices=["mlp", "multi"], default="mlp")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--n_env", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--random_start", action="store_true")
    p.add_argument("--out", default="runs0b_moving")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = MovingGoal(args.n_env, rng, random_start=args.random_start)
    env.reset(np.ones(args.n_env, dtype=bool), rng.uniform(0, 1, args.n_env))
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
               o2=np.zeros((cap, obs_dim), np.float32))
    n_buf, ptr = 0, 0
    warmup, tau, noise, tnoise = 2000, 0.005, 0.2, 0.2

    log = []
    for step in range(args.steps):
        o = env.obs()
        if n_buf < warmup:
            a = rng.uniform(-1, 1, (args.n_env, 2))
        else:
            with torch.no_grad():
                a, _ = select(actor, critic, torch.tensor(o, dtype=torch.float32),
                              torch.tensor(env.pref, dtype=torch.float32))
            a = np.clip(a.numpy() + noise * rng.standard_normal(a.shape), -1, 1)
        _, done, _ = env.step(a)
        o2 = env.obs()
        for i in range(args.n_env):
            buf["o"][ptr], buf["a"][ptr], buf["o2"][ptr] = o[i], a[i], o2[i]
            ptr = (ptr + 1) % cap
            n_buf = min(n_buf + 1, cap)
        if done.any():
            env.reset(done, rng.uniform(0, 1, done.sum()))
        if n_buf < warmup:
            continue

        idx = rng.integers(0, n_buf, 256)
        ob, ab, o2b = (torch.tensor(buf[k][idx]) for k in ("o", "a", "o2"))
        t = torch.rand(256)                                        # preference relabelling
        hit = (torch.linalg.norm(o2b[:, :2] - goal_pos(t), dim=-1) < GOAL_R).float()
        w = torch.stack([1 - t, t], 1)
        rs = -COST + hit * (w * goal_vec(t)).sum(1)
        with torch.no_grad():
            a2, _ = select(actor_t, critic_t, o2b, t)
            a2 = (a2 + (tnoise * torch.randn_like(a2)).clamp(-0.5, 0.5)).clamp(-1, 1)
            q1t, q2t = critic_t(o2b, t, a2)
            y = rs + args.gamma * (1 - hit) * torch.min(q1t, q2t)
        q1, q2 = critic(ob, t, ab)
        lc = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        opt_c.zero_grad()
        lc.backward()
        opt_c.step()

        if step % 2 == 0:
            A = actor.all_heads(ob, t)
            B = len(ob)
            q, _ = critic(ob.repeat_interleave(H, 0), t.repeat_interleave(H), A.reshape(B * H, 2))
            q = q.view(B, H)
            la = -q[torch.arange(B), q.detach().argmax(1)].mean() if H > 1 else -q.mean()
            opt_a.zero_grad()
            la.backward()
            opt_a.step()
            with torch.no_grad():
                for net, tgt in ((actor, actor_t), (critic, critic_t)):
                    for pp, tp in zip(net.parameters(), tgt.parameters()):
                        tp.mul_(1 - tau).add_(tau * pp)

        if (step + 1) % 10000 == 0:
            reg, reached = evaluate(actor, critic, np.linspace(0, 1, 101))
            log.append(dict(step=step + 1, regret=float(reg.mean()), miss=float((~reached).mean())))
            print(json.dumps(log[-1]), flush=True)

    ts = np.linspace(0, 1, 1001)
    reg, reached = evaluate(actor, critic, ts)
    tag = f"moving_{args.actor}_w{args.width}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, tag + ".npz"), ts=ts, regret=reg, reached=reached,
                        log=json.dumps(log))
    print(json.dumps(dict(tag=tag, regret=float(reg.mean()), miss=float((~reached).mean()),
                          regret_hit=float(reg[reached].mean()) if reached.any() else None)))


if __name__ == "__main__":
    main()
