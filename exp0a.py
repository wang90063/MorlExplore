"""Stage 0a: does preference-conditioned Q-learning fail near policy switch points?

Two training modes on a tabular two-objective MDP with exact solutions:
  sup : regress Q_theta(s, ., t) onto the exact Q*(s, ., t)  (pure approximation)
  fqi : fitted Q iteration with expected backups and a target network
        (adds bootstrapping, no sampling noise)

Everything is measured against the distance from t to the nearest switch point:
  - |Q_theta - Q*| and greedy-action mismatch vs. distance to the state's own switch points
  - scalarised regret of the greedy policy vs. distance to the nearest corner weight
  - cosine between per-preference gradients, split by whether the two preferences
    share the same optimal policy (controlling for |t_i - t_j|)
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from morl_kink.mdp import (corner_weights, dist_to_nearest, exact_tables,
                           local_switch_points, policy_eval_vec, random_momdp)


class QNet(nn.Module):
    def __init__(self, S, A, width, depth):
        super().__init__()
        layers, d = [], S + 1
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(nn.Linear(d, A))
        self.net = nn.Sequential(*layers)
        self.S = S

    def forward(self, s_onehot, t):
        return self.net(torch.cat([s_onehot, 2.0 * t[:, None] - 1.0], dim=1))

    def all_states(self, t):
        """Q for every state at each preference in t: [len(t), S, A]."""
        T = t.shape[0]
        eye = torch.eye(self.S).repeat(T, 1)
        tt = t.repeat_interleave(self.S)
        return self(eye, tt).view(T, self.S, -1)


def td_targets(mdp_t, tgt, t):
    """Expected one-step backup r_t + gamma * P @ max_a' Q_tgt(s', a', t): [len(t), S, A]."""
    P, R, gamma = mdp_t
    with torch.no_grad():
        v = tgt.all_states(t).max(-1).values                    # [T, S]
        r = (1 - t)[:, None, None] * R[..., 0] + t[:, None, None] * R[..., 1]
        return r + gamma * torch.einsum("sau,tu->tsa", P, v)


def per_pref_loss(net, t, mode, Qstar_t, mdp_t, tgt):
    q = net.all_states(t)
    target = Qstar_t if mode == "sup" else td_targets(mdp_t, tgt, t)
    return ((q - target) ** 2).mean(dim=(1, 2))                # [T]


def grad_cosines(net, ts_idx, ts, mode, Qstar, mdp_t, tgt):
    grads = []
    for i in ts_idx:
        t = ts[i:i + 1]
        loss = per_pref_loss(net, t, mode, Qstar[i:i + 1], mdp_t, tgt).sum()
        g = torch.autograd.grad(loss, list(net.parameters()))
        grads.append(torch.cat([x.flatten() for x in g]))
    G = torch.stack(grads)
    G = G / (G.norm(dim=1, keepdim=True) + 1e-12)
    return (G @ G.T).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--S", type=int, default=30)
    p.add_argument("--A", type=int, default=4)
    p.add_argument("--corr", type=float, default=0.0)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--mode", choices=["sup", "fqi"], default="sup")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_t", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target_every", type=int, default=250)
    p.add_argument("--grid", type=int, default=2001)
    p.add_argument("--n_cos", type=int, default=80)
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mdp = random_momdp(S=args.S, A=args.A, corr=args.corr, seed=args.seed)
    ts_np = np.linspace(0, 1, args.grid)
    Qstar_np, pistar, Vstar = exact_tables(mdp, ts_np)
    corners = corner_weights(mdp)
    switches = local_switch_points(ts_np, pistar)

    ts = torch.tensor(ts_np, dtype=torch.float32)
    Qstar = torch.tensor(Qstar_np, dtype=torch.float32)
    mdp_t = (torch.tensor(mdp.P, dtype=torch.float32),
             torch.tensor(mdp.R, dtype=torch.float32), mdp.gamma)

    net = QNet(args.S, args.A, args.width, args.depth)
    tgt = QNet(args.S, args.A, args.width, args.depth)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    cos_idx = np.sort(np.random.choice(args.grid, args.n_cos, replace=False))
    cos_snapshots = {}
    snap_steps = {int(args.steps * f) for f in (0.0, 0.1, 0.5, 1.0)}

    for step in range(args.steps + 1):
        if step in snap_steps:
            cos_snapshots[step] = grad_cosines(net, cos_idx, ts, args.mode, Qstar, mdp_t, tgt)
        if step == args.steps:
            break
        idx = torch.randint(0, args.grid, (args.batch_t,))
        loss = per_pref_loss(net, ts[idx], args.mode, Qstar[idx], mdp_t, tgt).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if args.mode == "fqi" and step % args.target_every == 0:
            tgt.load_state_dict(net.state_dict())

    # ---- evaluation on the full grid ----
    with torch.no_grad():
        Qhat = net.all_states(ts).numpy()                      # [T, S, A]
    abs_err = np.abs(Qhat - Qstar_np).mean(-1)                  # [T, S]
    pihat = Qhat.argmax(-1)
    mismatch = (pihat != pistar).astype(float)                  # [T, S]
    w = np.stack([1 - ts_np, ts_np], 1)
    regret = np.empty(args.grid)
    for i in range(args.grid):
        V, _ = policy_eval_vec(mdp, pihat[i])
        regret[i] = w[i] @ Vstar[i] - w[i] @ (mdp.mu @ V)
    d_local = np.stack([dist_to_nearest(ts_np, sw) for sw in switches], 1)   # [T, S]
    d_corner = dist_to_nearest(ts_np, corners)
    hamming = (pistar[cos_idx][:, None, :] != pistar[cos_idx][None, :, :]).sum(-1)

    tag = f"{args.mode}_S{args.S}_c{args.corr}_w{args.width}_d{args.depth}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out, tag + ".npz"),
        ts=ts_np, corners=corners, abs_err=abs_err, mismatch=mismatch, regret=regret,
        d_local=d_local, d_corner=d_corner, cos_idx=cos_idx, hamming=hamming,
        n_switch=np.array([len(s) for s in switches]),
        Vstar=Vstar, Qstar_scale=np.abs(Qstar_np).mean(),
        **{f"cos_{k}": v for k, v in cos_snapshots.items()})
    summary = dict(tag=tag, n_corners=len(corners),
                   n_local_switches=int(sum(len(s) for s in switches)),
                   mean_abs_err=float(abs_err.mean()), mismatch=float(mismatch.mean()),
                   mean_regret=float(regret.mean()), Qstar_scale=float(np.abs(Qstar_np).mean()))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
