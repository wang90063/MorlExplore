"""Preference-conditioned SAC (CAPQL-style) and two single-change variants, on COLA's own MO-MuJoCo envs.

Arms (P2 kill test):
  base     preference-conditioned SAC with vector critic Q(s,a,w) in R^k and entropy-augmented vector target,
           as CAPQL (Lu, Herman & Yu, ICLR 2023), which is also Momba's backbone. --n_env_w > 1 turns on an
           Envelope-style max over candidate preferences in the target (diverged on Ant, see --n_env_w).
  momba    base + Momba-style network changes (arXiv 2608.07180, items i-ii): running observation
           normalisation, LayerNorm on hidden features, weight normalisation. Item iii (distributional
           returns) is not included.
  perobj   base, but each objective gets its own critic network Q_i(s,a,w) (separate trunks)

Same envs, eval protocol and hyper-parameters as COLA's configs: batch 256, hidden [256,256], lr 3e-4,
gamma 0.99, tau 0.005, auto entropy, start_steps 10000, 1 update per env step, eval every 50k steps on
the 201 preferences of generate_w_batch_test(2, 0.005). The env reset has no noise and the eval policy is
deterministic, so one eval episode per preference equals COLA's 3.

Usage: OMP_NUM_THREADS=1 python p2/esac.py --arm base --env MO-Hopper-v2 --seed 1 --out runs/p2/... \
       [--steps 1000000] [--eval_interval 50000]
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "third_party", "cola_compat"), os.path.join(ROOT, "third_party", "COLA")]
import gym  # noqa: E402  (cola_compat shim)
import environments  # noqa: E402,F401  (registers COLA's envs)

torch.set_num_threads(1)


def eval_prefs(k=2, step=0.005):
    # identical to COLA's generate_w_batch_test, so objs_*.npy rows line up across arms
    mesh = [np.arange(0, 1 + step, step)] * k
    w = np.array(list(itertools.product(*mesh)))
    return np.unique(w[w.sum(axis=1) == 1, :], axis=0)


class RunningNorm(nn.Module):
    def __init__(self, dim, clip=10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    def update(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, self.mean.numel())
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d, tot = bm - self.mean, self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def mlp(inp, out, hidden=(256, 256), momba=False):
    layers, d = [], inp
    for h in hidden:
        lin = nn.Linear(d, h)
        if momba:
            lin = nn.utils.parametrizations.weight_norm(lin)
            layers += [lin, nn.LayerNorm(h), nn.ReLU()]
        else:
            layers += [lin, nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class Critic(nn.Module):
    """Twin vector critic: returns (q1, q2), each [B, k]."""

    def __init__(self, sd, ad, k, arm):
        super().__init__()
        m = arm == "momba"
        if arm == "perobj":
            self.q1 = nn.ModuleList([mlp(sd + ad + k, 1) for _ in range(k)])
            self.q2 = nn.ModuleList([mlp(sd + ad + k, 1) for _ in range(k)])
        else:
            self.q1, self.q2 = mlp(sd + ad + k, k, momba=m), mlp(sd + ad + k, k, momba=m)
        self.perobj = arm == "perobj"

    def forward(self, s, a, w):
        x = torch.cat([s, a, w], -1)
        if self.perobj:
            return torch.cat([q(x) for q in self.q1], -1), torch.cat([q(x) for q in self.q2], -1)
        return self.q1(x), self.q2(x)


class Actor(nn.Module):
    def __init__(self, sd, ad, k, arm):
        super().__init__()
        self.net = mlp(sd + k, 2 * ad, momba=arm == "momba")

    def forward(self, s, w, deterministic=False):
        mu, log_std = self.net(torch.cat([s, w], -1)).chunk(2, -1)
        log_std = log_std.clamp(-20, 2)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp


def pick_twin(q1, q2, w):
    """Per sample, the twin with the lower scalarised value (clipped double-Q on w·Q)."""
    use1 = ((q1 * w).sum(-1, keepdim=True) <= (q2 * w).sum(-1, keepdim=True)).float()
    return use1 * q1 + (1 - use1) * q2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["base", "momba", "perobj"], required=True)
    p.add_argument("--env", default="MO-Hopper-v2")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--eval_interval", type=int, default=50_000)
    p.add_argument("--start_steps", type=int, default=10_000)
    # 1 = CAPQL-style target (same w only). 4 (envelope max) diverged on Ant at ~240k steps in both seeds
    # (critic loss 1e5, runs/p2/kill_envelope_diverged/): max over w' overestimates, and Ant never terminates.
    p.add_argument("--n_env_w", type=int, default=1, help="preferences in the envelope max (incl. w)")
    p.add_argument("--exit_after", type=int, default=0, help="testing: exit right after the first checkpoint "
                   "at or past this step, to simulate an interruption")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    log_path = os.path.join(args.out, "log.jsonl")
    ck = None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        same = {kk: v for kk, v in ck["args"].items() if kk != "exit_after"}
        assert same == {kk: v for kk, v in vars(args).items() if kk != "exit_after"}, "checkpoint args differ"
        with open(log_path, "r+") as fh:       # drop log lines written after the checkpoint
            fh.truncate(ck["log_pos"])
    else:
        json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
        if os.path.exists(log_path):
            os.remove(log_path)                 # a run without checkpoint starts from step 0
    log = open(log_path, "a", buffering=1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    k, sd, ad = env.reward_num, env.observation_space.shape[0], env.action_space.shape[0]
    max_a = env.action_space.high
    T = env.max_episode_steps

    actor, critic = Actor(sd, ad, k, args.arm), Critic(sd, ad, k, args.arm)
    critic_t = Critic(sd, ad, k, args.arm)
    critic_t.load_state_dict(critic.state_dict())
    for q in critic_t.parameters():
        q.requires_grad_(False)
    norm = RunningNorm(sd) if args.arm == "momba" else None
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_a = torch.optim.Adam(actor.parameters(), 3e-4)
    opt_q = torch.optim.Adam(critic.parameters(), 3e-4)
    opt_al = torch.optim.Adam([log_alpha], 3e-4)
    target_ent = -float(ad)
    B, gamma, tau = 256, 0.99, 0.005

    N = args.steps + 1
    S = np.zeros((N, sd), np.float32); S2 = np.zeros((N, sd), np.float32)
    A = np.zeros((N, ad), np.float32); R = np.zeros((N, k), np.float32); D = np.zeros((N, 1), np.float32)
    nbuf = 0

    def nz(x):
        return norm(x) if norm is not None else x

    def act(s, w, deterministic):
        with torch.no_grad():
            st = nz(torch.as_tensor(s, dtype=torch.float32)[None])
            a, _ = actor(st, torch.as_tensor(w, dtype=torch.float32)[None], deterministic)
        return a[0].numpy()

    def update():
        idx = rng.randint(0, nbuf, B)
        s, a, r = torch.from_numpy(S[idx]), torch.from_numpy(A[idx]), torch.from_numpy(R[idx])
        s2, d = torch.from_numpy(S2[idx]), torch.from_numpy(D[idx])
        s, s2 = nz(s), nz(s2)
        w = torch.from_numpy(rng.dirichlet(np.ones(k), B).astype(np.float32))
        alpha = log_alpha.exp().detach()
        with torch.no_grad():
            # envelope target: for each sample, try n preferences w' (w itself + random), keep the
            # vector return whose w-scalarised soft value is highest
            n = args.n_env_w
            wp = torch.from_numpy(rng.dirichlet(np.ones(k), (n - 1) * B).astype(np.float32))
            wp = torch.cat([w, wp], 0)                      # [n*B, k], block j = j-th candidate
            s2n = s2.repeat(n, 1)
            a2, logp2 = actor(s2n, wp)
            q1t, q2t = critic_t(s2n, a2, wp)
            wn = w.repeat(n, 1)
            qv = pick_twin(q1t, q2t, wn) - (alpha * logp2)[:, None]   # soft vector value, [n*B, k]
            score = (qv * wn).sum(-1).view(n, B)
            best = score.argmax(0)                          # [B]
            qv = qv.view(n, B, k)[best, torch.arange(B)]
            y = r + gamma * (1 - d) * qv
        q1, q2 = critic(s, a, w)
        lq = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        opt_q.zero_grad(); lq.backward(); opt_q.step()

        an, logp = actor(s, w)
        q1n, q2n = critic(s, an, w)
        qs = torch.min((q1n * w).sum(-1), (q2n * w).sum(-1))
        la = (alpha * logp - qs).mean()
        opt_a.zero_grad(); la.backward(); opt_a.step()

        lal = -(log_alpha * (logp.detach() + target_ent)).mean()
        opt_al.zero_grad(); lal.backward(); opt_al.step()

        with torch.no_grad():
            for pt, pp in zip(critic_t.parameters(), critic.parameters()):
                pt.mul_(1 - tau).add_(pp, alpha=tau)
        return lq.item(), la.item(), alpha.item()

    prefs = eval_prefs(k)

    def evaluate(step):
        t0 = time.time()
        objs = np.zeros((len(prefs), k))
        for i, w in enumerate(prefs):
            s, done, ret = env.reset(), False, np.zeros(k)
            while not done:
                s, r, done, _ = env.step(act(s, w, True) * max_a)
                ret += r
            objs[i] = ret
        np.save(os.path.join(args.out, f"objs_{step}.npy"), objs)
        log.write(json.dumps({"eval_step": step, "eval_sec": round(time.time() - t0, 1),
                              "mean_obj": objs.mean(0).round(1).tolist()}) + "\n")

    step, t_start, next_eval, stats = 0, time.time(), args.eval_interval, None
    modules = {"actor": actor, "critic": critic, "critic_t": critic_t, "opt_a": opt_a, "opt_q": opt_q,
               "opt_al": opt_al}
    if norm is not None:
        modules["norm"] = norm

    def save_ckpt():
        # called only between episodes, right after an eval; env.reset() has no noise, so no sim state is needed
        st = {kk: m.state_dict() for kk, m in modules.items()}
        st.update(args=vars(args), log_alpha=log_alpha.detach().clone(), step=step, next_eval=next_eval,
                  stats=stats, nbuf=nbuf, wall=time.time() - t_start, log_pos=log.tell(),
                  buf={"S": S[:nbuf], "S2": S2[:nbuf], "A": A[:nbuf], "R": R[:nbuf], "D": D[:nbuf]},
                  rng=rng.get_state(), np_global=np.random.get_state(), torch_rng=torch.get_rng_state(),
                  env_rng=env.np_random.get_state(), act_rng=env.action_space.np_random.bit_generator.state)
        tmp = ckpt_path + ".tmp"
        torch.save(st, tmp)
        os.replace(tmp, ckpt_path)              # atomic: a crash mid-save leaves the previous checkpoint

    if ck is not None:
        for kk, m in modules.items():
            m.load_state_dict(ck[kk])
        with torch.no_grad():
            log_alpha.copy_(ck["log_alpha"])
        step, next_eval, stats, nbuf = ck["step"], ck["next_eval"], ck["stats"], ck["nbuf"]
        t_start = time.time() - ck["wall"]
        for kk, arr in zip("S S2 A R D".split(), (S, S2, A, R, D)):
            arr[:nbuf] = ck["buf"][kk]
        rng.set_state(ck["rng"]); np.random.set_state(ck["np_global"]); torch.set_rng_state(ck["torch_rng"])
        env.np_random.set_state(ck["env_rng"])
        env.action_space.np_random.bit_generator.state = ck["act_rng"]
        log.write(json.dumps({"resumed_at": step, "time": time.strftime("%F %T")}) + "\n")
        del ck

    while step < args.steps:
        w = rng.dirichlet(np.ones(k))
        s, done, ep_len, ret = env.reset(), False, 0, np.zeros(k)
        while not done:
            a = env.action_space.sample() / max_a if step < args.start_steps else act(s, w, False)
            s2, r, done, _ = env.step(a * max_a)
            ep_len += 1; step += 1; ret += r
            S[nbuf], A[nbuf], R[nbuf], S2[nbuf] = s, a, r, s2
            D[nbuf] = float(done and ep_len < T)   # time-limit ends are not terminal (as COLA)
            nbuf += 1
            if norm is not None:
                norm.update(s)
            if step >= args.start_steps:
                stats = update()
            s = s2
            if step >= args.steps:
                break
        log.write(json.dumps({"step": step, "ep_len": ep_len, "w0": round(float(w[0]), 3),
                              "ret": ret.round(1).tolist(), "wall": round(time.time() - t_start, 1),
                              "lq_la_alpha": None if stats is None else [round(x, 4) for x in stats]}) + "\n")
        if step >= next_eval and step < args.steps:
            evaluate(step)
            next_eval += args.eval_interval
            save_ckpt()
            if args.exit_after and step >= args.exit_after:
                sys.exit(3)
    evaluate(step)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)                    # finished: the objs files and log are the result


if __name__ == "__main__":
    main()
