"""Kill test for the leaderboard audit: are MORL-Baselines algorithm rankings stable
when the hypervolume reference point, the utility metric or the seed aggregate changes?

Note: per-objective positive rescaling multiplies every HV by the same constant, so HV
rankings depend on the reference point only. The ref sweep below is therefore the whole
HV story; normalisation matters only for utility metrics and IGD+.

Usage: python audit/analyze.py  (after audit/fetch.py)
"""
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from pymoo.indicators.hv import HV
from scipy.stats import kendalltau

D = os.path.join(os.path.dirname(__file__), "data")
MIN_SEEDS = 3
ALPHAS = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]           # ref = nadir - alpha * range
rng = np.random.default_rng(0)


def load():
    runs = json.load(open(os.path.join(D, "runs.json")))
    by_env = defaultdict(lambda: defaultdict(list))
    for r in runs:
        p = os.path.join(D, "fronts", r["name"] + ".json")
        if r["state"] != "finished" or not os.path.exists(p):
            continue
        F = np.array(json.load(open(p))["data"], dtype=float)
        if F.ndim != 2 or len(F) == 0 or not np.isfinite(F).all():
            continue
        cfg = r["config"]
        by_env[cfg["env_id"]][cfg["algo"]].append(dict(
            F=F, ref=cfg.get("ref_point"), hv_logged=r["summary"].get("eval/hypervolume"),
            eum_logged=r["summary"].get("eval/eum"), steps=cfg.get("total_timesteps"), name=r["name"]))
    return {e: {a: rs for a, rs in algos.items() if len(rs) >= MIN_SEEDS}
            for e, algos in by_env.items()}


def hv(F, ref):
    F = F[(F > ref).all(1)]
    return 0.0 if len(F) == 0 else float(HV(ref_point=-np.asarray(ref, float))(-F))


def nondominated(P):
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if keep[i]:
            dom = (P >= P[i]).all(1) & (P > P[i]).any(1)
            if dom.any():
                keep[i] = False
    return P[keep]


def igd_plus(F, Z):
    """IGD+ for maximisation: mean over reference points z of min_f ||max(z - f, 0)||."""
    d = np.maximum(Z[:, None, :] - F[None], 0)
    return float(np.linalg.norm(d, axis=-1).min(1).mean())


def eum(F, W):
    return float((W @ F.T).max(1).mean())


def iqm(x):
    x = np.sort(x)
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def env_metrics(algos):
    P = np.concatenate([r["F"] for rs in algos.values() for r in rs])
    lo, hi = P.min(0), P.max(0)
    rngv = np.where(hi > lo, hi - lo, 1.0)
    d = P.shape[1]
    W = rng.dirichlet(np.ones(d), 2000)
    Z = (nondominated(P) - lo) / rngv
    out = defaultdict(dict)                          # metric -> algo -> per-seed array (higher=better)
    for a, rs in algos.items():
        for key in ["hv_orig"] + [f"hv_a{al}" for al in ALPHAS] + ["eum_raw", "eum_norm", "neg_igd+"]:
            out[key][a] = []
        for r in rs:
            if r["ref"] is not None:
                out["hv_orig"][a].append(hv(r["F"], r["ref"]))
            for al in ALPHAS:
                out[f"hv_a{al}"][a].append(hv(r["F"], lo - al * rngv))
            Fn = (r["F"] - lo) / rngv
            out["eum_raw"][a].append(eum(r["F"], W))
            out["eum_norm"][a].append(eum(Fn, W))
            out["neg_igd+"][a].append(-igd_plus(Fn, Z))
    return {m: {a: np.array(v) for a, v in vs.items() if len(v)} for m, vs in out.items()}


def ranking(vals, agg):
    f = dict(mean=np.mean, median=np.median, iqm=iqm)[agg]
    s = {a: f(v) for a, v in vals.items()}
    return sorted(s, key=lambda a: -s[a]), s


def p_better(x, y, n=2000):
    """Bootstrap P(mean x > mean y)."""
    bx = rng.choice(x, (n, len(x))).mean(1)
    by = rng.choice(y, (n, len(y))).mean(1)
    return float((bx > by).mean())


def main():
    data = load()
    report, flips_all = {}, []
    sanity = []
    for env in sorted(data):
        algos = data[env]
        if len(algos) < 2:
            continue
        M = env_metrics(algos)
        for a, rs in algos.items():                 # recomputed vs logged HV
            for r, h in zip(rs, M["hv_orig"].get(a, [])):
                if r["hv_logged"]:
                    sanity.append(abs(h - r["hv_logged"]) / max(abs(r["hv_logged"]), 1e-9))
        base = "hv_orig" if len(M["hv_orig"]) == len(algos) else "hv_a0.1"
        base_rank, _ = ranking(M[base], "mean")
        rows = {}
        for m in M:
            if len(M[m]) != len(algos):
                continue
            for agg in ("mean", "median", "iqm"):
                order, s = ranking(M[m], agg)
                tau = kendalltau([base_rank.index(a) for a in algos], [order.index(a) for a in algos])[0]
                rows[f"{m}/{agg}"] = dict(order=order, tau=float(tau), top_same=order[0] == base_rank[0])
        # pairwise flips that are confident under both metrics
        for a, b in itertools.combinations(algos, 2):
            pb = {m: p_better(M[m][a], M[m][b]) for m in M if a in M[m] and b in M[m]}
            conf_a = [m for m, p in pb.items() if p > 0.95]
            conf_b = [m for m, p in pb.items() if p < 0.05]
            if conf_a and conf_b:
                flips_all.append(dict(env=env, a=a, b=b, a_wins=conf_a, b_wins=conf_b))
        steps = {a: sorted({str(r["steps"]) for r in rs}) for a, rs in algos.items()}
        report[env] = dict(base=base, base_rank=base_rank, n={a: len(rs) for a, rs in algos.items()},
                           steps=steps, rows=rows,
                           means={m: {a: float(v.mean()) for a, v in M[m].items()} for m in M})
    json.dump(dict(report=report, flips=flips_all), open(os.path.join(D, "audit.json"), "w"), indent=1)

    s = np.array(sanity)
    print(f"sanity: recomputed vs logged HV, median rel err {np.median(s):.2e}, "
          f"95th pct {np.percentile(s, 95):.2e}, n={len(s)}\n")
    for env, r in report.items():
        print(f"== {env}  (base {r['base']}, seeds {r['n']})")
        print("   base ranking:", " > ".join(r["base_rank"]))
        if len({tuple(v) for v in r["steps"].values()}) > 1:
            print("   budgets differ:", r["steps"])
        changed = {k: v for k, v in r["rows"].items() if v["tau"] < 0.999}
        for k, v in changed.items():
            print(f"   {k:22s} tau={v['tau']:+.2f} top_same={v['top_same']}  {' > '.join(v['order'])}")
    print(f"\nconfident pairwise flips (P>0.95 under one metric, <0.05 under another): {len(flips_all)}")
    for f in flips_all:
        print(f"   {f['env']}: {f['a']} vs {f['b']}  | a wins {f['a_wins']}  | b wins {f['b_wins']}")


if __name__ == "__main__":
    main()
