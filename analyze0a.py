"""Aggregate stage 0a runs into concentration statistics and figures.

Concentration ratio = mean(metric | distance to switch < delta) / mean(metric | distance >= delta).
The null ratio replaces the true switch points with the same number of uniformly random
points, so a ratio well above the null means errors really sit at the switch points.
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from morl_kink.mdp import dist_to_nearest


def ratio(metric, dist, delta):
    near = dist < delta
    if near.sum() == 0 or (~near).sum() == 0:
        return np.nan
    return metric[near].mean() / max(metric[~near].mean(), 1e-12)


def null_dist_local(ts, n_switch, rng):
    """Distances to random switch points, one random set per state with the true count."""
    return np.stack([dist_to_nearest(ts, rng.uniform(0, 1, n)) for n in n_switch], 1)


def config_key(path):
    tag = os.path.basename(path)[:-4]
    return re.sub(r"_s\d+$", "", tag)


def binned(x, y, bins):
    idx = np.digitize(x, bins)
    return np.array([y[idx == i].mean() if (idx == i).any() else np.nan
                     for i in range(1, len(bins))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--delta", type=float, default=0.005)
    p.add_argument("--fig", default="figs")
    args = p.parse_args()
    os.makedirs(args.fig, exist_ok=True)
    rng = np.random.default_rng(0)

    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.runs, "*.npz"))):
        groups[config_key(f)].append(dict(np.load(f)))

    bins = np.concatenate([[0], np.logspace(-4, np.log10(0.2), 14)])
    centers = np.sqrt(bins[1:] * np.maximum(bins[:-1], 1e-5))

    rows = []
    for key, runs in groups.items():
        r_err, r_mis, r_reg, n_err, n_mis, n_reg = [], [], [], [], [], []
        curves = defaultdict(list)
        cos_same, cos_diff = defaultdict(list), defaultdict(list)
        for R in runs:
            ts, dl, dc = R["ts"], R["d_local"], R["d_corner"]
            err = R["abs_err"] / R["Qstar_scale"]
            dl_null = null_dist_local(ts, R["n_switch"], rng)
            dc_null = dist_to_nearest(ts, rng.uniform(0, 1, len(R["corners"])))
            r_err.append(ratio(err.ravel(), dl.ravel(), args.delta))
            r_mis.append(ratio(R["mismatch"].ravel(), dl.ravel(), args.delta))
            r_reg.append(ratio(R["regret"], dc, args.delta))
            n_err.append(ratio(err.ravel(), dl_null.ravel(), args.delta))
            n_mis.append(ratio(R["mismatch"].ravel(), dl_null.ravel(), args.delta))
            n_reg.append(ratio(R["regret"], dc_null, args.delta))
            curves["err"].append(binned(dl.ravel(), err.ravel(), bins))
            curves["mis"].append(binned(dl.ravel(), R["mismatch"].ravel(), bins))
            curves["reg"].append(binned(dc, R["regret"], bins))

            # gradient cosine vs |dt|, same optimal policy vs different
            ci = R["cos_idx"]
            dt = np.abs(ts[ci][:, None] - ts[ci][None, :])
            iu = np.triu_indices(len(ci), 1)
            same = (R["hamming"] == 0)[iu]
            for k in [k for k in R if k.startswith("cos_") and k[4:].isdigit()]:
                step = int(k[4:])
                c = R[k][iu]
                for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2)]:
                    m = (dt[iu] >= lo) & (dt[iu] < hi)
                    if (m & same).sum() >= 3 and (m & ~same).sum() >= 3:
                        cos_same[(step, lo)].append(c[m & same].mean())
                        cos_diff[(step, lo)].append(c[m & ~same].mean())

        row = dict(config=key, n=len(runs),
                   corners=np.mean([len(R["corners"]) for R in runs]),
                   err_ratio=np.nanmean(r_err), err_null=np.nanmean(n_err),
                   mis_ratio=np.nanmean(r_mis), mis_null=np.nanmean(n_mis),
                   reg_ratio=np.nanmean(r_reg), reg_null=np.nanmean(n_reg),
                   rel_err=np.mean([(R["abs_err"] / R["Qstar_scale"]).mean() for R in runs]),
                   mismatch=np.mean([R["mismatch"].mean() for R in runs]),
                   regret=np.mean([R["regret"].mean() for R in runs]))
        steps = sorted({s for s, _ in cos_same})
        if steps:
            last = steps[-1]
            for lo in (0, 0.02, 0.05, 0.1):
                if (last, lo) in cos_same:
                    row[f"cos_same@{lo}"] = np.mean(cos_same[(last, lo)])
                    row[f"cos_diff@{lo}"] = np.mean(cos_diff[(last, lo)])
        rows.append(row)

        fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
        for ax, (name, xl) in zip(axes[:3], [("err", "dist to state's switch"),
                                             ("mis", "dist to state's switch"),
                                             ("reg", "dist to nearest corner")]):
            C = np.array(curves[name])
            ax.errorbar(centers, np.nanmean(C, 0), yerr=np.nanstd(C, 0) / np.sqrt(len(C)), marker="o")
            ax.set_xscale("log")
            ax.set_xlabel(xl)
            ax.set_title({"err": "|Q-Q*| / |Q*|", "mis": "greedy mismatch", "reg": "regret"}[name])
        ax = axes[3]
        if steps:
            for lo, mk in zip((0, 0.02, 0.05, 0.1), "osd^"):
                ys = [np.mean(cos_same[(s, lo)]) if (s, lo) in cos_same else np.nan for s in steps]
                yd = [np.mean(cos_diff[(s, lo)]) if (s, lo) in cos_diff else np.nan for s in steps]
                ax.plot(steps, ys, "-" + mk, color="C0", label=f"same pi*, dt~{lo}")
                ax.plot(steps, yd, "--" + mk, color="C3", label=f"diff pi*, dt~{lo}")
            ax.set_xlabel("train step")
            ax.set_title("grad cosine")
            ax.legend(fontsize=6)
        fig.suptitle(key)
        fig.tight_layout()
        fig.savefig(os.path.join(args.fig, key + ".png"), dpi=110)
        plt.close(fig)

    cols = ["config", "n", "corners", "rel_err", "err_ratio", "err_null", "mismatch",
            "mis_ratio", "mis_null", "regret", "reg_ratio", "reg_null",
            "cos_same@0", "cos_diff@0", "cos_same@0.02", "cos_diff@0.02",
            "cos_same@0.05", "cos_diff@0.05"]
    print("\t".join(cols))
    for r in sorted(rows, key=lambda r: r["config"]):
        print("\t".join(f"{r.get(c, np.nan):.3g}" if not isinstance(r.get(c), str) else r[c]
                        for c in cols))


if __name__ == "__main__":
    main()
