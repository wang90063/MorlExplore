"""Post-hoc metrics on objs_<step>.npy (201 x 2, row i = return of the policy run with preference i of
generate_w_batch_test(2, 0.005)), identical for COLA and the esac arms.

  HV       2-D hypervolume, ref point [0, 0] (as COLA)
  UT       COLA's utility: mean over its 18-entry PREF table of max over ALL points of w.F (envelope; ignores
           which w made which point)
  EU_cond  mean over w of w.F(pi_w): utility of the point the policy actually returns for w
  ctrl     controllability: mean of Spearman(w0, F0) and Spearman(w0, -F1); 1 = returns are perfectly
           ordered by the preference, 0 = preference input is ignored (cf. arXiv 2605.10585)

Usage: python p2/metrics.py runs/p2/<dir> [...]   -> one row per run at the last and the best-HV eval
"""
import glob
import itertools
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr


def prefs(step=0.005):
    mesh = [np.arange(0, 1 + step, step)] * 2
    w = np.array(list(itertools.product(*mesh)))
    return np.unique(w[w.sum(axis=1) == 1, :], axis=0)


W = prefs()
# COLA's 2-objective UT table, verbatim from agent.py PREF (incl. its [0.55, 0.35] typo), so UT matches its logs
PREF_COLA = np.array([[0.95, 0.05], [0.9, 0.1], [0.85, 0.15], [0.8, 0.2], [0.75, 0.25], [0.7, 0.3], [0.65, 0.35],
                      [0.6, 0.4], [0.55, 0.35], [0.5, 0.5], [0.45, 0.55], [0.4, 0.6], [0.35, 0.65], [0.3, 0.7],
                      [0.25, 0.75], [0.2, 0.8], [0.15, 0.85], [0.1, 0.9]])


def hv2d(F, ref=(0.0, 0.0)):
    P = F[(F[:, 0] > ref[0]) & (F[:, 1] > ref[1])]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(-P[:, 0])]
    hv, best1 = 0.0, ref[1]
    for x, y in P:                       # sweep from large F0 down; add the new strip above best1
        if y > best1:
            hv += (x - ref[0]) * (y - best1)
            best1 = y
    return hv


def _rho(a, b):
    if np.ptp(b) == 0:
        return 0.0
    r = spearmanr(a, b).correlation
    return 0.0 if np.isnan(r) else float(r)


def metrics(F):
    return {
        "HV": hv2d(F),
        "UT": float(np.mean(np.max(PREF_COLA @ F.T, axis=1))),
        "EU_cond": float(np.mean(np.sum(W * F, axis=1))),
        "ctrl": 0.5 * (_rho(W[:, 0], F[:, 0]) + _rho(W[:, 0], -F[:, 1])),
    }


def run_table(d):
    # esac writes objs_*.npy in the run dir, COLA in run_dir/logs/<env>/<name>/summary/
    files = glob.glob(os.path.join(d, "**", "objs_*.npy"), recursive=True)
    by_step = sorted((int(re.search(r"objs_(\d+)\.npy", f).group(1)), f) for f in files)
    return [(s, metrics(np.load(f))) for s, f in by_step]


if __name__ == "__main__":
    print(f"{'run':40s} {'step':>8s} {'HV':>10s} {'UT':>8s} {'EU_cond':>8s} {'ctrl':>6s} | best-HV step")
    for d in sys.argv[1:]:
        t = run_table(d)
        if not t:
            continue
        s, m = t[-1]
        sb, mb = max(t, key=lambda x: x[1]["HV"])
        print(f"{d[-40:]:40s} {s:8d} {m['HV']:10.4g} {m['UT']:8.1f} {m['EU_cond']:8.1f} {m['ctrl']:6.2f} | "
              f"{sb} HV={mb['HV']:.4g}")
