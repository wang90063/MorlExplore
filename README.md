# morl-kink

Stage-0 go/no-go experiments for the hypothesis:

> In preference-conditioned MORL, much of the reported "objective conflict" comes from a
> single network having to fit a **discontinuous** map ω → π\*. Under linear
> scalarisation the optimal policy jumps at the corner weights of the convex coverage set.

Preferences are w(t) = (1 − t, t), t ∈ [0, 1].

## Setup

```bash
git clone --recurse-submodules <this repo>   # or: git submodule update --init
/usr/bin/python3 -m venv .venv               # Python 3.9
./.venv/bin/python -m pip install -r requirements.txt
```

The stage-0 scripts need only numpy, torch, matplotlib and scipy. The P2 experiments also need
mujoco, gymnasium and pymoo. `third_party/COLA` is a submodule pinned to upstream commit `c437df0`
and is never edited; `third_party/cola_compat/` makes it run on the current stack. Set
`OMP_NUM_THREADS=1` when running jobs in parallel.

| dir | what |
|---|---|
| `morl_kink/`, `exp0*.py`, `analyze0a.py`, `sweep0*.sh` | stage 0 (table below) |
| `audit/` | MORL-Baselines ranking audit (`audit/data/runs.json` is not committed: `python audit/fetch.py`) |
| `geom/`, `esr/` | follow-up candidates (see 2026-10-05) |
| `p2/` | P2 kill test: `esac.py` (base / momba / perobj arms), `metrics.py`, `queue.sh`, `stop.sh` |
| `third_party/` | COLA (submodule) and `cola_compat` shims |
| `runs*/`, `figs/`, `*.log` | results cited below |

## Layout

| file | what |
|---|---|
| `morl_kink/mdp.py` | random two-objective tabular MDPs; exact value iteration; exact corner weights by recursive bisection; per-state switch points |
| `exp0a.py` | 0a: discrete actions. Q-net trained by `sup` (regression onto exact Q\*) or `fqi` (fitted Q iteration, expected backups). Measures error / greedy mismatch / regret vs distance to switch points and per-preference gradient cosines |
| `analyze0a.py` | concentration ratios (near/far a switch) against a random-switch-point null, plus figures in `figs/` |
| `exp0b.py` | 0b: continuous 2D point mass, K goals on an arc, exactly K − 1 known corners. Preference-conditioned TD3 with `mlp` or winner-take-all `multi`-head actor; `--random_start` fixes exploration |
| `exp0b_exact.py` | 0b actor-only: actor trained by DPG against the closed-form Q\* (no critic learning, no exploration); `bc` regresses onto the optimal direction |
| `exp0b_moving.py` | 0b control: one goal sliding continuously with t (no corners, same difficulty) |
| `sweep0a.sh`, `sweep0b.sh` | sweeps |

## Results so far (2026-10-04)

**0a, discrete actions, tabular (60 runs).** Hypothesis not supported here.
- |Q − Q\*| is not concentrated near switch points (ratio 0.85–1.3, null ≈ 1) in both `sup` and `fqi`.
- Greedy-action mismatch is strongly concentrated near switch points (10–1000×), but the
  cost is negligible: regret ≈ 1e-5 of the Q scale, because the two actions tie at the corner.
- Gradient cosine is driven by |Δt|, not by whether two preferences straddle a corner.
- Caveat: one-hot states give no cross-state generalisation.

**0b actor-only, exact critic (81 runs).**
- K = 3, 5 (sparse corners): no corner effect. Closed-loop feedback breaks the tie after
  one step, so an averaged first action still commits to a goal.
- K = 9 (dense corners): MLP actors (width 64 and 256, all seeds) miss goals mostly near
  corners: 83–95 % of misses lie within 0.02 of a corner, a band that covers 32 % of t;
  far-from-corner regret ≈ 0.
- A 4-head winner-take-all actor removes the effect completely, even with 9 goals.
- Confound still open: at K = 9 corners are denser **and** the gaps between goals are narrower.

**0b full TD3, random training starts, width 256, 3 seeds (`runs0b_rs/`, `sweep0b_rs.log`).**
- K = 9, MLP: in every seed, 4–6 % of preferences miss all goals. Each miss segment sits
  exactly between two consecutive goals in the learned goal sequence (e.g. 1, −1, 2, −1, 3 …),
  which is the mode-averaging signature. Regret 0.05–0.07, and 2 of the 9 goals are never
  used (the end goals collapse).
- K = 9, 4-head winner-take-all: 0 misses, all 9 goals used, regret 0.004–0.008 (≈10× lower).
  Its "wrong" goals are cheap near-ties with the neighbouring goal.
- Concentration measured against the *true* corners is only modest (38–67 % of misses in a
  band covering 32 % of t), because the learned switch points drift away from the true ones.
- K = 3: one MLP seed collapses to a single goal and one uses only 2 of 3 goals. This is a
  coverage failure, not corner averaging. The multi-head actor is clean on all seeds.

**Optimisation-confound controls (full TD3, random starts, width 256, 3 seeds).**
- K = 1 (`runs0b_k1/`): both actors perfect (regret 0, miss 0). Uninformative, ceiling.
- Moving goal (`exp0b_moving.py`, `runs0b_moving/`): one goal that slides continuously along
  the same arc with t, so ω → π\* is continuous but the target is as narrow and the reward as
  sparse as at K = 9. MLP: regret 0, miss 0 on every seed; multi: miss 0, regret ≤ 2.5e-4.
  The multi-head actor has no optimisation advantage without corners, so its 10× gain at
  K = 9 comes from the discontinuity (removing mode averaging).
- Decomposition on K = 9: 55–88 % of MLP far-from-corner regret is misses at its own drifted
  switch points; regret on preferences where it picks the right goal is comparable to multi.

## Follow-up candidates (2026-10-05)

Direction 1 was dropped after the literature review (PGMORL, Hyper-MORL, Kubo et al. STCH,
PCHC, D³PO). The kill tests for the next candidates live in their own folders.

| folder | what |
|---|---|
| `audit/fetch.py` | downloads config, summary and final `eval/front` of all 792 public runs in openrlbenchmark/MORL-Baselines (wandb GraphQL via curl) |
| `audit/analyze.py` | ranking stability under HV reference-point sweep, EUM (raw/normalised), IGD+, mean/median/IQM; bootstrap pairwise flips |
| `geom/sac_mo.py` | SAC on an MO-Gymnasium env with fixed preference r0 + c r1; `--init_seed` shares initialisation across preferences |
| `geom/interp.py` | interpolation barriers between per-preference actors, raw and after weight matching |
| `geom/sweep.sh` | 5 preferences × 2 seeds on mo-halfcheetah-v5 |

Extra deps: `pip install pymoo "mujoco==3.2.7" mo-gymnasium imageio` (Python 3.9 has no newer mujoco wheel).

**Leaderboard audit: killed.** Recomputed HV matches the logged HV (median relative error 1e-16).
HV rankings depend only on the reference point (per-objective rescaling multiplies all HVs by the
same constant). The top-ranked algorithm changes with the reference point in 7 of 15 envs, but only
where the top-2 gap is 0.1–3 % and not significant. On the 9 deep-RL envs, 48 of 64 pairs are
separated at 95 %, and those orders do not depend on the reference point. Confident metric-dependent
flips (7) are HV vs linear EUM on deep-sea-treasure-concave and fruit-tree, which is expected by
construction, plus MORL-D ablation variants.

**Pareto-geometry (interpolation barriers): killed** (`geom/interp.log`). 10 SAC actors (c ∈ {0.05,
0.2, 0.5, 1, 2} × 2 seeds, 300k steps), 45 pairs. Median barrier in normalised units, raw → aligned:
same c 0.73 → 0.64, different c / shared init 0.50 → 0.49, different c / different init 0.53 → 0.52.
Everything is disconnected, including the same-preference baseline, and weight matching barely helps.
Barriers shrink with preference distance because they track how much the interpolated weight
emphasises speed (interpolated actors lose speed), not any family structure. Shared initialisation has
no effect after training from scratch. Caveats: actor-only matching; c = 1, seed 1 stuck at a slow
gait (speed 1432 vs 6796).

**A+B (non-linear unknown utility, ESR) K1: passes for kinked utilities** (`esr/k1.py`, `esr/k1.log`).
40 random MOMDPs per transition setting (S = 8, A = 3, H = 10, rewards (k, 2 − k)), exact ESR optimum by
DP over (s, t, accrued return). Median normalised regret (1 = as bad as uniform random), stochastic
transitions (Dirichlet 0.1 / 1.0):
- min utility: oracle-linear 0.67 / 0.58, SER-optimised Markov 0.29 / 0.42, best Markov found 0.09 / 0.23.
- threshold utility: oracle-linear 0.33 / 0.36, SER 0.53 / 0.56, best Markov 0.08 / 0.18.
- sqrt-concave: everything ≤ 0.10; convex: ≤ 0.01. Smooth utilities do not need ESR machinery.
- Deterministic transitions: SER and Markov reach the optimum (median 0); only linear loses.
So the gap exists only when transitions are stochastic **and** the utility has kinks or thresholds.
Caveats: random synthetic MDPs, not a standard benchmark; Markov baselines are local coordinate ascent
(lower bounds); the fitted-linear baseline uses 300 noisy comparisons of random-policy returns.

**A+B K2: passes for reachable kinks, fails for rare thresholds** (`esr/k2.py`, `esr/k2.log`). 30 random
MOMDPs (Dirichlet 1.0) per utility, simulated Bradley-Terry user, monotone non-negative feature model with
a bootstrap ensemble, ESR planning on the ensemble-mean utility. Median normalised regret at 20 / 50 / 100
queries:
- min: Thompson 0.031 / 0.010 / 0.006, random queries 0.11 / 0.08 / 0.09, linear 1.39 / 1.42 / 1.29
  (SER with the *true* utility: 0.42, K1).
- threshold at 0.6 of max: Thompson 0.065 / 0.026 / 0.022, random 0.07 / 0.06 / 0.06, linear ≈ 0.45–0.50.
- threshold at 0.9 of max: Thompson 0.38 / 0.32 / 0.24, random 0.66 / 0.49 / 0.45, **linear 0.25–0.27**.
Diagnosis for 0.9: only 0.9 % of random episodes cross the threshold, and the loss is knife-edge and
asymmetric. Planning with the true utility but the threshold 1 unit too low gives regret 0.76
(2 units: 1.08); 1 unit too high gives 0.18. Open problems this exposes: queries that target the kink
location, and planning that is conservative about it.

**A+B K2b: rare-threshold failure fixed by an anchored (randomised-prior) ensemble** (`esr/k2b.py`,
`esr/k2b.log`). Cause in K2: the L1 prior sets unseen step weights to 0 in every member, so Thompson never
queried near the threshold (0–3 % of queried returns, most members put their R2 step at 1). Anchoring each
member to a draw from a sparse monotone prior keeps diversity where data is missing. Median regret at
20 / 50 / 100 queries, Thompson queries:
- thr0.9: L1 0.39 / 0.27 / 0.24 → anchored 0.18 / 0.088 / 0.054 (linear 0.25–0.27).
- thr0.6: L1 0.067 / 0.018 / 0.011, anchored 0.095 / 0.045 / 0.017. min: L1 0.10 / 0.024 / 0.009,
  anchored 0.083 / 0.030 / 0.011. The prior costs a little on easy utilities at small budgets.
- Conservative planning (pointwise 25 % quantile utility) and max-disagreement (QBC) queries did not help.
With anchoring, learned-utility ESR beats linear and true-utility SER on all three utilities by 50 queries.

**A+B K3: one utility-conditioned augmented Q-network generalises to unseen kinked utilities**
(`esr/k3.py`, `esr/k3.log`). 6 random MOMDPs (Dirichlet 1.0), double DQN, 100k updates, terminal reward
u(R_H) with utility relabelling (transitions are utility-independent, so each batch gets fresh random
utilities). Two families: u = n1 + 1[n2 ≥ θ], θ ∈ [0.3, 1]; u = min(α n1, (1 − α) n2), α ∈ [0.2, 0.8].
Tested on 12 grid values per family, policy read off the network and evaluated exactly.
- Augmented net (sees s, t, R, utility): median regret 0.031 (thr), 0.025 (min); p90 0.09 / 0.06.
  Flat across θ and α, including θ = 0.98.
- Oracle linear on the same utilities: 0–0.56 (median ≈ 0.36 thr, 0.47 min).
- Ablation without R in the input: 0.77 / 0.86, worse than linear. This overstates the cost of dropping
  augmentation: with terminal-only reward the aliased Q-targets are biased, so DQN breaks, whereas K1's
  best Markov policy loses only 0.08–0.23. The fair statement is "augmented learning works; naive
  non-augmented learning fails badly; even the best Markov policy loses 8–23 %".
Caveats: still tabular dynamics (one-hot state, H = 10); no continuous-state test yet.

**A+B K4: full interactive loop on a continuous-state stochastic env works; the amortised network is the
bottleneck** (`esr/k4.py`, `esr/k4.log`, training `esr/k4_train.log`). Two-patch env (continuous position,
noisy moves, Bernoulli harvests, H = 20). One Q-network conditioned on the 142-d utility weight vector,
trained 300k updates with utility relabelling on prior draws, means of draws, joint-step staircases and
linear weights (test utilities never seen). Loop: anchored ensemble + Thompson queries answered by rolling
out the network for two members (no re-planning). Reference: a DQN trained on the true utility alone.
Median normalised regret at 20 / 50 / 100 queries (20 seeds):
- thr0.3: loop 0.25 / 0.23 / 0.22, linear 0.58; network given the *true* utility 0.21.
- thr0.5: loop 0.25 / 0.18 / 0.17, linear 0.31; true-utility network 0.14.
- min: loop 0.13 / 0.14 / 0.15, linear 1.04; true-utility network 0.10.
Elicitation is nearly lossless by 50 queries; almost all remaining regret is the conditioned network's own
approximation gap (10–21 % vs a dedicated DQN).
The first version (`esr/k4_v1_far_patches.log`, patches at 0.2 / 0.8, thresholds 0.6 / 0.9) was invalid:
switching took ~6 of 20 steps so linear was already optimal (checked with scripted policies), and the
network had never seen joint-step utilities, so min failed even with the true utility (0.97).
Caveats: one hand-designed env; no non-augmented learned baseline here (see K1/K3); refs are single seeds.

**A+B line abandoned (2026-10-06).** For a top-venue target the K1–K4 contribution reads as an assembly,
and the proposed pivot (preference learning fails on rare kinks; randomized priors fix it) largely overlaps
RUNE (arXiv 2205.12401), randomized prior functions (arXiv 1806.03335) and active-learning bias under
misspecification (arXiv 2205.13698). Our own data also undercut the "permanent trap" claim (L1 regret on
thr0.9 keeps falling, 0.49 → 0.24), and anchoring costs regret on common utilities. Decision: stop here.

**0b full TD3, fixed start.** Invalid as a test: sparse reward + start at origin means most
runs commit to one goal regardless of preference (exploration failure). Log kept as
`sweep0b_fixedstart_broken.log`. Also found and fixed a bug: the box action space let
diagonal moves go √2 faster; actions are now clipped to the unit disc.

## P2 kill test: COLA on the Mac (2026-10-07)

Route P2 (see the Claude Doc): compare competing diagnoses of why preference-conditioned
networks fail, on one backbone. Step 1 is getting the original COLA code
(`third_party/COLA`, github.com/yeshenpy/COLA @ c437df0, **unmodified**) to run here.

**Port (`third_party/cola_compat/`, details in its README).** COLA targets gym 0.15 + mujoco-py.
A small `gym` shim gives it the old API on top of gymnasium spaces and the official `mujoco` 3.2.7
bindings (old `MujocoEnv` semantics, `TimeLimit` with done at 500). `rltorch`, `visdom` and `fastrand` are
replaced by the few lines COLA actually uses. `pymoo.factory` and tensorboard are import-only stubs.
wandb is replaced by a local JSONL logger, so COLA's hardcoded API key is never used.
- `hopper.xml` and `walker2d.xml` use `coordinate="global"`, which mujoco 3 rejects. They were converted
  by load + `mj_saveLastXML` in mujoco 2.3.3. A 1000-step random rollout matches the original:
  max |Δqpos| is 4e-16 for hopper and 2e-6 for walker2d.
- Caveat: physics is mujoco 3.2.7, not the mujoco 1.5/2.0 of the paper, so absolute HV is not
  comparable to the paper. All arms will share the same simulator, so comparisons between arms are valid.
- Run: `third_party/cola_compat/launch.sh <hopper|ant|walker|cheetah> <seed> <run_dir> [steps] [eval_interval]`.
  Flags are copied from COLA's `run.sh`.

**Speed (Apple M5, 1 torch thread per process).**
- `torch.autograd.set_detect_anomaly(True)` wraps every backward in COLA. It is numerically irrelevant
  (two seeded runs with it on and off gave identical episodes), so it is disabled (`COLA_NO_ANOMALY=1`).
  The speed gain is small.
- In this torch 2.8 arm64 build, `F.elu` costs 366 µs on a 256×256 tensor, against 61 µs for `exp`,
  because ELU/expm1 are not vectorised. ELU sits in COLA's latent encoder, and 22.5 s of 65 s of
  `learn()` was ELU. `COLA_FAST_ELU=1` swaps in `where(x>0, x, exp(x)-1)`. Max output difference is
  6e-8 and the gradients are identical.
- Learning-phase throughput:

| setting | Hopper | Ant |
|---|---|---|
| 1 process alone | 70 steps/s | not measured cleanly |
| 6 in parallel (3 Hopper + 3 Ant) | 24–43 steps/s each | 39 steps/s each |

- COLA is about 10× slower than a plain preference-conditioned SAC (647 steps/s, `bench_sac`).
  Per update it trains the latent dynamics, computes two critic losses and a conflict-gradient
  term, and runs the policy loss over the full preference set.
- Full evaluation (201 preferences × 3 episodes) takes about 30–60 s on Hopper and about 2 min on Ant.
- Therefore 1M COLA steps ≈ 7–8 h per run in a 6-way parallel batch, plus about 40 min of evals.
- **Pitfall:** one batch "took" 50 min for a single Ant eval. `pmset -g log` shows the lid was closed
  at 19:54 on battery ("Clamshell Sleep"), so the Mac slept. Long runs need the Mac on AC power with
  the lid open.

Smoke results (20–25k steps, only checking that it learns): the HV of the final 201-preference
evaluation was 0.6–1.5e6 on Hopper and 1.3–2.8e5 on Ant (logs in `runs/cola/speed_*`).

## P2 kill test: the other three arms, metrics, launch (2026-10-07)

`p2/esac.py --arm {base,momba,perobj}` runs on COLA's own envs (via `cola_compat`) with COLA's
hyper-parameters (batch 256, [256,256], lr 3e-4, γ 0.99, τ 0.005, auto entropy, 10k random start
steps, 1 update per env step, time-limit ends not terminal, actions scaled by `action_space.high`).

| arm | what it is |
|---|---|
| `base` | Envelope SAC: twin vector critic Q(s,a,w)∈R², preference w ~ Dirichlet(1) per episode, fresh w per transition in each batch; target = vector soft value of the best of 4 candidate w′ (w plus 3 random), scored by w·Q; twin chosen by lower w·Q |
| `momba` | `base` + Momba-style network changes (arXiv 2608.07180): running obs normalisation, weight norm + LayerNorm in every hidden layer. **Not included:** its distributional returns (item iii) |
| `perobj` | `base`, but one critic network per objective (D3PO-style decomposition) |
| `cola` | original COLA code, `third_party/cola_compat/launch.sh` |

Differences from COLA kept on purpose: COLA runs a whole episode and then does that many
updates in a row with a single preference pair; `esac` updates after every env step (same number of
updates). Eval is one episode per preference (reset has no noise and the eval policy is
deterministic, so COLA's three episodes are identical).

`p2/metrics.py <run dirs>` reads every `objs_<step>.npy` (201×2, same preference grid for all arms):

- `HV`: ref point [0,0]. Checked against pymoo and against COLA's own logged `hypervolume` on 9 COLA
  runs; it matches to the last digit.
- `UT`: COLA's definition. It uses its 18-entry PREF table, typo [0.55,0.35] included, and the max
  over all points. Matches COLA's logged `UT` exactly.
- `EU_cond`: mean of w·F(π_w), i.e. the point the policy actually returns for each w.
- `ctrl`: mean of Spearman(w₀, F₀) and Spearman(w₀, −F₁). It is 1 when returns follow the
  preference and 0 when the preference is ignored.

Speed (3 processes in parallel, Hopper, training phase): base ≈ 180, momba ≈ 110, perobj ≈ 120 steps/s,
vs COLA ≈ 30. One full eval: 3–7 s on Hopper, ≈ 27 s on Ant.

Launch: `caffeinate -i p2/queue.sh 8 1000000` runs the 24 jobs (4 arms × Hopper/Ant × seeds 1–3),
8 at a time, with COLA first. Output goes to `runs/p2/kill/<arm>_<task>_s<seed>/`, with `done.txt`
per job. Re-running the script skips jobs that finished with exit=0. A dry run with 1.2k steps finished
all 24 jobs with exit=0. Full run started 2026-10-07 21:27 CST.

### Fix: `base` switched from Envelope target to CAPQL-style target (2026-10-07 22:50)

At 1.3 h the interim metrics showed the Envelope `base` arm diverging on Ant in both seeds. Critic
loss climbed from about 1–5 to 1e4–3.6e5 starting near 240k steps. Eval returns fell from +340/+460
at 200k to about −750 at 400k, and HV dropped to 0. Ant never terminates early, so every target
bootstraps through the full 500-step horizon. Taking the max over 4 candidate preferences in the
target adds a steady upward bias, and that bias compounds along the bootstrap chain.

- New default `--n_env_w 1`: the target uses only the transition's own w. This is CAPQL
  (Lu, Herman & Yu, ICLR 2023): preference-conditioned SAC with a vector critic and an
  entropy-augmented vector target. It is also the backbone the Momba paper builds on
  (via morl-baselines), so the `momba` arm now matches that paper's setup more closely.
- The diverged runs are kept in `runs/p2/kill_envelope_diverged/`.
- base_ant_s1 and base_ant_s2 were stopped (exit=143). A waiter (`kill -0` on the queue PID)
  re-runs `p2/queue.sh` after the main queue ends, which re-does only those two.
- Every other esac job started after the change, so all esac results use the CAPQL-style target.
  The COLA jobs are unaffected.

Interim at about 150k steps: COLA Hopper HV 1.07–1.25e7, COLA Ant HV 1.8–3.0e6.

### Paused by the user (2026-10-07 22:57)

Every process is stopped: the queue, the rerun waiter, caffeinate, and all python jobs. None of
the jobs save checkpoints, so a restart begins from step 0. Logs and `objs_*.npy` up to the stop are
kept in `runs/p2/kill/`.

| run | reached |
|---|---|
| COLA Ant s1–3 | 178.5k |
| COLA Hopper s1–3 | 212–214k |
| base (CAPQL) Ant s3 | 98k |
| base (CAPQL) Hopper s1 | 109k |

To resume, move or delete `runs/p2/kill/` (or point `OUT=` somewhere new), then run
`caffeinate -i p2/queue.sh 8 1000000`.

### Checkpointing added (2026-10-08)

Every job now saves `ckpt.pt` in its run dir after each eval (every 50k steps). The save is atomic:
it writes a temp file, then renames it. If a job is killed, starting it again continues from the last
checkpoint, and the result is **bitwise identical** to an uninterrupted run. When the run finishes,
the checkpoint is deleted.

- **esac (base / momba / perobj)**, in `p2/esac.py`. The checkpoint holds:
  - the networks, optimisers, alpha, obs-normaliser and filled buffer slice;
  - every RNG (python, numpy global, the agent's rng, torch, env, action space);
  - the log position.

  Test: momba Hopper, 25k steps, killed at 10k and resumed. All `objs_*.npy` and all 234 log lines
  matched the uninterrupted run.
- **COLA**, in `third_party/cola_compat/run_cola.py`, a `SacAgent` subclass. COLA's own code is
  untouched. The checkpoint pickles the whole agent `__dict__` in one go, so shared references
  survive (for example, the conflict module holds the optimisers). It leaves out the env and the
  logger, and stores all RNGs plus the log position. Test: Hopper, 30k steps, killed at 20k
  (`COLA_EXIT_AFTER`) and resumed. All 3 `objs_*.npy` and all 27 log lines matched.
- **On resume:**
  - `log.jsonl` is cut back to the checkpoint position, and a `{"resumed_at": step}` line is added.
  - `stdout.txt` is appended to, not truncated.
  - COLA's own `*_obj_list.txt` files may repeat lines from the lost segment. Use `objs_*.npy` and
    `log.jsonl` for results.
- **Size.**
  - COLA preallocates its 1M replay buffer, so its checkpoint is a fixed ~140 MB on Hopper and
    ~250 MB on Ant.
  - esac checkpoints grow with the buffer, up to roughly 100–300 MB.
  - All 24 jobs together take a few GB.
- **Stop:** `p2/stop.sh` kills the queue and every job, then lists the checkpoints.
- **Resume:** run `caffeinate -i p2/queue.sh 8 1000000` again. Jobs whose `done.txt` says `exit=0`
  are skipped, and every other job continues from its `ckpt.pt`.
- The partial runs in `runs/p2/kill/` from 2026-10-07 predate this change and have no checkpoints.
  A restart has to begin from step 0.
