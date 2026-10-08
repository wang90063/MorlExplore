"""Launch unmodified COLA main.run() under cola_compat.

Usage (from anywhere):
  RUN_DIR=runs/cola/hopper_s0 COLA_NUM_STEPS=1000000 [COLA_EVAL_INTERVAL=50000] [COLA_NO_ANOMALY=1] \
  OMP_NUM_THREADS=1 python third_party/cola_compat/run_cola.py --env_id MO-Hopper-v2 ... (COLA's own flags)

- wandb is never contacted: wandb.init is replaced by a JSONL logger (RUN_DIR/log.jsonl, with wall time).
- COLA_NO_ANOMALY=1 turns off torch.autograd.set_detect_anomaly(True) inside COLA's utils.update_params.
  It only adds NaN checks / stack traces; it does not change the numbers.
- COLA_FAST_ELU=1 swaps nn.ELU for an exp-based equivalent (see below).
"""
import contextlib
import json
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
COLA = os.path.normpath(os.path.join(HERE, "..", "COLA"))
sys.path[:0] = [HERE, COLA]

# Import-only modules missing from (or removed in) the current stack.
_tb = types.ModuleType("torch.utils.tensorboard")
_tb.SummaryWriter = object
sys.modules["torch.utils.tensorboard"] = _tb
_pf = types.ModuleType("pymoo.factory")
_pf.get_performance_indicator = None
sys.modules["pymoo.factory"] = _pf

import torch  # noqa: E402

if os.environ.get("COLA_NO_ANOMALY") == "1":
    torch.autograd.set_detect_anomaly = lambda *a, **k: contextlib.nullcontext()

if os.environ.get("COLA_FAST_ELU") == "1":
    # torch 2.8 arm64 CPU: F.elu/expm1 are ~6x slower than exp. Same function, fp32 rounding
    # differs by <=6e-8 near 0; gradients identical. Used by COLA's latent encoder.
    def _fast_elu(self, x):
        return torch.where(x > 0, x, torch.exp(x) - 1)

    torch.nn.ELU.forward = _fast_elu


class JsonlRun:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(os.path.join(os.path.dirname(path), "ckpt.pt")) and os.path.exists(path):
            os.remove(path)  # a run without checkpoint starts from step 0
        self.f = open(path, "a", buffering=1)
        self.t0 = time.time()

    def log(self, d):
        rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}
        rec["wall"] = round(time.time() - self.t0, 2)
        self.f.write(json.dumps(rec) + "\n")

    def save(self, *a, **k):
        pass


run_dir = os.path.abspath(os.environ.get("RUN_DIR", "cola_run"))
os.makedirs(run_dir, exist_ok=True)
for f in os.listdir(COLA):
    if f.endswith("pref_table.npy") and not os.path.exists(os.path.join(run_dir, f)):
        os.symlink(os.path.join(COLA, f), os.path.join(run_dir, f))
os.chdir(run_dir)

import wandb  # noqa: E402

wandb.init = lambda *a, **k: JsonlRun(os.path.join(run_dir, "log.jsonl"))
os.environ.pop("WANDB_API_KEY", None)

import agent  # noqa: E402

_Orig = agent.SacAgent


CKPT = os.path.join(run_dir, "ckpt.pt")
_SKIP = ("env", "our_wandb")  # rebuilt on resume: the env from main.py, the logger from wandb.init


class SacAgent(_Orig):
    def __init__(self, *a, **k):
        if "COLA_NUM_STEPS" in os.environ:
            k["num_steps"] = int(os.environ["COLA_NUM_STEPS"])
        if "COLA_EVAL_INTERVAL" in os.environ:
            k["eval_interval"] = int(os.environ["COLA_EVAL_INTERVAL"])
        self._cfg = {kk: vv for kk, vv in k.items() if isinstance(vv, (int, float, str, bool, list, type(None)))}
        if not os.path.exists(CKPT):
            with open(os.path.join(run_dir, "config.json"), "w") as fh:
                json.dump(self._cfg, fh, indent=1)
        super().__init__(*a, **k)

    # Checkpoint = the whole agent __dict__ (minus env/logger) in ONE pickle, so shared references survive
    # (e.g. Conflict_caculate holds the q1/q2/policy optimisers), plus every RNG. Saved after a train_episode
    # that ran an eval, i.e. between episodes: the next episode starts with env.reset(), which has no noise.
    def _save_ckpt(self):
        import numpy as np
        import random
        st = {"agent": {kk: vv for kk, vv in self.__dict__.items() if kk not in _SKIP},
              "rng": (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                      self.env.np_random.get_state(), self.env.action_space.np_random.bit_generator.state),
              "log_pos": self.our_wandb.f.tell(), "wall": time.time() - self.our_wandb.t0}
        torch.save(st, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)

    def _load_ckpt(self):
        import numpy as np
        import random
        st = torch.load(CKPT, weights_only=False)
        assert st["agent"]["_cfg"] == self._cfg, "checkpoint config differs"
        self.__dict__.update(st["agent"])
        r = st["rng"]
        random.setstate(r[0]); np.random.set_state(r[1]); torch.set_rng_state(r[2])
        self.env.np_random.set_state(r[3]); self.env.action_space.np_random.bit_generator.state = r[4]
        log = self.our_wandb
        log.f.close()
        with open(log.path, "r+") as fh:  # drop log lines written after the checkpoint
            fh.truncate(st["log_pos"])
        log.f = open(log.path, "a", buffering=1)
        log.t0 = time.time() - st["wall"]
        log.f.write(json.dumps({"resumed_at": self.steps, "time": time.strftime("%F %T")}) + "\n")
        print(f"resumed from checkpoint at step {self.steps}", flush=True)

    def run(self, our_wandb):
        self.our_wandb = our_wandb
        if os.path.exists(CKPT):
            self._load_ckpt()
        super().run(our_wandb)
        if os.path.exists(CKPT):
            os.remove(CKPT)  # finished: objs files and log are the result

    def train_episode(self, *a, **k):
        before = self.previous_evluate
        super().train_episode(*a, **k)
        if self.previous_evluate != before and self.steps <= self.num_steps:
            self._save_ckpt()
            stop = int(os.environ.get("COLA_EXIT_AFTER", "0"))  # testing: simulate an interruption
            if stop and self.steps >= stop:
                sys.exit(3)


agent.SacAgent = SacAgent

import main  # noqa: E402  (main.py sets WANDB_API_KEY at import; wandb.init is already replaced)

os.environ.pop("WANDB_API_KEY", None)
main.run()
