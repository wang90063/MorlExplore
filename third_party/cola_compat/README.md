# cola_compat

Shims that let unmodified COLA (third_party/COLA @ c437df0, written for gym 0.15 + mujoco-py)
run on Python 3.9 + gymnasium 1.1 + mujoco 3.2.7. COLA source is not edited.

- `gym/`: minimal gym 0.15 API (make/register/TimeLimit/Wrapper/ActionWrapper/EzPickle/seeding)
  and a `MujocoEnv` over the official `mujoco` bindings (`self.sim.data`, `do_simulation`, ...).
- `assets/hopper.xml`, `assets/walker2d.xml`: COLA's XMLs use `coordinate="global"`, which mujoco 3
  rejects. Converted by loading + `mj_saveLastXML` in mujoco 2.3.3 (the official conversion route).
  Check: 1000 random-action steps in 2.3.3, original vs converted, max |Δqpos| = 4e-16 (hopper),
  2e-6 (walker2d). Other XMLs load unchanged.
- `rltorch/`, `visdom.py`, `fastrand.py`: only the pieces COLA touches.
- `run_cola.py`: launcher (stubs pymoo.factory / tensorboard, replaces wandb with a JSONL logger,
  optional step/eval overrides via env vars).
