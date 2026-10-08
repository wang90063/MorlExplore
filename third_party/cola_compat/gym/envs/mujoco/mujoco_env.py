"""gym 0.15 MujocoEnv semantics on top of the official mujoco (>=3) bindings."""
import os

import mujoco
import numpy as np
from gymnasium import spaces

from gym.core import Env
from gym.utils import seeding

_CONVERTED = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")


def _resolve(model_path):
    # COLA's hopper.xml / walker2d.xml use coordinate="global"; use the 2.3.3-converted copies.
    alt = os.path.normpath(os.path.join(_CONVERTED, os.path.basename(model_path)))
    return alt if os.path.exists(alt) else model_path


class _Sim:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)


class MujocoEnv(Env):
    def __init__(self, model_path, frame_skip):
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "assets", model_path)
        self.frame_skip = frame_skip
        self.model = mujoco.MjModel.from_xml_path(_resolve(model_path))
        self.sim = _Sim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self.metadata = {"render.modes": [], "video.frames_per_second": int(np.round(1.0 / self.dt))}
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        self.action_space = spaces.Box(low=bounds[:, 0], high=bounds[:, 1], dtype=np.float32)
        self.seed()
        observation, _reward, done, _info = self.step(self.action_space.sample())
        assert not done
        self.obs_dim = observation.size
        high = np.inf * np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_model(self):
        raise NotImplementedError

    def viewer_setup(self):
        pass

    def reset(self):
        self.sim.reset()
        return self.reset_model()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def get_body_com(self, body_name):
        return self.sim.data.body(body_name).xpos.copy()

    def state_vector(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])

    def render(self, *args, **kwargs):
        raise NotImplementedError("rendering not supported in cola_compat")
