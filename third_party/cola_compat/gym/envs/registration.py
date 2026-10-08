import importlib

from gym.core import Wrapper

_REGISTRY = {}


def register(id, entry_point, max_episode_steps=None, **kwargs):
    _REGISTRY[id] = (entry_point, max_episode_steps, kwargs)


class TimeLimit(Wrapper):
    """gym 0.15 TimeLimit: done=True once max_episode_steps is reached."""

    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return obs, rew, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


def make(id, **kwargs):
    entry_point, max_steps, reg_kwargs = _REGISTRY[id]
    mod_name, cls_name = entry_point.split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    env = cls(**{**reg_kwargs, **kwargs})
    if max_steps is not None:
        env = TimeLimit(env, max_steps)
    return env
