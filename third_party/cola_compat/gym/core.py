class Env:
    metadata = {}
    reward_range = (-float("inf"), float("inf"))

    @property
    def unwrapped(self):
        return self

    def close(self):
        pass


class Wrapper(Env):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def seed(self, seed=None):
        return self.env.seed(seed)


class ActionWrapper(Wrapper):
    def step(self, action):
        return self.env.step(self.action(action))
