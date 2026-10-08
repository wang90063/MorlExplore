class Visdom:  # COLA only references this in an unused Monitor class
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None
