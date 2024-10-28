class Identity:
    def __call__(self, x):
        return x


class Shift:
    def __init__(self, s):
        self.s = s

    def __call__(self, x):
        return x + self.s
