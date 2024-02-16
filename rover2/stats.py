"""Real time statistics."""


class DutyCycle:

    def __init__(self) -> None:
        self.min = 1.0
        self.max = 0.0
        self.m1 = 0.
        self.m2 = 0.
        self.n = 0

    def observe(self, x: float = 0) -> None:
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        self.m1 += x
        self.m2 += x**2
        self.n += 1

    def print(self):
        print("min:", self.min)
        print("max:", self.max)
        print("avg:", self.m1 / self.n)
        print("std:", ((self.m2 / self.n) - (self.m1 / self.n)**2)**0.5)

