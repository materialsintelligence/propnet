class ACoercibleClass:
    def __init__(self, x):
        self.x = x


class AnIncoercibleClass:
    def __init__(self, x, y):
        self.x, self.y = x, y
