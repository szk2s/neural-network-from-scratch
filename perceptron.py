import numpy as np


class Perceptron:
    """
    2 in 1 out
    """

    def __init__(self, w: np.array, bias: float):
        self.w = w
        self.bias = bias

    def feed(self, x: np.array):
        if np.dot(x, self.w.T) + self.bias > 0:
            return 1
        return 0
