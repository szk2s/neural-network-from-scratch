import numpy as np


def step_function(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(int)


def sigmoid_function(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
