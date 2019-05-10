import numpy as np
from typing import Callable


def make_perceptron(w: np.array, bias: float) -> Callable:
    def perceptron(x: np.array):
        if w.shape != x.shape:
            raise Exception('The shape of x does not match one of w')
        if np.dot(x, w.T) + bias > 0:
            return 1
        return 0

    return perceptron


def make_gate(gate_type: str) -> Callable:
    if gate_type == 'and':
        return make_perceptron(np.array([0.5, 0.5]), -0.75)
    if gate_type == 'nand':
        return make_perceptron(np.array([-0.5, -0.5]), 0.75)
    if gate_type == 'or':
        return make_perceptron(np.array([0.5, 0.5]), -0.25)
    if gate_type == 'nor':
        return make_perceptron(np.array([-0.5, -0.5]), 0.25)
    if gate_type == 'xor':
        and_gate = make_gate('and')
        nand_gate = make_gate('nand')
        or_gate = make_gate('or')

        def xor_gate(x: np.array):
            s1 = nand_gate(x)
            s2 = or_gate(x)
            return and_gate(np.array([s1, s2]))

        return xor_gate
    raise Exception('gate_type should be one of the next. "and", "nand", "or", "nor", "xor".')
