from .context import perceptron
import numpy as np

Perceptron = perceptron.Perceptron


class TestPerceptron(object):
    def test_and_gate(self):
        and_gate = Perceptron(np.array([0.5, 0.5]), -0.75)

        assert and_gate.feed(np.array([0, 0])) == 0
        assert and_gate.feed(np.array([1, 0])) == 0
        assert and_gate.feed(np.array([0, 1])) == 0
        assert and_gate.feed(np.array([1, 1])) == 1

    def test_nand_gate(self):
        nand_gate = Perceptron(np.array([-0.5, -0.5]), 0.75)

        assert nand_gate.feed(np.array([0, 0])) == 1
        assert nand_gate.feed(np.array([1, 0])) == 1
        assert nand_gate.feed(np.array([0, 1])) == 1
        assert nand_gate.feed(np.array([1, 1])) == 0

    def test_or_gate(self):
        or_gate = Perceptron(np.array([0.5, 0.5]), -0.25)

        assert or_gate.feed(np.array([0, 0])) == 0
        assert or_gate.feed(np.array([1, 0])) == 1
        assert or_gate.feed(np.array([0, 1])) == 1
        assert or_gate.feed(np.array([1, 1])) == 1
