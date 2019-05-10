from .context import *
import numpy as np


class TestPerceptron(object):
    def test_and_gate(self):
        and_gate = make_logic_gate('and')

        assert and_gate(np.array([0, 0])) == 0
        assert and_gate(np.array([1, 0])) == 0
        assert and_gate(np.array([0, 1])) == 0
        assert and_gate(np.array([1, 1])) == 1

    def test_nand_gate(self):
        nand_gate = make_logic_gate('nand')

        assert nand_gate(np.array([0, 0])) == 1
        assert nand_gate(np.array([1, 0])) == 1
        assert nand_gate(np.array([0, 1])) == 1
        assert nand_gate(np.array([1, 1])) == 0

    def test_or_gate(self):
        or_gate = make_logic_gate('or')

        assert or_gate(np.array([0, 0])) == 0
        assert or_gate(np.array([1, 0])) == 1
        assert or_gate(np.array([0, 1])) == 1
        assert or_gate(np.array([1, 1])) == 1

    def test_nor_gate(self):
        nor_gate = make_logic_gate('nor')

        assert nor_gate(np.array([0, 0])) == 1
        assert nor_gate(np.array([1, 0])) == 0
        assert nor_gate(np.array([0, 1])) == 0
        assert nor_gate(np.array([1, 1])) == 0

    def test_xor_gate(self):
        xor_gate = make_logic_gate('xor')

        assert xor_gate(np.array([0, 0])) == 0
        assert xor_gate(np.array([1, 0])) == 1
        assert xor_gate(np.array([0, 1])) == 1
        assert xor_gate(np.array([1, 1])) == 0
