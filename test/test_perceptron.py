from .context import make_logic_gate
from .helpers import export_heatmap
from pytest import mark
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


@mark.graph
class TestPerceptronGraph(object):
    def test_and_graph(self):
        export_domain_graph('and')

    def test_nand_graph(self):
        export_domain_graph('nand')

    def test_or_graph(self):
        export_domain_graph('or')

    def test_nor_graph(self):
        export_domain_graph('nor')

    def test_xor_graph(self):
        export_domain_graph('xor')


def export_domain_graph(gate_type: str):
    # Prepare input data
    x1, x2 = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))

    # Set up logic gate
    logic_gate = make_logic_gate(gate_type)

    # Calc y
    y = np.empty(x1.shape)

    it = np.nditer(y, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        # print("%s %s" % idx, end=' ')
        y[idx] = logic_gate(np.array([x1[idx], x2[idx]]))
        it.iternext()

    # Export graph
    # export_graph(x1, gate_type + '_x1')  # Optional
    # export_graph(x2, gate_type + '_x2')  # Optional
    y[200, 200] = 2
    y[200, 300] = 2
    y[300, 200] = 2
    y[300, 300] = 2
    export_heatmap(y, gate_type + '_y')
