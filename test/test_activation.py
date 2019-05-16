from .context import step_function, sigmoid_function
from .helpers import export_line_chart
from pytest import mark
import numpy as np


class TestActivationFunction(object):
    def test_step_function(self):
        actual = step_function(np.array([-2, -1, -0.5, 0, 0.5, 1, 2]))
        expected = np.array([0, 0, 0, 0, 1, 1, 1])
        assert np.array_equal(actual, expected)

    def test_sigmoid_function(self):
        actual = sigmoid_function(np.array([-2, -1, -0.5, 0, 0.5, 1, 2]))
        expected = np.array([0.11920292, 0.26894142, 0.37754067, 0.5, 0.62245933, 0.73105858, 0.88079708])
        assert np.allclose(actual, expected)


@mark.graph
class TestActivationGraph(object):
    def test_step_function_graph(self):
        x = np.linspace(-10, 10, 100)
        export_line_chart(x, step_function(x), 'step_function')

    def test_sigmoid_function(self):
        x = np.linspace(-10, 10, 100)
        export_line_chart(x, sigmoid_function(x), 'sigmoid_function')
