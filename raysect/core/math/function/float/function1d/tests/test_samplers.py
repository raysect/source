import unittest
from raysect.core.math.function.float.function1d.samplers import sample1d, sample1d_points
import numpy as np


class TestSamplers(unittest.TestCase):
    def setUp(self) -> None:

        self.power_sampling = np.array(
            [0.            , 0.052631578947, 0.105263157895, 0.157894736842, 0.210526315789, 0.263157894737,
             0.315789473684, 0.368421052632, 0.421052631579, 0.473684210526, 0.526315789474, 0.578947368421,
             0.631578947368, 0.684210526316, 0.736842105263, 0.789473684211, 0.842105263158, 0.894736842105,
             0.947368421053, 1.],
            dtype=np.float64,
        )

        self.power_series = np.array(
            [1., 0.856440465581, 0.78900857969, 0.747182121597, 0.720341087935, 0.703761280735,
             0.69488837643 , 0.692200903397, 0.694745666856, 0.70191450928 , 0.713325219075, 0.728753279957,
             0.748090779899, 0.771320888818, 0.798501799632, 0.829756728352, 0.865267989283, 0.90527395035,
             0.950068132968, 1.],
            dtype=np.float64,
        )

        self.power_fun = lambda x: x ** x

    def test_sample1d(self):
        x_points, f_values = sample1d(self.power_fun, 0, 1, 20)
        np.testing.assert_array_almost_equal(self.power_series, f_values, decimal=10)
        np.testing.assert_array_almost_equal(self.power_sampling, x_points, decimal=10)

    def test_sample1d_points(self):
        f_values = sample1d_points(self.power_fun, self.power_sampling)
        np.testing.assert_array_almost_equal(self.power_series, f_values, decimal=10)
