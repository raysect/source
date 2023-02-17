# Copyright (c) 2014-2023, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
