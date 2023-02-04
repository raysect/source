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

"""
Unit tests for the utility package.
"""

import unittest
import numpy as np
from raysect.core.math.cython.utility import _minimum as minimum, _maximum as maximum, _peak_to_peak as peak_to_peak
from raysect.core.math.cython.utility import _test_winding2d as winding2d, _point_inside_polygon as point_inside_polygon


class TestUtility(unittest.TestCase):

    def test_maximum(self):
        """Tests the maximum value calculation for memoryviews."""

        data = np.array([2, 4, -3, 9], dtype=float)
        self.assertEqual(maximum(data), 9)

    def test_minimum(self):
        """Tests the minimum value calculation for memoryviews."""

        data = np.array([2, 4, -3, 9], dtype=float)
        self.assertEqual(minimum(data), -3)

    def test_peak_to_peak(self):
        """Tests the peak to peak value calculation for memoryviews."""

        data = np.array([2, 4, -3, 9], dtype=float)
        self.assertEqual(peak_to_peak(data), 12)

    def test_clockwise_polygon_winding(self):
        """Tests the algorithm returns True (clockwise) for a clockwise polygon."""

        poly = np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=np.double)
        self.assertTrue(winding2d(poly))

    def test_anticlockwise_polygon_winding(self):
        """Tests the algorithm returns False (anti-clockwise) for an anti-clockwise polygon."""

        poly = np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=np.double)
        poly = np.array(poly[::-1])
        self.assertFalse(winding2d(poly))

    def test_point_inside_polygon(self):
        """Tests points inside and outside a polygon are correctly identified."""

        poly = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.double)

        self.assertTrue(point_inside_polygon(poly, 0.5, 0.5))
        self.assertTrue(point_inside_polygon(poly, 0.0000001, 0.00000001))
        self.assertFalse(point_inside_polygon(poly, -0.5, 0.5))
        self.assertFalse(point_inside_polygon(poly, -0.5, -0.5))
        self.assertFalse(point_inside_polygon(poly, 1.000001, 1.000001))


if __name__ == "__main__":
    unittest.main()
