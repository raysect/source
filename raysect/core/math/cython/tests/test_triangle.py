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
Unit tests for the Vector3D object.
"""

import unittest
import numpy as np
from raysect.core.math.cython.triangle import _test_inside_triangle as inside_triangle


class TestTriangle(unittest.TestCase):

    def test_inside_triangle(self):
        """Tests the inside triangle algorithm."""

        # defining triangle vertices
        v1x, v1y = 0, 0
        v2x, v2y = 1, 1
        v3x, v3y = 1, 0

        # test vertices are inside
        self.assertTrue(inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, v1x, v1y))
        self.assertTrue(inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, v2x, v2y))
        self.assertTrue(inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, v3x, v3y))

        # check line segments are inside
        self.assertTrue(inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, 0.5, 0))
        self.assertTrue(inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, 1, 0.5))
        self.assertTrue(inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, 0.5, 0.5))

        # check an interior point
        self.assertTrue(inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, 0.5, 0.1))

        # check an exterior point
        self.assertFalse(inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, -0.5, -0.5))
        self.assertFalse(inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, 0.5, -0.01))
        self.assertFalse(inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, 1.01, 0.5))
        self.assertFalse(inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, 0.49999, 0.5001))


if __name__ == "__main__":
    unittest.main()



