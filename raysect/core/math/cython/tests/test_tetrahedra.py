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
# from raysect.core.math.cython.tetrahedra import _test_inside_tetrahedra as inside_tetrahedra
from raysect.core.math.cython.tetrahedra import _test_barycentric_inside_tetrahedra as inside_tetrahedra


class TestTetrahedra(unittest.TestCase):

    def test_inside_tetrahedra(self):
        """Tests the inside tetrahedra algorithm."""

        # defining triangle vertices
        v1x, v1y, v1z = 0, 0, 0
        v2x, v2y, v2z = 1, 0, 0
        v3x, v3y, v3z = 0, 1, 0
        v4x, v4y, v4z = 0, 0, 1

        # test vertices are inside
        self.assertTrue(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, v1x, v1y, v1z))
        self.assertTrue(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, v2x, v2y, v2z))
        self.assertTrue(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, v3x, v3y, v3z))
        self.assertTrue(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, v4x, v4y, v4z))

        # check line segments are inside
        self.assertTrue(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, 0.5, 0, 0))
        self.assertTrue(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, 0, 0.5, 0))
        self.assertTrue(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, 0, 0, 0.5))
        self.assertTrue(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, 0.5, 0, 0.5))
        self.assertTrue(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, 0, 0.5, 0.5))
        self.assertTrue(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, 0.5, 0.5, 0))

        # check an interior point
        self.assertTrue(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, 0.25, 0.25, 0.25))

        # check an exterior point
        self.assertFalse(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, -0.5, -0.5, -0.5))
        self.assertFalse(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, 0.5, -0.01, 0.5))
        self.assertFalse(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, -0.01, 0.5, 0.5))
        self.assertFalse(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, 0.25, 0.25, -0.01))
        self.assertFalse(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, 0.5, 0.5, 0.0001))
        self.assertFalse(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, 1.0, 1.0, 1.0))
        self.assertFalse(inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, 0.3333, 0.3333, 0.335))


if __name__ == "__main__":
    unittest.main()
