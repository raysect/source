# Copyright (c) 2014-2019, Dr Alex Meakins, Raysect Project
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
Unit tests for the math utility functions.
"""

import unittest
from raysect.core.math import Point3D, cartesian_to_cylindrical, cylindrical_to_cartesian
import numpy as np


class TestUtilities(unittest.TestCase):

    def test_to_cylindrical(self):
        """Test the to_cylindrical() math utility function."""

        origin = Point3D(0, 0, 0)
        r, z, phi = cartesian_to_cylindrical(origin)
        self.assertEqual(r, 0.0, "Origin in cartesian (0, 0, 0) did not map to the origin in cylindrical coordinates.")
        self.assertEqual(z, 0.0, "Origin in cartesian (0, 0, 0) did not map to the origin in cylindrical coordinates.")
        self.assertEqual(phi, 0.0, "Origin in cartesian (0, 0, 0) did not map to the origin in cylindrical coordinates.")

        point = Point3D(1, 1, 1)
        r_test = np.sqrt(point.x**2 + point.y**2)
        z_test = point.z
        phi_test = np.rad2deg(np.arctan2(point.y, point.x))
        r, z, phi = cartesian_to_cylindrical(point)
        self.assertEqual(r, r_test, "R coordinate did not map successfully.")
        self.assertEqual(z, z_test, "Z coordinate did not map successfully.")
        self.assertEqual(phi, phi_test, "Phi coordinate did not map successfully.")

    def test_to_cartesian(self):
        """Test the to_cartesian() math utility function."""

        r, z, phi = 0, 0, 0
        cartesian_point = cylindrical_to_cartesian(r, z, phi)
        self.assertEqual(cartesian_point.x, 0.0, "Origin in cylindrical coordinates did not map to the origin in cartesian coordinates.")
        self.assertEqual(cartesian_point.y, 0.0, "Origin in cylindrical coordinates did not map to the origin in cartesian coordinates.")
        self.assertEqual(cartesian_point.z, 0.0, "Origin in cylindrical coordinates did not map to the origin in cartesian coordinates.")

        # check invalid radial coordinate
        with self.assertRaises(ValueError, msg="Invalid radial coordinate was not detected."):
            cartesian_point = cylindrical_to_cartesian(-1, 1, 1)

        r, z, phi = 1, 1, 45
        x_test = r * np.cos(np.deg2rad(phi))
        y_test = r * np.sin(np.deg2rad(phi))
        z_test = z
        cartesian_point = cylindrical_to_cartesian(r, z, phi)
        self.assertEqual(cartesian_point.x, x_test, "X coordinate did not map successfully.")
        self.assertEqual(cartesian_point.y, y_test, "Y coordinate did not map successfully.")
        self.assertEqual(cartesian_point.z, z_test, "Z coordinate did not map successfully.")


if __name__ == "__main__":
    unittest.main()
