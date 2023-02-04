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
Unit tests for the Arg3D class.
"""

import unittest
from raysect.core.math.function.float.function3d.arg import Arg3D

# TODO: expand tests to cover the cython interface
class TestArg3D(unittest.TestCase):

    def test_arg(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            for y in v:
                for z in v:
                    argx = Arg3D("x")
                    argy = Arg3D("y")
                    argz = Arg3D("z")
                    self.assertEqual(argx(x, y, z), x, "Arg3D('x') call did not match reference value.")
                    self.assertEqual(argy(x, y, z), y, "Arg3D('y') call did not match reference value.")
                    self.assertEqual(argz(x, y, z), z, "Arg3D('z') call did not match reference value.")

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError, msg="Arg3D did not raise ValueError with incorrect string."):
            Arg3D("w")
