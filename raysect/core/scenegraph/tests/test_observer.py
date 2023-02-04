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
from raysect.core.scenegraph import Node, Observer
from raysect.core.math import AffineMatrix3D, translate

# TODO: Port to Cython to allow testing of the Cython API and allow access to internal structures
# TODO: Add tests for functionality inherited from Node.

class TestObserver(unittest.TestCase):
    """Tests the function of the scenegraph Observer class."""

    def assertTransformAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """
        Checks 4x4 matrices are equal to a given tolerance.

        This function takes the same arguments as unittest.assertAlmostEqual().
        """

        for i in range(0, 4):

            for j in range(0, 4):

                self.assertAlmostEqual(first[i,j], second[i,j], places, msg, delta)

    def test_initialise_default(self):
        """Default initialisation."""

        n = Observer()
        self.assertEqual(n.parent, None, "Parent should be None.")
        self.assertEqual(n.root, n, "Observer should be it's own root as it is not attached to a parent.")
        self.assertEqual(len(n.children), 0, "Child list should be empty.")
        self.assertTransformAlmostEqual(n.transform, AffineMatrix3D(), delta = 1e-14, msg ="Transform should be an identity matrix.")
        self.assertTransformAlmostEqual(n._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Root transform should be an identity matrix.")
        self.assertTransformAlmostEqual(n._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Inverse root transform should be an identity matrix.")
        self.assertEqual(n.name, None, "Observer name should be None.")

    def test_initialise_with_all_arguments(self):
        """Initialisation with all arguments."""

        a = Node()
        b = Observer(a, translate(1,2,3), "My New Observer")

        # node a
        self.assertEqual(a.parent, None, "Node a's parent should be None.")
        self.assertEqual(a.root, a, "Node a's root should be Node a.")
        self.assertEqual(a.children.count(b), 1, "Node a's child list should contain Observer b.")
        self.assertTransformAlmostEqual(a.transform, AffineMatrix3D(), delta = 1e-14, msg ="Node a's transform should be an identity matrix.")
        self.assertTransformAlmostEqual(a._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Node a's root transform should be an identity matrix.")
        self.assertTransformAlmostEqual(a._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Node a's inverse root transform should be an identity matrix.")

        # node b
        self.assertEqual(b.parent, a, "Observer b's parent should be Node a.")
        self.assertEqual(b.root, a, "Observer b's root should be Node a.")
        self.assertEqual(len(b.children), 0, "Observer b's child list should be empty.")
        self.assertTransformAlmostEqual(b.transform, translate(1,2,3), delta = 1e-14, msg = "Observer b's transform was not set correctly.")
        self.assertTransformAlmostEqual(b._root_transform, translate(1,2,3), delta = 1e-14, msg = "Observer b's root transform is incorrect.")
        self.assertTransformAlmostEqual(b._root_transform_inverse, translate(1,2,3).inverse(), delta = 1e-14, msg = "Observer b's inverse root transform is incorrect.")
        self.assertEqual(b.name, "My New Observer", "Observer's name is incorrect.")

    def test_observe(self):
        """Method observe() is virtual and should raise an exception if called."""

        n = Observer()

        with self.assertRaises(NotImplementedError, msg="Virtual method did not raise NotImplementedError exception when called."):
            n.observe()


if __name__ == "__main__":
    unittest.main()