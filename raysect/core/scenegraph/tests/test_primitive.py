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
from raysect.core.scenegraph import Primitive, Node
from raysect.core.math import Point3D, AffineMatrix3D, translate
from raysect.core import Material, Ray

# TODO: Port to Cython to allow testing of the Cython API and allow access to internal structures
# TODO: Add tests for functionality inherited from Node.

class TestPrimitive(unittest.TestCase):
    """
    Tests the functionality of the scenegraph Primitive class.
    """

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

        n = Primitive()

        self.assertEqual(n.parent, None, "Parent should be None.")
        self.assertEqual(n.root, n, "Primitive should be it's own root as it is not attached to a parent.")
        self.assertEqual(len(n.children), 0, "Child list should be empty.")
        self.assertTransformAlmostEqual(n.transform, AffineMatrix3D(), delta = 1e-14, msg ="Transform should be an identity matrix.")
        self.assertTransformAlmostEqual(n._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Root transform should be an identity matrix.")
        self.assertTransformAlmostEqual(n._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Inverse root transform should be an identity matrix.")
        self.assertEqual(n.name, None, "Primitive name should be None.")
        self.assertTrue(isinstance(n.material, Material), "Primitive material is not a Material object.")

    def test_initialise_with_material(self):
        """Initialisation with a material."""

        m = Material()
        n = Primitive(material = m)

        self.assertEqual(n.parent, None, "Parent should be None.")
        self.assertEqual(n.root, n, "Primitive should be it's own root as it is not attached to a parent.")
        self.assertEqual(len(n.children), 0, "Child list should be empty.")
        self.assertTransformAlmostEqual(n.transform, AffineMatrix3D(), delta = 1e-14, msg ="Transform should be an identity matrix.")
        self.assertTransformAlmostEqual(n._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Root transform should be an identity matrix.")
        self.assertTransformAlmostEqual(n._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Inverse root transform should be an identity matrix.")
        self.assertEqual(n.name, None, "Primitive name should be None.")
        self.assertTrue(n.material is m, "Primitive material was not correctly initialised.")

    def test_initialise_with_all_arguments(self):
        """Initialisation with all arguments."""

        m = Material()
        a = Node()
        b = Primitive(a, translate(1,2,3), m, "My New Primitive")

        # node a
        self.assertEqual(a.parent, None, "Node a's parent should be None.")
        self.assertEqual(a.root, a, "Node a's root should be Node a.")
        self.assertEqual(a.children.count(b), 1, "Node a's child list should contain Node b.")
        self.assertTransformAlmostEqual(a.transform, AffineMatrix3D(), delta = 1e-14, msg ="Node a's transform should be an identity matrix.")
        self.assertTransformAlmostEqual(a._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Node a's root transform should be an identity matrix.")
        self.assertTransformAlmostEqual(a._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Node a's inverse root transform should be an identity matrix.")
        self.assertEqual(a.name, None, "Node a's name should be None.")

        # node b
        self.assertEqual(b.parent, a, "Primitive b's parent should be Node a.")
        self.assertEqual(b.root, a, "Primitive b's root should be Node a.")
        self.assertEqual(len(b.children), 0, "Primitive b's child list should be empty.")
        self.assertTransformAlmostEqual(b.transform, translate(1,2,3), delta = 1e-14, msg = "Primitive b's transform was not set correctly.")
        self.assertTransformAlmostEqual(b._root_transform, translate(1,2,3), delta = 1e-14, msg = "Primitive b's root transform is incorrect.")
        self.assertTransformAlmostEqual(b._root_transform_inverse, translate(1,2,3).inverse(), delta = 1e-14, msg = "Primitive b's inverse root transform is incorrect.")
        self.assertEqual(b.name, "My New Primitive", "Primitive b's name is incorrect.")
        self.assertTrue(b.material is m, "Primitive b's material was not correctly initialised.")

    def test_hit(self):
        """Method hit() is virtual and should raise an exception if called."""

        n = Primitive()

        with self.assertRaises(NotImplementedError, msg="Virtual method did not raise NotImplementedError exception when called."):
            n.hit(Ray())

    def test_contains(self):
        """Method contains() is virtual and should raise an exception if called."""

        n = Primitive()

        with self.assertRaises(NotImplementedError, msg="Virtual method did not raise NotImplementedError exception when called."):
            n.contains(Point3D())

    def test_bounding_box(self):
        """Method bounding_box() is virtual and should raise an exception if called."""

        n = Primitive()

        with self.assertRaises(NotImplementedError, msg="Virtual method did not raise NotImplementedError exception when called."):
            n.bounding_box()

    def test_bounding_sphere(self):
        """Method bounding_sphere() by default calls bounding_box() and should raise an exception if called."""

        n = Primitive()

        with self.assertRaises(NotImplementedError, msg="Method did not raise NotImplementedError exception when called."):
            n.bounding_sphere()

    def test_material(self):
        """Setting and getting material."""

        m = Material()
        n = Primitive()

        n.material = m
        self.assertTrue(n.material is m, "Primitive's material was not correctly set/returned.")

    def test_material_set_invalid(self):
        """Setting an invalid material should raise an exception."""

        n = Primitive()

        with self.assertRaises(TypeError, msg="Attempting to set material with an invalid type (a string) did not raise an exception."):
            n.material = "This should fail!"

        with self.assertRaises(TypeError, msg="Attempting to set material with an invalid type (None) did not raise an exception."):
            n.material = None


if __name__ == "__main__":
    unittest.main()