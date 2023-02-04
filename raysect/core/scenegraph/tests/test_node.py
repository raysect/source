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
from raysect.core.scenegraph.node import Node
from raysect.core.math import AffineMatrix3D, translate

# TODO: Port to Cython to allow testing of the Cython API and allow access to internal structures

class TestNode(unittest.TestCase):
    """Tests the function of the scenegraph Node class."""

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

        n = Node()
        self.assertEqual(n.parent, None, "Parent should be None.")
        self.assertEqual(n.root, n, "Node should be it's own root as it is not attached to a parent.")
        self.assertEqual(len(n.children), 0, "Child list should be empty.")
        self.assertTransformAlmostEqual(n.transform, AffineMatrix3D(), delta = 1e-14, msg ="Transform should be an identity matrix.")
        self.assertTransformAlmostEqual(n._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Root transform should be an identity matrix.")
        self.assertTransformAlmostEqual(n._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Inverse root transform should be an identity matrix.")
        self.assertEqual(n.name, None, "Node name should be None.")

    def test_initialise_with_parent(self):
        """Initialisation with a parent."""

        a = Node()
        b = Node(a)

        # node a
        self.assertEqual(a.parent, None, "Node a's parent should be None.")
        self.assertEqual(a.root, a, "Node a's root should be Node a.")
        self.assertEqual(a.children.count(b), 1, "Node a's child list should contain Node b.")
        self.assertTransformAlmostEqual(a.transform, AffineMatrix3D(), delta = 1e-14, msg ="Node a's transform should be an identity matrix.")
        self.assertTransformAlmostEqual(a._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Node a's root transform should be an identity matrix.")
        self.assertTransformAlmostEqual(a._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Node a's inverse root transform should be an identity matrix.")
        self.assertEqual(a.name, None, "Node name should be None.")

        # node b
        self.assertEqual(b.parent, a, "Node b's parent should be Node a.")
        self.assertEqual(b.root, a, "Node b's root should be Node a.")
        self.assertEqual(len(b.children), 0, "Node b's child list should be empty.")
        self.assertTransformAlmostEqual(b.transform, AffineMatrix3D(), delta = 1e-14, msg ="Node b's transform should be an identity matrix.")
        self.assertTransformAlmostEqual(b._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Node b's root transform should be an identity matrix.")
        self.assertTransformAlmostEqual(b._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Node b's inverse root transform should be an identity matrix.")
        self.assertEqual(b.name, None, "Node name should be None.")

    def test_initialise_with_transform(self):
        """Initialisation with a transform."""

        n = Node(transform = translate(1,2,3))

        self.assertEqual(n.parent, None, "Parent should be None.")
        self.assertEqual(n.root, n, "Node should be it's own root as it is not attached to a parent.")
        self.assertEqual(len(n.children), 0, "Child list should be empty.")
        self.assertTransformAlmostEqual(n.transform, translate(1,2,3), delta = 1e-14, msg = "Transform was not set correctly.")
        self.assertTransformAlmostEqual(n._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Root transform is incorrect.")
        self.assertTransformAlmostEqual(n._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Inverse root is incorrect.")
        self.assertEqual(n.name, None, "Node name should be None.")

    def test_initialise_with_parent_and_transform(self):
        """Initialisation with a parent and a transform."""

        a = Node()
        b = Node(a, translate(1,2,3))

        # node a
        self.assertEqual(a.parent, None, "Node a's parent should be None.")
        self.assertEqual(a.root, a, "Node a's root should be Node a.")
        self.assertEqual(a.children.count(b), 1, "Node a's child list should contain Node b.")
        self.assertTransformAlmostEqual(a.transform, AffineMatrix3D(), delta = 1e-14, msg ="Node a's transform should be an identity matrix.")
        self.assertTransformAlmostEqual(a._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Node a's root transform should be an identity matrix.")
        self.assertTransformAlmostEqual(a._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Node a's inverse root transform should be an identity matrix.")
        self.assertEqual(a.name, None, "Node name should be None.")

        # node b
        self.assertEqual(b.parent, a, "Node b's parent should be Node a.")
        self.assertEqual(b.root, a, "Node b's root should be Node a.")
        self.assertEqual(len(b.children), 0, "Node b's child list should be empty.")
        self.assertTransformAlmostEqual(b.transform, translate(1,2,3), delta = 1e-14, msg = "Node b's transform was not set correctly.")
        self.assertTransformAlmostEqual(b._root_transform, translate(1,2,3), delta = 1e-14, msg = "Node b's root transform is incorrect.")
        self.assertTransformAlmostEqual(b._root_transform_inverse, translate(1,2,3).inverse(), delta = 1e-14, msg = "Node b's inverse root transform is incorrect.")
        self.assertEqual(b.name, None, "Node name should be None.")

    def test_initialise_with_name(self):
        """Initialisation with a name."""

        r = Node(name="Test Node")
        self.assertEqual(r.name, "Test Node", "Node's name is incorrect.")

    def test_initialise_with_all_arguments(self):
        """Initialisation with all arguments."""

        a = Node()
        b = Node(a, translate(1,2,3), "My New Node")

        # node a
        self.assertEqual(a.parent, None, "Node a's parent should be None.")
        self.assertEqual(a.root, a, "Node a's root should be Node a.")
        self.assertEqual(a.children.count(b), 1, "Node a's child list should contain Node b.")
        self.assertTransformAlmostEqual(a.transform, AffineMatrix3D(), delta = 1e-14, msg ="Node a's transform should be an identity matrix.")
        self.assertTransformAlmostEqual(a._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Node a's root transform should be an identity matrix.")
        self.assertTransformAlmostEqual(a._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Node a's inverse root transform should be an identity matrix.")

        # node b
        self.assertEqual(b.parent, a, "Node b's parent should be Node a.")
        self.assertEqual(b.root, a, "Node b's root should be Node a.")
        self.assertEqual(len(b.children), 0, "Node b's child list should be empty.")
        self.assertTransformAlmostEqual(b.transform, translate(1,2,3), delta = 1e-14, msg = "Node b's transform was not set correctly.")
        self.assertTransformAlmostEqual(b._root_transform, translate(1,2,3), delta = 1e-14, msg = "Node b's root transform is incorrect.")
        self.assertTransformAlmostEqual(b._root_transform_inverse, translate(1,2,3).inverse(), delta = 1e-14, msg = "Node b's inverse root transform is incorrect.")
        self.assertEqual(b.name, "My New Node", "Node's name is incorrect.")

    def test_parent_set_invalid(self):
        """Setting parent with an invalid value."""

        n = Node()

        with self.assertRaises(TypeError, msg="Initialising with an invalid value (a string) did not raise a TypeError."):
            n.parent = "spoon"

    def test_parent_to_node_from_none(self):
        """Setting parent to a node from an initially unparented state."""

        # build initial tree
        a = Node()
        b = Node(a, translate(1,2,3))

        c = Node(transform = translate(10,20,30))
        d1 = Node(c, translate(100,200,300))
        d2 = Node(c, translate(200,400,600))
        e = Node(d1, translate(1000,2000,3000))

        # set c's parent to b
        c.parent = b

        # is parent correct?
        self.assertEqual(c.parent, b, "Node parent is incorrect.")

        # is c in b's child list?
        self.assertEqual(b.children.count(c), 1, "Node was not added to parents child list.")

        # is c.transform correct?
        self.assertTransformAlmostEqual(c.transform, translate(10,20,30), delta = 1e-14, msg = "Transform matrix should not have been modified by change of parent.")

        # is the root node correct?
        self.assertEqual(c.root, a, "Root node is incorrect.")

        # has the root transform been correctly propagated to the re-parented node and its children?
        self.assertTransformAlmostEqual(a._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Root node's root transform should not have been modified by change of parent.")
        self.assertTransformAlmostEqual(a._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Root node's inverse root transform should not have been modified by change of parent.")

        self.assertTransformAlmostEqual(b._root_transform, translate(1,2,3), delta = 1e-14, msg = "Parent's root transform should not have been modified by change of parent.")
        self.assertTransformAlmostEqual(b._root_transform_inverse, translate(1,2,3).inverse(), delta = 1e-14, msg = "Parent's inverse root transform should not have been modified by change of parent.")

        self.assertTransformAlmostEqual(c._root_transform, translate(11,22,33), delta = 1e-14, msg = "Root transform has not correctly propagated to re-parented node.")
        self.assertTransformAlmostEqual(c._root_transform_inverse, translate(11,22,33).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to re-parented node.")

        self.assertTransformAlmostEqual(d1._root_transform, translate(111,222,333), delta = 1e-14, msg = "Root transform has not correctly propagated to re-parented node's 1st immediate child.")
        self.assertTransformAlmostEqual(d1._root_transform_inverse, translate(111,222,333).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to re-parented node's 1st immediate child.")

        self.assertTransformAlmostEqual(d2._root_transform, translate(211,422,633), delta = 1e-14, msg = "Root transform has not correctly propagated to re-parented node's 2nd immediate child.")
        self.assertTransformAlmostEqual(d2._root_transform_inverse, translate(211,422,633).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to re-parented node's 2nd immediate child.")

        self.assertTransformAlmostEqual(e._root_transform, translate(1111,2222,3333), delta = 1e-14, msg = "Root transform has not correctly propagated to re-parented node's distant children.")
        self.assertTransformAlmostEqual(e._root_transform_inverse, translate(1111,2222,3333).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to re-parented node's distant children.")

    def test_parent_to_node_from_node(self):
        """Setting parent to a node from an initially parented state."""

        # build initial tree
        a = Node()
        b1 = Node(a, translate(1,2,3))
        b2 = Node(a, translate(2,4,6))
        c = Node(b1, translate(10,20,30))
        d1 = Node(c, translate(100,200,300))
        d2 = Node(c, translate(200,400,600))
        e = Node(d1, translate(1000,2000,3000))

        # set c's parent to b2
        c.parent = b2

        # is parent correct?
        self.assertEqual(c.parent, b2, "Node parent is incorrect.")

        # is c in b1's child list?
        self.assertEqual(b1.children.count(c), 0, "Node was not removed from previous parents child list.")

        # is c in b2's child list?
        self.assertEqual(b2.children.count(c), 1, "Node was not added to new parents child list.")

        # is c.transform correct?
        self.assertTransformAlmostEqual(c.transform, translate(10,20,30), delta = 1e-14, msg = "Transform matrix should not have been modified by change of parent.")

        # is the root node correct?
        self.assertEqual(c.root, a, "Root node is incorrect.")

        # has the root transform been correctly propagated to the re-parented node and its children?
        self.assertTransformAlmostEqual(a._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Root node's root transform should not have been modified by change of parent.")
        self.assertTransformAlmostEqual(a._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Root node's inverse root transform should not have been modified by change of parent.")

        self.assertTransformAlmostEqual(b1._root_transform, translate(1,2,3), delta = 1e-14, msg = "Previous parent's root transform should not have been modified by change of parent.")
        self.assertTransformAlmostEqual(b1._root_transform_inverse, translate(1,2,3).inverse(), delta = 1e-14, msg = "Previous parent's inverse root transform should not have been modified by change of parent.")

        self.assertTransformAlmostEqual(b2._root_transform, translate(2,4,6), delta = 1e-14, msg = "Parent's root transform should not have been modified by change of parent.")
        self.assertTransformAlmostEqual(b2._root_transform_inverse, translate(2,4,6).inverse(), delta = 1e-14, msg = "Parent's inverse root transform should not have been modified by change of parent.")

        self.assertTransformAlmostEqual(c._root_transform, translate(12,24,36), delta = 1e-14, msg = "Root transform has not correctly propagated to re-parented node.")
        self.assertTransformAlmostEqual(c._root_transform_inverse, translate(12,24,36).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to re-parented node.")

        self.assertTransformAlmostEqual(d1._root_transform, translate(112,224,336), delta = 1e-14, msg = "Root transform has not correctly propagated to re-parented node's 1st immediate child.")
        self.assertTransformAlmostEqual(d1._root_transform_inverse, translate(112,224,336).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to re-parented node's 1st immediate child.")

        self.assertTransformAlmostEqual(d2._root_transform, translate(212,424,636), delta = 1e-14, msg = "Root transform has not correctly propagated to re-parented node's 2nd immediate child.")
        self.assertTransformAlmostEqual(d2._root_transform_inverse, translate(212,424,636).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to re-parented node's 2nd immediate child.")

        self.assertTransformAlmostEqual(e._root_transform, translate(1112,2224,3336), delta = 1e-14, msg = "Root transform has not correctly propagated to re-parented node's distant children.")
        self.assertTransformAlmostEqual(e._root_transform_inverse, translate(1112,2224,3336).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to re-parented node's distant children.")

    def test_parent_to_none_from_node(self):
        """Unparenting a node."""

        # build initial tree
        a = Node()
        b = Node(a, translate(1,2,3))
        c = Node(b, translate(10,20,30))
        d1 = Node(c, translate(100,200,300))
        d2 = Node(c, translate(200,400,600))
        e = Node(d1, translate(1000,2000,3000))

        # set c's parent to None
        c.parent = None

        # is parent correct?
        self.assertEqual(c.parent, None, "Node parent is incorrect.")

        # is c in b's child list?
        self.assertEqual(b.children.count(c), 0, "Node was not removed from the previous parents child list.")

        # is c.transform correct?
        self.assertTransformAlmostEqual(c.transform, translate(10,20,30), delta = 1e-14, msg = "Transform matrix should not have been modified by change of parent.")

        # is the root node correct?
        self.assertEqual(c.root, c, "Root node is incorrect.")

        # has the root transform been correctly propagated to the re-parented node and its children?
        self.assertTransformAlmostEqual(a._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Root node's root transform should not have been modified by change of parent.")
        self.assertTransformAlmostEqual(a._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Root node's inverse root transform should not have been modified by change of parent.")

        self.assertTransformAlmostEqual(b._root_transform, translate(1,2,3), delta = 1e-14, msg = "Previous parent's root transform should not have been modified by change of parent.")
        self.assertTransformAlmostEqual(b._root_transform_inverse, translate(1,2,3).inverse(), delta = 1e-14, msg = "Previous parent's inverse root transform should not have been modified by change of parent.")

        self.assertTransformAlmostEqual(c._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Root transform is not correct for unparented node.")
        self.assertTransformAlmostEqual(c._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Inverse root transform is not correct for unparented node.")

        self.assertTransformAlmostEqual(d1._root_transform, translate(100,200,300), delta = 1e-14, msg = "Root transform has not correctly propagated to re-parented node's 1st immediate child.")
        self.assertTransformAlmostEqual(d1._root_transform_inverse, translate(100,200,300).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to re-parented node's 1st immediate child.")

        self.assertTransformAlmostEqual(d2._root_transform, translate(200,400,600), delta = 1e-14, msg = "Root transform has not correctly propagated to re-parented node's 2nd immediate child.")
        self.assertTransformAlmostEqual(d2._root_transform_inverse, translate(200,400,600).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to re-parented node's 2nd immediate child.")

        self.assertTransformAlmostEqual(e._root_transform, translate(1100,2200,3300), delta = 1e-14, msg = "Root transform has not correctly propagated to re-parented node's distant children.")
        self.assertTransformAlmostEqual(e._root_transform_inverse, translate(1100,2200,3300).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to re-parented node's distant children.")

    def test_parent_enforce_tree(self):
        """Prevention of cyclic parenting."""

        # build test tree
        a = Node()
        b = Node(a)
        c1 = Node(b)
        c2 = Node(b)

        # test tree at different depths to confirm resursive check succeeds
        with self.assertRaises(ValueError, msg = "Illegal cyclic parenting (a -> a) did not raise an exception."):
            a.parent = a

        with self.assertRaises(ValueError, msg = "Illegal cyclic parenting (a -> b) did not raise an exception."):
            a.parent = b

        with self.assertRaises(ValueError, msg = "Illegal cyclic parenting (a -> c2) did not raise an exception."):
            a.parent = c2

        with self.assertRaises(ValueError, msg = "Illegal cyclic parenting (b -> c1) did not raise an exception."):
            b.parent = c1

    def test_transform(self):
        """Setting the Node transform."""

        # build initial tree
        a = Node()
        b = Node(a, translate(1,2,3))
        c = Node(b, translate(10,20,30))
        d1 = Node(c, translate(100,200,300))
        d2 = Node(c, translate(200,400,600))
        e = Node(d1, translate(1000,2000,3000))

        # define new transform
        m = translate(20,40,60)

        # set c's transform
        c.transform = m

        # is c.transform correct?
        self.assertTransformAlmostEqual(c.transform, translate(20,40,60), delta = 1e-14, msg = "Transform matrix was set correctly.")

        # have the root transforms been correctly propagated to this nodes children, parent nodes should be unaffected
        self.assertTransformAlmostEqual(a._root_transform, AffineMatrix3D(), delta = 1e-14, msg ="Root node's root transform should not have been modified by change of transform.")
        self.assertTransformAlmostEqual(a._root_transform_inverse, AffineMatrix3D(), delta = 1e-14, msg ="Root node's inverse root transform should not have been modified by change of transform.")

        self.assertTransformAlmostEqual(b._root_transform, translate(1,2,3), delta = 1e-14, msg = "Parent's root transform should not have been modified by change of transform.")
        self.assertTransformAlmostEqual(b._root_transform_inverse, translate(1,2,3).inverse(), delta = 1e-14, msg = "Parent's inverse root transform should not have been modified by change of transform.")

        self.assertTransformAlmostEqual(c._root_transform, translate(21,42,63), delta = 1e-14, msg = "Root transform has not correctly propagated for node c.")
        self.assertTransformAlmostEqual(c._root_transform_inverse, translate(21,42,63).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated for node c.")

        self.assertTransformAlmostEqual(d1._root_transform, translate(121,242,363), delta = 1e-14, msg = "Root transform has not correctly propagated to node's 1st immediate child.")
        self.assertTransformAlmostEqual(d1._root_transform_inverse, translate(121,242,363).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to node's 1st immediate child.")

        self.assertTransformAlmostEqual(d2._root_transform, translate(221,442,663), delta = 1e-14, msg = "Root transform has not correctly propagated to node's 2nd immediate child.")
        self.assertTransformAlmostEqual(d2._root_transform_inverse, translate(221,442,663).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to node's 2nd immediate child.")

        self.assertTransformAlmostEqual(e._root_transform, translate(1121,2242,3363), delta = 1e-14, msg = "Root transform has not correctly propagated to node's distant children.")
        self.assertTransformAlmostEqual(e._root_transform_inverse, translate(1121,2242,3363).inverse(), delta = 1e-14, msg = "Inverse root transform has not correctly propagated to node's distant children.")

    def test_transform_set_invalid(self):
        """Setting the Node transform with an invalid value."""

        n = Node()

        with self.assertRaises(TypeError, msg="Setting transform with an invalid value (a string) did not raise a TypeError."):
            n.transform = "spoon"

        with self.assertRaises(TypeError, msg="Setting transform with an invalid value (None) did not raise a TypeError."):
            n.transform = None

    def test_name(self):
        """Setting and getting the Node name."""

        n = Node()
        n.name = "Spongle"

        self.assertEqual(n.name, "Spongle", "Node name was not set correctly.")

    def test_name_set_invalid(self):
        """Setting Node name to an invalid value."""

        n = Node()

        with self.assertRaises(TypeError, msg="Setting Node name with an invalid value (None) did not raise a TypeError."):
            n.name = None

        with self.assertRaises(TypeError, msg="Setting transform with an invalid value (a float) did not raise a TypeError."):
            n.name = 57.1

    def test_to(self):
        """Test the to() method returns a matrix transform between nodes.

        The returned affine matrix should transform a point in the coordinate
        space defined by the calling node to the coordinate space defined by the
        specified node in the tree."""

        # build test tree
        root = Node()
        a1 = Node(root, translate(1,0,0))
        a2 = Node(a1, translate(10,0,0))
        b1 = Node(root, translate(0,1,0))
        b2 = Node(b1, translate(0,10,0))
        c = Node(root, translate(0,0,1))

        # test a2 to root
        self.assertTransformAlmostEqual(a2.to(root), translate(11,0,0), delta = 1e-14, msg="The a2.to(root) transform is incorrect.")

        # test b2 to root
        self.assertTransformAlmostEqual(b2.to(root), translate(0,11,0), delta = 1e-14, msg="The b2.to(root) transform is incorrect.")

        # test a2 to b2
        self.assertTransformAlmostEqual(a2.to(b2), translate(11,-11,0), delta = 1e-14, msg="The a2.to(b2) transform is incorrect.")

        # test b2 to a2
        self.assertTransformAlmostEqual(b2.to(a2), translate(-11,11,0), delta = 1e-14, msg="The b2.to(a2) transform is incorrect.")

        # test a2 to c
        self.assertTransformAlmostEqual(a2.to(c), translate(11,0,-1), delta = 1e-14, msg="The a2.to(c) transform is incorrect.")

        # test c to b1
        self.assertTransformAlmostEqual(c.to(b1), translate(0,-1,1), delta = 1e-14, msg="The c.to(b1) transform is incorrect.")

    def test_to_invalid(self):
        """Calling the to() method with invalid arguements."""

        # two nodes, not linked so each forms a seperate scenegraph
        a = Node()
        b = Node()

        with self.assertRaises(ValueError, msg="Attempting to find a transform between two independent scenegraphs should have raised a ValueError exception."):
            a.to(b)

        with self.assertRaises(TypeError, msg="Passing an invalid arguement (a string) did not raise a TypeError."):
            a.to("bad input")

    def test_to_local(self):
        """to_local should return a matrix transform from root to local space."""

        # build test tree
        root = Node()
        a1 = Node(root, translate(1,0,0))
        a2 = Node(a1, translate(10,0,0))
        b1 = Node(root, translate(0,1,0))
        b2 = Node(b1, translate(0,10,0))
        c = Node(root, translate(0,0,1))

        # test a2 from root to local
        self.assertTransformAlmostEqual(a2.to_local(), translate(-11,0,0), delta = 1e-14, msg="The a2.to_local transform is incorrect.")

        # test b2 from root to local
        self.assertTransformAlmostEqual(b2.to_local(), translate(0,-11,0), delta = 1e-14, msg="The b2.to_local transform is incorrect.")

    def test_to_root(self):
        """to_root should return a matrix transform from local to root space."""

        # build test tree
        root = Node()
        a1 = Node(root, translate(1,0,0))
        a2 = Node(a1, translate(10,0,0))
        b1 = Node(root, translate(0,1,0))
        b2 = Node(b1, translate(0,10,0))
        c = Node(root, translate(0,0,1))

        # test a2 from local to root
        self.assertTransformAlmostEqual(a2.to_root(), translate(11,0,0), delta = 1e-14, msg="The a2.to_root transform is incorrect.")

        # test b2 from local to root
        self.assertTransformAlmostEqual(b2.to_root(), translate(0,11,0), delta = 1e-14, msg="The b2.to_root transform is incorrect.")


if __name__ == "__main__":
    unittest.main()