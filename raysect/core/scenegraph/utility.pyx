# cython: language_level=3

# Copyright (c) 2014-2016, Dr Alex Meakins, Raysect Project
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

from raysect.core.math.affinematrix cimport AffineMatrix3D
from raysect.core.scenegraph._nodebase cimport _NodeBase
from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.scenegraph.node cimport Node
from raysect.core.boundingbox cimport BoundingBox3D
from raysect.core.classes cimport Material


cdef class BridgeNode(Node):
    """
    Specialised scene-graph root node that propagates geometry notifications.
    """

    def __init__(self, Primitive owner):

        super().__init__()
        self.owner = owner

    def _change(self, _NodeBase node):
        """
        Handles a scenegraph node change handler.

        Propagates geometry change notifications to the enclosing primitive and
        it's scenegraph.
        """

        # propagate geometry change notification from local scene-graph to owner's scene-graph
        self.owner.root._change(self.owner)


# TODO: docstrings
# TODO: move to raysect.primitive.utility
cdef class EncapsulatedPrimitive(Primitive):
    """
    allows developers to hide primitive attributes from users

    where the primitive dimensions are defined by a wrapper e.g. CSG biconvex lens - two spheres and a cylinder with dimensiosn defineds by blah blah...

    can only be used to encapsulate a single primitive, any attached children will be removed automatically
    (they would be ignored anyway)

    :param Primitive:
    :return:
    """

    def __init__(self, Primitive primitive not None, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):

        super().__init__(parent, transform, material, name)

        # a bridge node connects the internal scene-graph to the main scene-graph
        # and forwards geometry change notifications
        self._localroot = BridgeNode(self)
        self._primitive = primitive

        # attach the primitive to the local (encapsulated) scene-graph
        self._primitive.parent = self._localroot

        # mirror the coordinate space transforms of the main scene-graph
        # internally so that the encapsulated primitive's transforms are
        # passed the correct transform matrices
        self._primitive.transform = self.to_root()

        # disconnect any children attached to the primitive
        for child in self._primitive.children:
            child.parent = None

    def __str__(self):
        """String representation."""

        if self.name:
            return self.name + " <EncapsulatedPrimitive at {}>".format(str(hex(id(self))))
        else:
            return "<EncapsulatedPrimitive at {}>".format(str(hex(id(self))))

    def _modified(self):

        # update the local transform to mirror the transform of the main scene-graph
        self._primitive.transform = self.to_root()

    cpdef Intersection hit(self, Ray ray):

        cdef Intersection intersection

        # pass hit calculation on to "hidden" primitive
        intersection = self._primitive.hit(ray)

        # the intersection will reference the internal primitive it must be
        # modified to point at the enclosing primitive
        if intersection is not None:
            intersection.primitive = self

        return intersection

    cpdef Intersection next_intersection(self):
        return self._primitive.next_intersection()

    cpdef bint contains(self, Point3D p) except -1:
        return self._primitive.contains(p)

    cpdef BoundingBox3D bounding_box(self):
        return self._primitive.bounding_box()


def print_scenegraph(node):
    """
    Pretty-prints a scene-graph.

    This function will print the scene-graph that contains the specified node.
    The specified node will be highlighted in the tree by post-fixing the node
    with the string: "[referring node]".

    :param _NodeBase node: The target node.
    """

    # start from root node
    root = node.root

    # print node
    if root is node:
        print(str(root) + " [referring node]")
    else:
        print(str(root))

    # print children
    n = len(root.children)
    for i in range(0, n):
        if i < (n-1):
            _print_node(root.children[i], "", " |  ", node)
        else:
            _print_node(root.children[i], "", "    ", node)


def _print_node(node, indent, link, highlight):
    """
    Internal function called recursively to print a scene-graph.
    """

    # print node
    print(indent + " |  ")

    if node is highlight:
        print(indent + " |_ " + str(node) + " [referring node]")
    else:
        print(indent + " |_ " + str(node))

    # print children
    n = len(node.children)
    for i in range(0, n):
        if i < (n-1):
            _print_node(node.children[i], indent + link, " |  ", highlight)
        else:
            _print_node(node.children[i], indent + link, "    ", highlight)