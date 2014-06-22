# cython: language_level=3

# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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

from raysect.core.acceleration.kdtree cimport KDTree
from raysect.core.scenegraph.primitive cimport Primitive

cdef class World(_NodeBase):

    def __init__(self, unicode name not None = ""):
        """
        World constructor.
        """

        super().__init__()

        self._name = name
        self. _primitives = list()
        self. _rebuild_accelerator = True
        self._accelerator = KDTree()

    def __str__(self):
        """String representation."""

        if self._name == "":

            return "<World at " + str(hex(id(self))) + ">"

        else:

            return self._name + " <World at " + str(hex(id(self))) + ">"

    cpdef AffineMatrix to(self, _NodeBase node):
        """
        Returns an affine transform that, when applied to a vector or point,
        transforms the vector or point from the co-ordinate space of the calling
        node to the co-ordinate space of the target node.

        For example, if space B is translated +100 in x compared to space A and
        A.to(B) is called then the matrix returned would represent a translation
        of -100 in x. Applied to point (0,0,0) in A, this would produce the
        point (-100,0,0) in B as B is translated +100 in x compared to A.
        """

        if self.root is node.root:

            return node._root_transform_inverse

        else:

            raise ValueError("The target node must be in the same scenegraph.")

    cpdef Intersection hit(self, Ray ray):
        """
        Calculates the closest intersection of the Ray with the Primitives in
        the scenegraph, if such an intersection exists.

        If a hit occurs an Intersection object is returned which contains the
        mathematical details of the intersection. None is returned if the ray
        does not intersect any primitive.

        This method automatically rebuilds the Acceleration object that is used
        to optimise hit calculations - if a Primitive's geometry or a transform
        affecting a primitive has changed since the last call to hit() or
        contains(), the Acceleration structure used to optimise hit calculations
        is rebuilt to represent the new scenegraph state.
        """

        self.build_accelerator()
        return self._accelerator.hit(ray)

    cpdef list contains(self, Point point):
        """
        Returns a list of Primitives that contain the specified point within
        their surface.

        An empty list is returned if no Primitives contain the Point.

        This method automatically rebuilds the Acceleration object that is used
        to optimise the contains calculation - if a Primitive's geometry or a
        transform affecting a primitive has changed since the last call to hit()
        or contains(), the Acceleration structure used to optimise the contains
        calculation is rebuilt to represent the new scenegraph state.
        """

        self.build_accelerator()
        return self._accelerator.contains(point)

    cpdef build_accelerator(self):
        """
        This method manually triggers a rebuild of the Acceleration object.

        If the Acceleration object is already in a consistent state this method
        will do nothing.

        The Acceleration object is used to accelerate hit() and contains()
        calculations, typically using a spatial subdivion method. If changes are
        made to the scenegraph structure, transforms or to a primitive's
        geometry the acceleration structures may no longer represent the
        geometry of the scene and hence must be rebuilt. This process is
        usually performed automatically as part of the first call to hit() or
        contains() following a change in the scenegraph. As calculating these
        structures can take some time, this method provides the option of
        triggering a rebuild outside of hit() and contains() incase the user wants
        to be able to benchmark without including the overhead of the
        Acceleration object rebuild.
        """

        if self._rebuild_accelerator:

            self._accelerator.build(self._primitives)
            self._rebuild_accelerator = False

    def _register(self, _NodeBase node):
        """Adds primitives to the World's primitive list."""

        if isinstance(node, Primitive):

            self._primitives.append(node)
            self._rebuild_accelerator = True

    def _deregister(self, _NodeBase node):
        """Removes primitives from the World's primitive list."""

        if isinstance(node, Primitive):

            self._primitives.remove(node)
            self._rebuild_accelerator = True

    def _change(self, _NodeBase node):
        """
        Alerts the world that a change to the scenegraph has occurred that could
        have made the acceleration structure no longer a valid representation
        of the scenegaph geometry.
        """

        self._rebuild_accelerator = True

    property accelerator:

        def __get__(self):

            return self._accelerator

        def __set__(self, Accelerator accelerator not None):

            self._accelerator = accelerator
            self._rebuild_accelerator = True

    property name:

        def __get__(self):

            return self._name

        def __set__(self, unicode value not None):

            self._name = value

