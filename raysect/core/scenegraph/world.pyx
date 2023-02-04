# cython: language_level=3

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

from raysect.core.scenegraph.signal import GEOMETRY

from raysect.core.acceleration.kdtree cimport KDTree
from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.scenegraph.observer cimport Observer
from raysect.core.scenegraph.signal cimport ChangeSignal


cdef class World(_NodeBase):
    """
    The root node of the scene-graph.

    The world node tracks all primitives and observers in the world. It maintains acceleration structures to speed up
    the ray-tracing calculations. The particular acceleration algorithm used is selectable. The default acceleration
    structure is a kd-tree.

    :param name: A string defining the node name.
    """

    def __init__(self, str name=None):

        super().__init__(name)

        self._primitives = list()
        self._observers = list()
        self._rebuild_accelerator = True
        self._accelerator = KDTree()

    @property
    def accelerator(self):
        """
        The acceleration structure used for this world's scene-graph.
        """
        return self._accelerator

    @accelerator.setter
    def accelerator(self, Accelerator accelerator not None):
        self._accelerator = accelerator
        self._rebuild_accelerator = True

    @property
    def name(self):
        """
        The name for this world node.

        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, str value not None):
        self._name = value

    @property
    def primitives(self):
        """
        The list of primitives maintained in this scene-graph.

        :rtype: list
        """
        return self._primitives

    @property
    def observers(self):
        """
        The list of observers in this scene-graph.

        :rtype: list
        """
        return self._observers

    cpdef AffineMatrix3D to(self, _NodeBase node):
        """
        Returns an affine transform that, when applied to a vector or point,
        transforms the vector or point from the co-ordinate space of the calling
        node to the co-ordinate space of the target node.

        For example, if space B is translated +100 in x compared to space A and
        A.to(B) is called then the matrix returned would represent a translation
        of -100 in x. Applied to point (0,0,0) in A, this would produce the
        point (-100,0,0) in B as B is translated +100 in x compared to A.

        :param Node node: The target node.
        :return: An AffineMatrix3D describing the coordinate transform.
        :rtype: AffineMatrix3D
        """

        if self.root is node.root:
            return node._root_transform_inverse
        else:
            raise ValueError("The target node must be in the same scene-graph.")

    # TODO - like hit() on primitive, is there a better name?
    cpdef Intersection hit(self, Ray ray):
        """
        Calculates the closest intersection of the Ray with the Primitives in
        the scene-graph, if such an intersection exists.

        If a hit occurs an Intersection object is returned which contains the
        mathematical details of the intersection. None is returned if the ray
        does not intersect any primitive.

        This method automatically rebuilds the Acceleration object that is used
        to optimise hit calculations - if a Primitive's geometry or a transform
        affecting a primitive has changed since the last call to hit() or
        contains(), the Acceleration structure used to optimise hit calculations
        is rebuilt to represent the new scene-graph state.

        :param Ray ray: The ray to test.
        :return: An Intersection object or None if no intersection occurs.
        :rtype: Intersection
        """

        self.build_accelerator()
        return self._accelerator.hit(ray)

    # TODO - better name - world.primitives_containing(point)
    cpdef list contains(self, Point3D point):
        """
        Returns a list of Primitives that contain the specified point within
        their surface.

        An empty list is returned if no Primitives contain the Point3D.

        This method automatically rebuilds the Acceleration object that is used
        to optimise the contains calculation - if a Primitive's geometry or a
        transform affecting a primitive has changed since the last call to hit()
        or contains(), the Acceleration structure used to optimise the contains
        calculation is rebuilt to represent the new scene-graph state.

        :param Point3D point: The point to test.
        :return: A list containing all Primitives that enclose the Point3D.
        :rtype: list
        """

        self.build_accelerator()
        return self._accelerator.contains(point)

    cpdef build_accelerator(self, bint force=False):
        """
        This method manually triggers a rebuild of the Acceleration object.

        If the Acceleration object is already in a consistent state this method
        will do nothing unless the force keyword option is set to True.

        The Acceleration object is used to accelerate hit() and contains()
        calculations, typically using a spatial sub-division method. If changes are
        made to the scene-graph structure, transforms or to a primitive's
        geometry the acceleration structures may no longer represent the
        geometry of the scene and hence must be rebuilt. This process is
        usually performed automatically as part of the first call to hit() or
        contains() following a change in the scene-graph. As calculating these
        structures can take some time, this method provides the option of
        triggering a rebuild outside of hit() and contains() in case the user wants
        to be able to perform a benchmark without including the overhead of the
        Acceleration object rebuild.

        :param bool force: If set to True, forces rebuilding of acceleration structure.
        """

        if self._rebuild_accelerator or force:
            self._accelerator.build(self._primitives)
            self._rebuild_accelerator = False

    def _register(self, _NodeBase node):
        """
        Adds observers and primitives to the World's object tracking lists.
        """

        if isinstance(node, Primitive):
            self._primitives.append(node)
            self._rebuild_accelerator = True

        if isinstance(node, Observer):
            self._observers.append(node)

    def _deregister(self, _NodeBase node):
        """
        Removes observers and primitives from the World's object tracking lists.
        """

        if isinstance(node, Primitive):
            self._primitives.remove(node)
            self._rebuild_accelerator = True

        if isinstance(node, Observer):
            self._observers.remove(node)

    def _change(self, _NodeBase node, ChangeSignal change not None):
        """
        Notifies the World of a change to the scene-graph.

        This method must be called is a change occurs that may have invalidated
        any acceleration structures held by the World.

        The node on which the change occurs and a ChangeSignal must be
        provided. The ChangeSignal must specify the nature of the change to the
        scene-graph.

        The core World object only recognises the GEOMETRY signal. When a
        GEOMETRY signal is received, the world will be instructed to rebuild
        it's spatial acceleration structures on the next call to any method
        that interacts with the scene-graph geometry.
        """

        if change is GEOMETRY:
            self._rebuild_accelerator = True

