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

from raysect.core.math cimport Point3D
from raysect.core.math.spatial.kdtree3d cimport Item3D
from raysect.core.scenegraph cimport Primitive
from raysect.core.ray cimport Ray
from raysect.core.acceleration.boundprimitive cimport BoundPrimitive
from libc.stdint cimport int32_t
cimport cython


cdef class _PrimitiveKDTree(_KDTreeCore):

    def __init__(self, list primitives, int max_depth=0, int min_items=1, double hit_cost=80.0, double empty_bonus=0.2):

        cdef:
            Primitive primitive
            BoundPrimitive bound_primitive
            int32_t id
            list items

        # wrap each primitive with its bounding box
        self.primitives = [BoundPrimitive(primitive) for primitive in primitives]

        # kd-Tree init requires the primitives's id (it's index here) and bounding box
        items = [Item3D(id, bound_primitive.box) for id, bound_primitive in enumerate(self.primitives)]
        super().__init__(items, max_depth, min_items, hit_cost, empty_bonus)

        self.hit_intersection = None

    def __getstate__(self):
        return self.primitives, super().__getstate__()

    def __setstate__(self, state):
        self.primitives, super_state = state
        super().__setstate__(super_state)
        self.hit_intersection = None

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _trace_leaf(self, int32_t id, Ray ray, double max_range):
        """
        Tests each item in the kd-Tree leaf node to identify if an intersection occurs.

        This is a virtual method and must be implemented in a derived class if
        ray intersections are to be identified. This method must return True
        if an intersection is found and False otherwise.

        Derived classes may need to return information about the intersection.
        This can be done by setting object attributes prior to returning True.
        The kd-Tree search algorithm stops as soon as the first leaf is
        identified that contains an intersection. Any attributes set when
        _trace_leaf() returns True are guaranteed not to be further modified.

        :param id: Index of node in node array.
        :param ray: Ray object.
        :param max_range: The maximum intersection search range.
        :return: True is an intersection occurs, false otherwise.
        """

        cdef:
            int32_t count, item, index
            double distance
            Intersection intersection, closest_intersection
            BoundPrimitive primitive

        # unpack leaf data
        count = self._nodes[id].count

        # find the closest primitive-ray intersection with initial search distance limited by node and ray limits
        distance = min(ray.max_distance, max_range)
        closest_intersection = None
        for item in range(count):

            # dereference the primitive
            index = self._nodes[id].items[item]
            primitive = <BoundPrimitive> self.primitives[index]

            # test for intersection
            intersection = primitive.hit(ray)
            if intersection is not None and intersection.ray_distance <= distance:
                distance = intersection.ray_distance
                closest_intersection = intersection

        if closest_intersection is None:
            self.hit_intersection = None
            return False

        self.hit_intersection = closest_intersection
        return True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list _items_containing_leaf(self, int32_t id, Point3D point):
        """
        Tests each item in the node to identify if they enclose the point.

        This is a virtual method and must be implemented in a derived class if
        the identification of items enclosing a point is required. This method
        must return a list of ids for the items that enclose the point. If no
        items enclose the point, an empty list must be returned.

        Derived classes may need to wish to return additional information about
        the enclosing items. This can be done by setting object attributes
        prior to returning the list. Any attributes set when
        _items_containing_leaf() returns are guaranteed not to be further
        modified.

        :param id: Index of node in node array.
        :param point: Point3D to evaluate.
        :return: List of items containing the point.
        """

        cdef:
            int32_t count, item
            list enclosing_primitives
            BoundPrimitive primitive

        # unpack leaf data
        count = self._nodes[id].count

        # dereference the primitives and check if they contain the point
        enclosing_primitives = []
        for item in range(count):

            index = self._nodes[id].items[item]
            primitive = <BoundPrimitive> self.primitives[index]
            if primitive.contains(point):
                enclosing_primitives.append(primitive.primitive)

        return enclosing_primitives


cdef class KDTree(_Accelerator):

    cpdef build(self, list primitives):
        self._kdtree = _PrimitiveKDTree(primitives)

    cpdef Intersection hit(self, Ray ray):

        # we explicitly use _trace() rather than trace() as _trace() is cdef, rather than cpdef
        if self._kdtree._trace(ray):
            return self._kdtree.hit_intersection
        return None

    cpdef list contains(self, Point3D point):

        # we explicitly use _items_containing() rather than items_containing() as _items_containing is cdef, rather than cpdef
        return self._kdtree._items_containing(point)