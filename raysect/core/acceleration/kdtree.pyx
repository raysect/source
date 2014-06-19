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

from raysect.core.scenegraph.primitive cimport Primitive
from libc.math cimport log, ceil

DEF X_AXIS = 0
DEF Y_AXIS = 1
DEF Z_AXIS = 2
DEF NO_AXIS = 3


cdef class KDTree(Accelerator):

    def __init__(self):

        self.world_box = BoundingBox()
        self.tree = Node()


        self.max_depth = -1
        self.min_primitives = 1

        # TODO: calculate or measure this, relative cost of a primitive hit
        # calculation compared to a kdtree split traversal
        self.hit_cost = 80.0

    cpdef build(self, list primitives):

        cdef int max_depth

        # wrap primitves with their bounding boxes
        primitives = self._accelerate_primitives(primitives)

        # construct a bounding box that contains all the primitive in the world
        self._build_world_box(primitives)

        if self.max_depth <= 0:

            # set max tree depth to value suggested in "Physically Based Rendering
            # From Theory to Implementation 2nd Edition", Matt Phar and Greg Humphreys,
            # Morgan Kaufmann 2010, p232
            max_depth = <int> ceil(8 + 1.3 * log(len(primitives)))

        else:

            max_depth = self.max_depth

        # calling build on the root node triggers a recursive rebuild of the tree
        self.tree.build(self.world_box, primitives, max_depth, self.min_primitives, self.hit_cost)

    cpdef Intersection hit(self, Ray ray):

        if not self.world_box.hit(ray):

            return None

        return self.tree.hit(ray)

    cpdef list contains(self, Point point):

        if not self.world_box.contains(point):

            return []

        return self.tree.contains(point)

    cdef list _accelerate_primitives(self, list primitives):

        cdef:
            Primitive primitive
            AcceleratedPrimitive accel_primitive
            list accel_primitives

        accel_primitives = []

        for primitive in primitives:

            # wrap primitive with it's bounding box
            accel_primitive = AcceleratedPrimitive(primitive)
            accel_primitives.append(accel_primitive)

        return accel_primitives

    cdef void _build_world_box(self, list primitives):

        cdef:
            AcceleratedPrimitive primitive
            BoundingBox box

        self.world_box = BoundingBox()

        for primitive in primitives:

            self.world_box.union(primitive.box)


cdef class Node:

    def __init__(self):

        self.lower_branch = None
        self.upper_branch = None
        self.primitives = []
        self.axis = NO_AXIS
        self.split = 0
        self.is_leaf = False

    cpdef object build(self, BoundingBox node_bounds, list primitives, int depth, int min_primitives, double hit_cost):

        cdef:
            double cost, best_cost, best_split
            double recip_total_sa, lower_sa, upper_sa
            int best_axis
            int lower_primitive_count, upper_primitive_count
            list edges, lower_primitives, upper_primitives
            Edge edge
            AcceleratedPrimitive primitive

        if depth == 0 or len(primitives) < min_primitives:

            self._become_leaf(primitives)
            return

        # store cost of leaf as current best solution
        best_cost = len(primitives) * hit_cost
        best_axis = NO_AXIS
        best_split = 0

        # cache reciprocal of node's surface area
        recip_total_sa = 1.0 / node_bounds.surface_area()

        # attempt splits along each axis to attempt to find a lower cost solution
        for axis in [X_AXIS, Y_AXIS, Z_AXIS]:

            # obtain sorted list of candidate edges along chosen axis
            edges = self._build_edges(primitives, axis)

            # cache primitive counts in lower and upper volumes for speed
            lower_primitive_count = 0
            upper_primitive_count = len(primitives)

            # scan through candidate edges from lowest to highest
            for edge in edges:

                # update primitive counts for upper volume
                # note: this occasionally creates invalid solutions if edges of
                # boxes are coincident however the invalid solutions cost
                # more than the valid solutions and will not be selected
                if edge.is_upper_edge:

                    upper_primitive_count -= 1

                # only consider edge if it lies inside the node bounds
                if edge.value > node_bounds.lower[axis] and edge.value < node_bounds.upper[axis]:

                    # calculate surface area of split volumes
                    lower_sa = self._calc_lower_bounds(node_bounds, edge.value, axis).surface_area()
                    upper_sa = self._calc_upper_bounds(node_bounds, edge.value, axis).surface_area()

                    # calculate SAH cost
                    cost = 1 + (lower_sa * lower_primitive_count + upper_sa * upper_primitive_count) * recip_total_sa * hit_cost

                    if cost < best_cost:

                        best_cost = cost
                        best_axis = axis
                        best_split = edge.value

                # update primitive counts for lower volume
                # note: this occasionally creates invalid solutions if edges of
                # boxes are coincident however the invalid solutions cost
                # more than the valid solutions and will not be selected
                if not edge.is_upper_edge:

                    lower_primitive_count += 1

        if best_axis == NO_AXIS:

            # no better solution found
            self._become_leaf(primitives)

        else:

            lower_primitives = []
            upper_primitives = []

            # using cached values split primitive into two lists
            for primitive in primitives:

                if primitive.box.lower[best_axis] < best_split:

                    lower_primitives.append(primitive)

                if primitive.box.upper[best_axis] > best_split:

                    upper_primitives.append(primitive)

                if primitive.box.lower[best_axis] == best_split and primitive.box.upper[best_axis] == best_split:

                    # an infinitesimally thin box should never happen, but just for safety
                    lower_primitives.append(primitive)
                    upper_primitives.append(primitive)

            # become a branch node
            self.lower_branch = Node()
            self.upper_branch = Node()
            self.primitives = None
            self.axis = best_axis
            self.split = best_split
            self.is_leaf = False

            # continue expanding the tree inside the two volumes
            self.lower_branch.build(self._calc_lower_bounds(node_bounds, best_split, best_axis),
                                    lower_primitives, depth - 1, min_primitives, hit_cost)

            self.upper_branch.build(self._calc_upper_bounds(node_bounds, best_split, best_axis),
                                    upper_primitives, depth - 1, min_primitives, hit_cost)

    cdef void _become_leaf(self, list primitives):

        self.lower_branch = None
        self.upper_branch = None
        self.primitives = primitives
        self.axis = NO_AXIS
        self.split = 0
        self.is_leaf = True

    cdef list _build_edges(self, list primitives, int axis):

        cdef:
            list edges
            AcceleratedPrimitive primitive

        edges = []
        for primitive in primitives:

            edges.append(Edge(primitive, primitive.box.lower[axis], is_upper_edge=False))
            edges.append(Edge(primitive, primitive.box.upper[axis], is_upper_edge=True))

        edges.sort()

        return edges

    cdef BoundingBox _calc_lower_bounds(self, BoundingBox node_bounds, double split_value, int axis):

        cdef Point upper

        upper = node_bounds.upper.copy()
        upper[axis] = split_value

        return BoundingBox(node_bounds.lower.copy(), upper)

    cdef BoundingBox _calc_upper_bounds(self, BoundingBox node_bounds, double split_value, int axis):

        cdef Point lower

        lower = node_bounds.lower.copy()
        lower[axis] = split_value

        return BoundingBox(lower, node_bounds.upper.copy())

    cpdef Intersection hit(self, Ray ray):



        return None

    cpdef list contains(self, Point point):

        return []


cdef class Edge:

    def __init__(self, primitive, value, is_upper_edge):

        self.primitive = primitive
        self.value = value
        self.is_upper_edge = is_upper_edge

    def __richcmp__(Edge x, Edge y, int operation):

        if operation == 0:  # __lt__(), less than

            # break tie by ensuring lower edge sorted before upper edge
            if x.value == y.value:

                # lower edge must always be encountered first
                if x.is_upper_edge:

                    return False

                else:

                    return True

            return x.value < y.value

        else:

            return NotImplemented
