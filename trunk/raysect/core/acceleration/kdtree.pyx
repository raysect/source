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

# TODO: this can be heavily optimised, a job for the future

from raysect.core.scenegraph.primitive cimport Primitive
from libc.math cimport log, ceil
cimport cython

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
        """
        Rebuilds the KDTree acceleration structure with the list of primitives.
        """

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

    @cython.boundscheck(False)
    cpdef Intersection hit(self, Ray ray):
        """
        Returns the first intersection with a primitive or None of no primitive
        is intersected.
        """

        cdef:
            tuple intersection
            double min_range, max_range
            bint hit

        # unpacking manually is marginally faster... but a pointless None check
        # is still performed which I will need to talk to the cython developers
        # about... we need a cython.nocheck(True) option
        intersection = self.world_box.full_intersection(ray)
        hit = intersection[0]
        min_range = intersection[1]
        max_range = intersection[2]

        if not hit:

            return None

        return self.tree.hit(ray, min_range, max_range)

    cpdef list contains(self, Point point):
        """
        Returns a list of primitives that contain the point.
        """

        if not self.world_box.contains(point):

            return []

        return self.tree.contains(point)

    cdef list _accelerate_primitives(self, list primitives):
        """
        Wraps the list of Primitives with AcceleratedPrimitives.
        """

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
        """
        Builds a bounding box that encloses all the supplied primitives .
        """

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

    cdef object build(self, BoundingBox node_bounds, list primitives, int depth, int min_primitives, double hit_cost):

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
                if edge.value > node_bounds.lower.get_index(axis) and edge.value < node_bounds.upper.get_index(axis):

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

                if primitive.box.lower.get_index(best_axis) < best_split:

                    lower_primitives.append(primitive)

                if primitive.box.upper.get_index(best_axis) > best_split:

                    upper_primitives.append(primitive)

                if primitive.box.lower.get_index(best_axis) == best_split and primitive.box.upper.get_index(best_axis) == best_split:

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

            edges.append(Edge(primitive, primitive.box.lower.get_index(axis), is_upper_edge=False))
            edges.append(Edge(primitive, primitive.box.upper.get_index(axis), is_upper_edge=True))

        edges.sort()

        return edges

    cdef BoundingBox _calc_lower_bounds(self, BoundingBox node_bounds, double split_value, int axis):

        cdef Point upper

        upper = node_bounds.upper.copy()
        upper.set_index(axis, split_value)

        return BoundingBox(node_bounds.lower.copy(), upper)

    cdef BoundingBox _calc_upper_bounds(self, BoundingBox node_bounds, double split_value, int axis):

        cdef Point lower

        lower = node_bounds.lower.copy()
        lower.set_index(axis, split_value)

        return BoundingBox(lower, node_bounds.upper.copy())

    cdef Intersection hit(self, Ray ray, double min_range, double max_range):

        cdef:
            double origin, direction
            Intersection lower_intersection, upper_intersection, intersection
            double plane_distance
            Node near, far

        if self.is_leaf:

            # find the closest primitive-ray intersection
            return self._hit_leaf(ray, max_range)

        else:

            origin = ray.origin.get_index(self.axis)
            direction = ray.direction.get_index(self.axis)

            if direction == 0:

                # ray propagating parallel to split plane
                if origin < self.split:

                    # only need to consider the lower node
                    return self.lower_branch.hit(ray, min_range, max_range)

                elif origin > self.split:

                    # only need to consider the upper node
                    return self.upper_branch.hit(ray, min_range, max_range)

                else:

                    # ray origin sitting in split plane
                    lower_intersection = self.lower_branch.hit(ray, min_range, max_range)
                    upper_intersection = self.upper_branch.hit(ray, min_range, max_range)

                    if lower_intersection is None:

                            return upper_intersection

                    else:

                        if upper_intersection is None:

                            return lower_intersection

                        else:

                            if lower_intersection.ray_distance < upper_intersection.ray_distance:

                                return lower_intersection

                            else:

                                return upper_intersection

            else:

                # ray propagation not parallel to split plane
                with cython.cdivision(True):

                    plane_distance = (self.split - origin) / direction

                if origin < self.split or (origin == self.split and direction < 0):

                    near = self.lower_branch
                    far = self.upper_branch

                else:

                    near = self.upper_branch
                    far = self.lower_branch

                if plane_distance > max_range or plane_distance < 0:

                    # only intersects the near node
                    return near.hit(ray, min_range, max_range)

                if plane_distance < min_range:

                    # only intersects the far node
                    return far.hit(ray, min_range, max_range)

                # intersects both nodes
                intersection = near.hit(ray, min_range, plane_distance)

                if intersection is not None:

                    return intersection

                intersection = far.hit(ray, plane_distance, max_range)

                return intersection

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline Intersection _hit_leaf(self, Ray ray, double max_range):

        cdef:
            double distance
            Intersection intersection, closest_intersection
            AcceleratedPrimitive primitive

        # find the closest primitive-ray intersection
        closest_intersection = None

        # intial search distance limited by node and ray limits
        distance = min(ray.max_distance, max_range)

        for primitive in self.primitives:

            intersection = primitive.hit(ray)

            if intersection is not None:

                if intersection.ray_distance < distance:

                    distance = intersection.ray_distance
                    closest_intersection = intersection

        return closest_intersection

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list contains(self, Point point):

        cdef:
            AcceleratedPrimitive primitive
            list enclosing_primitives
            double location

        if self.is_leaf:

            enclosing_primitives = []
            for primitive in self.primitives:

                if primitive.contains(point):

                    enclosing_primitives.append(primitive.primitive)

            return enclosing_primitives

        else:

            location = point.get_index(self.axis)

            if location < self.split:

                return self.lower_branch.contains(point)

            if location > self.split:

                return self.upper_branch.contains(point)

            else:

                # on split plane this should be relatively rare
                enclosing_primitives = self.lower_branch.contains(point)
                enclosing_primitives.extend(self.upper_branch.contains(point))

                # ensure primitives are unique in list
                return list(set(enclosing_primitives))


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
