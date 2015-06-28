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

# TODO: this can be heavily optimised by packing data into a cache aligned c-structure, a job for the future

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
        self.tree = _Node()

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

        # wrap primitives with their bounding boxes
        primitives = self._bound_primitives(primitives)

        # construct a bounding box that contains all the primitive in the world
        self._build_world_box(primitives)

        # default max tree depth is set to the value suggested in "Physically Based Rendering From Theory to
        # Implementation 2nd Edition", Matt Phar and Greg Humphreys, Morgan Kaufmann 2010, p232
        if self.max_depth <= 0:
            max_depth = <int> ceil(8 + 1.3 * log(len(primitives)))
        else:
            max_depth = self.max_depth

        # calling build on the root node triggers a recursive rebuild of the tree
        self.tree.build(self.world_box, primitives, max_depth, self.min_primitives, self.hit_cost)

    @cython.boundscheck(False)
    cpdef Intersection hit(self, Ray ray):
        """
        Returns the first intersection with a primitive or None if no primitive
        is intersected.
        """

        cdef:
            tuple intersection
            double min_range, max_range
            bint hit

        # unpacking manually is marginally faster...
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

    cdef list _bound_primitives(self, list primitives):
        """
        Wraps each primitive in the list with a BoundPrimitive object.

        A bound primitive is a primitive that is wrapped by its world space bounding box.
        """

        cdef:
            Primitive primitive
            BoundPrimitive bound_primitive
            list bound_primitives

        # wrap primitives with their bounding boxes
        bound_primitives = []
        for primitive in primitives:
            bound_primitive = BoundPrimitive(primitive)
            bound_primitives.append(bound_primitive)

        return bound_primitives

    cdef void _build_world_box(self, list primitives):
        """
        Builds a bounding box that encloses all the supplied primitives.
        """

        cdef:
            BoundPrimitive primitive
            BoundingBox box

        self.world_box = BoundingBox()
        for primitive in primitives:
            self.world_box.union(primitive.box)


cdef class _Edge:
    """
    Represents the upper or lower edge of a bounding box on a specified axis.
    """

    def __init__(self, BoundPrimitive primitive, int axis, bint is_upper_edge):

        self.primitive = primitive
        self.is_upper_edge = is_upper_edge

        if is_upper_edge:
            self.value = primitive.box.upper.get_index(axis)
        else:
            self.value = primitive.box.lower.get_index(axis)

    def __richcmp__(_Edge x, _Edge y, int operation):

        if operation == 0:  # __lt__(), less than
            # lower edge must always be encountered first
            # break tie by ensuring lower edge sorted before upper edge
            if x.value == y.value:
                if x.is_upper_edge:
                    return False
                else:
                    return True
            return x.value < y.value
        else:
            return NotImplemented


cdef class _Node:

    def __init__(self):

        self.lower_branch = None
        self.upper_branch = None
        self.primitives = []
        self.axis = NO_AXIS
        self.split = 0
        self.is_leaf = False

    cdef object build(self, BoundingBox node_bounds, list primitives, int depth, int min_primitives, double hit_cost):

        cdef:
            double cost, best_cost, split, best_split
            double recip_total_sa, lower_sa, upper_sa
            int best_axis
            int lower_primitive_count, upper_primitive_count
            list edges, lower_primitives, upper_primitives
            _Edge edge
            BoundPrimitive primitive

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

                # a split on the node boundary serves no useful purpose
                # only consider edges that lie inside the node bounds
                split = edge.value
                if node_bounds.lower.get_index(axis) < split < node_bounds.upper.get_index(axis):

                    # calculate surface area of split volumes
                    lower_sa = self._calc_lower_bounds(node_bounds, split, axis).surface_area()
                    upper_sa = self._calc_upper_bounds(node_bounds, split, axis).surface_area()

                    # calculate SAH cost
                    cost = 1 + (lower_sa * lower_primitive_count + upper_sa * upper_primitive_count) * recip_total_sa * hit_cost

                    # has a better split been found?
                    if cost < best_cost:
                        best_cost = cost
                        best_axis = axis
                        best_split = split

                # update primitive counts for lower volume
                # note: this occasionally creates invalid solutions if edges of
                # boxes are coincident however the invalid solutions cost
                # more than the valid solutions and will not be selected
                if not edge.is_upper_edge:
                    lower_primitive_count += 1

        # no better solution found?
        if best_axis == NO_AXIS:
            self._become_leaf(primitives)
            return

        # using cached values split primitive into two lists
        # note the split boundary is defined as lying in the upper node
        lower_primitives = []
        upper_primitives = []
        for primitive in primitives:

            # is the box present in the lower node?
            if primitive.box.lower.get_index(best_axis) < best_split:
                lower_primitives.append(primitive)

            # is the box present in the upper node?
            if primitive.box.upper.get_index(best_axis) >= best_split:
                upper_primitives.append(primitive)

        # become a branch node
        self.lower_branch = _Node()
        self.upper_branch = _Node()
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
            BoundPrimitive primitive

        edges = []
        for primitive in primitives:
            edges.append(_Edge(primitive, axis, is_upper_edge=False))
            edges.append(_Edge(primitive, axis, is_upper_edge=True))
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

        if self.is_leaf:
            return self._hit_leaf(ray, max_range)
        else:
            return self._hit_branch(ray, min_range, max_range)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline Intersection _hit_leaf(self, Ray ray, double max_range):
        """
        Find the closest primitive-ray intersection.
        """

        cdef:
            double distance
            Intersection intersection, closest_intersection
            BoundPrimitive primitive

        # find the closest primitive-ray intersection with intial search distance limited by node and ray limits
        closest_intersection = None
        distance = min(ray.max_distance, max_range)
        for primitive in self.primitives:
            intersection = primitive.hit(ray)
            if intersection is not None and intersection.ray_distance <= distance:
                distance = intersection.ray_distance
                closest_intersection = intersection
        return closest_intersection

    @cython.cdivision(True)
    cdef inline Intersection _hit_branch(self, Ray ray, double min_range, double max_range):

        cdef:
            double origin, direction
            Intersection lower_intersection, upper_intersection, intersection
            double plane_distance
            _Node near, far

        origin = ray.origin.get_index(self.axis)
        direction = ray.direction.get_index(self.axis)

        # is the ray propagating parallel to the split plane?
        if direction == 0:

            # a ray propagating parallel to the split plane will only ever interact with one of the nodes
            if origin < self.split:
                return self.lower_branch.hit(ray, min_range, max_range)
            else:
                return self.upper_branch.hit(ray, min_range, max_range)

        else:

            # ray propagation is not parallel to split plane
            plane_distance = (self.split - origin) / direction

            # identify the order in which the ray will interact with the nodes
            if origin < self.split:
                near = self.lower_branch
                far = self.upper_branch
            elif origin > self.split:
                near = self.upper_branch
                far = self.lower_branch
            else:
                # degenerate case, note split plane lives in upper branch
                if direction >= 0:
                    near = self.upper_branch
                    far = self.lower_branch
                else:
                    near = self.lower_branch
                    far = self.upper_branch

            # does ray only intersect with the near node?
            if plane_distance > max_range or plane_distance <= 0:
                return near.hit(ray, min_range, max_range)

            # does ray only intersect with the far node?
            if plane_distance < min_range:
                return far.hit(ray, min_range, max_range)

            # ray must intersect both nodes, try nearest node first
            intersection = near.hit(ray, min_range, plane_distance)
            if intersection is not None:
                return intersection

            intersection = far.hit(ray, plane_distance, max_range)
            return intersection

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list contains(self, Point point):

        cdef:
            BoundPrimitive primitive
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
            else:
                return self.upper_branch.contains(point)


