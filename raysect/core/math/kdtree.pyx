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

from random import random
from raysect.core.acceleration.boundingbox cimport BoundingBox
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.math cimport log, ceil
cimport cython

# this number of nodes will be pre-allocated when the kd-tree is initially created
DEF INITIAL_NODE_COUNT = 128

# node types
DEF LEAF = -1
DEF X_AXIS = 0  # branch, x-axis split
DEF Y_AXIS = 1  # branch, y-axis split
DEF Z_AXIS = 2  # branch, z-axis split


# c-structure that represent a kd-tree node
cdef struct kdnode:

    int type        # LEAF, X_AXIS, Y_AXIS, Z_AXIS
    double split
    int count       # upper index (BRANCH_*), item count (LEAF)
    int *items


cdef class Item:
    """
    Item class. Represents an item to place into the kd-tree.

    The id should be a unique integer value identifying an external object.
    For example the id could be the index into an array of polygon objects.
    The id values are stored in the kd-tree and returned by the hit() or
    contains() methods.

    A bounding box associated with the item defines the spatial extent of the
    item along each axis. This data is used to place the items in the tree.

    :param id: An integer item id.
    :param box: A BoundingBox object defining the item's spatial extent.
    """

    cdef:
        readonly int id
        readonly BoundingBox box

    def __init__(self, int id, BoundingBox box):

        self.id = id
        self.box = box


cdef class KDTree:

    cdef:
        kdnode *_nodes
        int _allocated_nodes
        int _next_node
        readonly BoundingBox bounds
        int _max_depth
        int _min_items
        double _hit_cost
        double _empty_bonus

    def __cinit__(self):

        self._nodes = NULL
        self._allocated_nodes = 0
        self._next_node = 0

    def __init__(self, items, max_depth=0, min_items=1, hit_cost=20.0, empty_bonus=0.2):

        # sanity check
        if empty_bonus < 0.0 or empty_bonus > 1.0:
            raise ValueError("The empty_bonus cost modifier must lie in the range [0.0, 1.0].")
        self._empty_bonus = empty_bonus

        # clamp other parameters
        self._max_depth = max(0, max_depth)
        self._min_items = max(1, min_items)
        self._hit_cost = max(1.0, hit_cost)

        # if max depth is set to zero, automatically calculate a reasonable depth
        # tree depth is set to the value suggested in "Physically Based Rendering From Theory to
        # Implementation 2nd Edition", Matt Phar and Greg Humphreys, Morgan Kaufmann 2010, p232
        if self._max_depth == 0:
            self._max_depth = <int> ceil(8 + 1.3 * log(len(items)))

        # calculate kd-tree bounds
        self.bounds = BoundingBox()
        for item in items:
            self.bounds.union(item.box)

        # start build with the longest axis to try to avoid large narrow nodes
        axis = self.bounds.largest_axis()
        self.build(axis, items, self.bounds, 0)

    def debug_print_all(self):

        for i in range(self._next_node):
            self.node_info(i)

    def debug_print_node(self, id):

        if 0 <= id < self._next_node:
            if self._nodes[id].type == LEAF:
                print("id={} LEAF: count {}, contents: [".format(id, self._nodes[id].count), end="")
                for i in range(self._nodes[id].count):
                    print("{}".format(self._nodes[id].items[i]), end="")
                    if i < self._nodes[id].count - 1:
                        print(", ", end="")
                print("]")
            else:
                print("id={} BRANCH: axis {}, split {}, count {}".format(id, self._nodes[id].type, self._nodes[id].split, self._nodes[id].count))

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cdef int build(self, int axis, list items, BoundingBox bounds, int depth):

        # cdef:
        #     int axis
        #     tuple result
        #     double split
        #     list lower_triangle_data, upper_triangle_data

        if depth == 0 or len(items) <= self._min_items:
            return self._new_leaf(items)

        # attempt to identify a suitable node split
        split_solution = self._split(axis, items, bounds)

        # split solution found?
        if split_solution is None:
            return self._new_leaf(items)
        else:
            return self._new_branch(axis, split_solution, depth)

    # TODO: write _split
    # TODO: write _hit
    # TODO: write _contains


    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    # cdef tuple _split(self, int axis, list triangle_data, BoundingBox node_bounds, double hit_cost):
    #
    #     cdef:
    #         double split, cost, best_cost, best_split
    #         bint is_leaf
    #         int lower_triangle_count, upper_triangle_count
    #         double recip_total_sa, lower_sa, upper_sa
    #         list edges, lower_triangle_data, upper_triangle_data
    #         _Edge edge
    #         _TriangleData data
    #
    #     # store cost of leaf as current best solution
    #     best_cost = len(triangle_data) * hit_cost
    #     best_split = 0
    #     is_leaf = True
    #
    #     # cache reciprocal of node's surface area
    #     recip_total_sa = 1.0 / node_bounds.surface_area()
    #
    #     # obtain sorted list of candidate edges along chosen axis
    #     edges = self._build_edges(triangle_data, axis)
    #
    #     # cache triangle counts in lower and upper volumes for speed
    #     lower_triangle_count = 0
    #     upper_triangle_count = len(triangle_data)
    #
    #     # scan through candidate edges from lowest to highest
    #     for edge in edges:
    #
    #         # update primitive counts for upper volume
    #         # note: this occasionally creates invalid solutions if edges of
    #         # boxes are coincident however the invalid solutions cost
    #         # more than the valid solutions and will not be selected
    #         if edge.is_upper_edge:
    #             upper_triangle_count -= 1
    #
    #         # a split on the node boundary serves no useful purpose
    #         # only consider edges that lie inside the node bounds
    #         split = edge.value
    #         if node_bounds.lower.get_index(axis) < split < node_bounds.upper.get_index(axis):
    #
    #             # calculate surface area of split volumes
    #             lower_sa = self._calc_lower_bounds(node_bounds, split, axis).surface_area()
    #             upper_sa = self._calc_upper_bounds(node_bounds, split, axis).surface_area()
    #
    #             # calculate SAH cost
    #             cost = 1 + (lower_sa * lower_triangle_count + upper_sa * upper_triangle_count) * recip_total_sa * hit_cost
    #
    #             # has a better split been found?
    #             if cost < best_cost:
    #                 best_cost = cost
    #                 best_split = split
    #                 is_leaf = False
    #
    #         # update triangle counts for lower volume
    #         # note: this occasionally creates invalid solutions if edges of
    #         # boxes are coincident however the invalid solutions cost
    #         # more than the valid solutions and will not be selected
    #         if not edge.is_upper_edge:
    #             lower_triangle_count += 1
    #
    #     if is_leaf:
    #         return None
    #
    #     # using cached values split triangles into two lists
    #     # note the split boundary is defined as lying in the upper node
    #     lower_triangle_data = []
    #     upper_triangle_data = []
    #     for data in triangle_data:
    #
    #         # is the triangle present in the lower node?
    #         if data.lower_extent[axis] < best_split:
    #             lower_triangle_data.append(data)
    #
    #         # is the triangle present in the upper node?
    #         if data.upper_extent[axis] >= best_split:
    #             upper_triangle_data.append(data)
    #
    #
    #   TODO: add bounds calculations here
    #
    #     return best_split, lower_triangle_data, upper_triangle_data

    cdef int _new_node(self):

        cdef:
            kdnode *new_nodes = NULL
            int id, new_size

        # have we exhausted the allocated memory?
        if self._next_node == self._allocated_nodes:

            # double allocated memory
            new_size = max(INITIAL_NODE_COUNT, self._allocated_nodes * 2)
            new_nodes = <kdnode *> PyMem_Realloc(self._nodes, sizeof(kdnode) * new_size)
            if not new_nodes:
                raise MemoryError()

            self._nodes = new_nodes
            self._allocated_nodes = new_size

        id = self._next_node
        self._next_node += 1
        return id

    cdef int _new_leaf(self, list items):

        cdef:
            int id, count, i

        count = len(items)

        id = self._new_node()
        self._nodes[id].type = LEAF
        self._nodes[id].count = count
        if count >= 0:
            self._nodes[id].items = <int *> PyMem_Malloc(sizeof(int) * count)
            if not self._nodes[id].items:
                raise MemoryError()

            for i in range(count):
                self._nodes[id].items[i] = items[i]

        return id

    cdef int _new_branch(self, int axis, tuple split_solution, int depth):

        cdef:
            int id, upper_id
            int next_axis
            double split
            list lower_items, upper_items
            BoundingBox lower_bounds,  upper_bounds

        id = self._new_node()

        # cycle to the next axis
        next_axis = (axis + 1) % 3

        # unpack split solution
        split, lower_items, lower_bounds, upper_items, upper_bounds = split_solution

        # recursively build lower and upper nodes
        # the lower node is always the next node in the list
        # the upper node may be an arbitrary distance along the list
        # we store the upper node id in count for future evaluation
        self.build(next_axis, lower_items, lower_bounds, depth + 1)
        upper_id = self.build(next_axis, upper_items, upper_bounds, depth + 1)

        # WARNING: Don't "optimise" this code by writing self._nodes[id].count = self._build(...)
        # it appears that the self._nodes[id] is de-referenced *before* the call to _build() and
        # subsequent assignment to count. If a realloc occurs during the execution of the build
        # call, the de-referenced address will become stale and access with cause a segfault.
        # This was a forking *NIGHTMARE* to debug!
        self._nodes[id].count = upper_id
        self._nodes[id].type = axis
        self._nodes[id].split = split

        return id

    def __dealloc__(self):

        cdef:
            int index
            kdnode *node

        # free all leaf node item arrays
        for index in range(self._next_node):
            if self._nodes[index].type == LEAF and self._nodes[index].count > 0:
                PyMem_Free(self._nodes[index].items)

        # free the nodes
        PyMem_Free(self._nodes)


