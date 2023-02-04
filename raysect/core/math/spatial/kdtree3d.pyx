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

import io
import struct

from raysect.core.boundingbox cimport new_boundingbox3d
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.bytes cimport PyBytes_AsString
from libc.stdlib cimport qsort
from libc.stdint cimport int32_t
from libc.math cimport log, ceil
cimport cython

# this number of nodes will be pre-allocated when the kd-tree is initially created
DEF INITIAL_NODE_COUNT = 128

# friendly name for first node
DEF ROOT_NODE = 0

# node types
DEF LEAF = -1    # leaf node
DEF X_AXIS = 0  # branch, x-axis split
DEF Y_AXIS = 1  # branch, y-axis split
DEF Z_AXIS = 2  # branch, z-axis split


cdef class Item3D:
    """
    Item3D class. Represents an item to place into the kd-tree.

    The id should be a unique integer value identifying an external object.
    For example the id could be the index into an array of polygon objects.
    The id values are stored in the kd-tree and returned by the hit() or
    contains() methods.

    A bounding box associated with the item defines the spatial extent of the
    item along each axis. This data is used to place the items in the tree.

    :param id: An integer item id.
    :param box: A BoundingBox3D object defining the item's spatial extent.
    """

    def __init__(self, int32_t id, BoundingBox3D box):

        self.id = id
        self.box = box


cdef int _edge_compare(const void *p1, const void *p2) nogil:

    cdef edge e1, e2

    e1 = (<edge *> p1)[0]
    e2 = (<edge *> p2)[0]

    # lower edge must always be encountered first
    # break tie by ensuring lower extent sorted before upper edge
    if e1.value == e2.value:
        if e2.is_upper_edge:
            return -1
        else:
            return 1

    if e1.value < e2.value:
        return -1
    else:
        return 1


cdef class KDTree3DCore:
    """
    Implements a 3D kd-tree for items with finite extents.

    This is a Cython abstract base class. It cannot be directly extended in
    Python due to the need to implement cdef methods _items_containing_leaf() and
     _trace_leaf(). Use the KDTree3D wrapper class if extending from Python.

    :param items: A list of Items.
    :param max_depth: The maximum tree depth (automatic if set to 0, default is 0).
    :param min_items: The item count threshold for forcing creation of a new leaf node (default 1).
    :param hit_cost: The relative computational cost of item hit evaluations vs kd-tree traversal (default 20.0).
    :param empty_bonus: The bonus applied to node splits that generate empty leaves (default 0.2).
    """

    def __cinit__(self):

        self._nodes = NULL
        self._allocated_nodes = 0
        self._next_node = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, list items, int32_t max_depth=0, int32_t min_items=1, double hit_cost=20.0, double empty_bonus=0.2):

        cdef:
            Item3D item

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
            self._max_depth = <int32_t> ceil(8 + 1.3 * log(len(items)))

        # calculate kd-tree bounds
        self.bounds = BoundingBox3D()
        for item in items:
            self.bounds.union(item.box)

        # start build
        self._build(items, self.bounds)

    def __getstate__(self):
        state = io.BytesIO()
        self.save(state)
        return state.getvalue()

    def __setstate__(self, state):
        self.load(io.BytesIO(state))

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    cdef int32_t _build(self, list items, BoundingBox3D bounds, int32_t depth=0):
        """
        Extends the kd-Tree by creating a new node.

        Attempts to partition space for efficient traversal.

        :param items: A list of items.
        :param bounds: A BoundingBox3D defining the node bounds.
        :param depth: The current tree depth.
        :return: The id (index) of the generated node.
        """

        if depth == self._max_depth or len(items) <= self._min_items:
            return self._new_leaf(items)

        # attempt to identify a suitable node split
        split_solution = self._split(items, bounds)

        # split solution found?
        if split_solution is None:
            return self._new_leaf(items)
        else:
            return self._new_branch(split_solution, depth)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef tuple _split(self, list items, BoundingBox3D bounds):
        """
        Attempts to locate a split solution that minimises the cost of traversing the node.

        The cost of the node traversal is evaluated using the Surface Area Heuristic (SAH) method.

        :param items: A list of items.
        :param bounds: A BoundingBox3D defining the node bounds.
        :return: A tuple containing the split solution or None if a split solution is not found.
        """

        cdef:
            double split, bonus, cost
            bint is_leaf
            int32_t longest_axis, axis,
            double best_cost, best_split
            int32_t best_axis
            edge *edges = NULL
            int32_t index, num_edges
            int32_t lower_count, upper_count
            double recip_total_sa, lower_sa, upper_sa
            list lower_items, upper_items
            Item3D item

        # store cost of leaf as current best solution
        best_cost = len(items) * self._hit_cost
        best_split = 0
        best_axis = -1
        is_leaf = True

        # cache reciprocal of node's surface area
        recip_total_sa = 1.0 / bounds.surface_area()

        # search for a solution along the longest axis first
        # if a split isn't found, then try the other axes
        longest_axis = bounds.largest_axis()
        for axis in [longest_axis, (longest_axis + 1) % 3, (longest_axis +2) % 3]:

            # obtain sorted list of candidate edges along chosen axis
            self._get_edges(items, axis, &num_edges, &edges)
            # print(edges[num_edges - 1])

            # cache item counts in lower and upper volumes for speed
            lower_count = 0
            upper_count = len(items)

            # scan through candidate edges from lowest to highest
            for index in range(num_edges):

                # update item counts for upper volume
                # note: this occasionally creates invalid solutions if edges of
                # boxes are coincident however the invalid solutions cost
                # more than the valid solutions and will not be selected
                if edges[index].is_upper_edge:
                    upper_count -= 1

                # a split on the node boundary serves no useful purpose
                # only consider edges that lie inside the node bounds
                split = edges[index].value
                if bounds.lower.get_index(axis) < split < bounds.upper.get_index(axis):

                    # calculate surface area of split volumes
                    lower_sa = self._get_lower_bounds(bounds, split, axis).surface_area()
                    upper_sa = self._get_upper_bounds(bounds, split, axis).surface_area()

                    # is there an empty bonus?
                    bonus = 1.0
                    if lower_count == 0 or upper_count == 0:
                        bonus -= self._empty_bonus

                    # calculate SAH cost
                    cost = 1 + bonus * (lower_sa * lower_count + upper_sa * upper_count) * recip_total_sa * self._hit_cost

                    # has a better split been found?
                    if cost < best_cost:
                        best_cost = cost
                        best_split = split
                        best_axis = axis
                        is_leaf = False

                # update item counts for lower volume
                # note: this occasionally creates invalid solutions if edges of
                # boxes are coincident however the invalid solutions cost
                # more than the valid solutions and will not be selected
                if not edges[index].is_upper_edge:
                    lower_count += 1

            # clean up edges memory
            self._free_edges(&edges)

            # stop searching through axes if we have found a reasonable split solution
            if not is_leaf:
                break

        if is_leaf:
            return None

        # using cached values split items into two lists
        # note the split boundary is defined as lying in the upper node
        lower_items = []
        upper_items = []
        for item in items:

            # is the item present in the lower node?
            if item.box.lower.get_index(best_axis) < best_split:
                lower_items.append(item)

            # is the item present in the upper node?
            if item.box.upper.get_index(best_axis) > best_split:
                upper_items.append(item)

        # construct bounding boxes that enclose the lower and upper nodes
        lower_bounds = self._get_lower_bounds(bounds, best_split, best_axis)
        upper_bounds = self._get_upper_bounds(bounds, best_split, best_axis)

        return best_axis, best_split, lower_items, lower_bounds, upper_items, upper_bounds

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _get_edges(self, list items, int32_t axis, int32_t *num_edges, edge **edges_ptr):
        """
        Generates a sorted list of edges along the specified axis.

        :param items: A list of items.
        :param axis: The axis to split along.
        :param num_edges: Pointer to number of edges (returned).
        :param edges_ptr: Pointer to array of edges (returned).
        """

        cdef:
            int32_t count, index, lower_index, upper_index
            Item3D item
            edge *edges

        # allocate edge array
        count = len(items) * 2
        edges = <edge *> PyMem_Malloc(sizeof(edge) * count)
        if not edges:
            raise MemoryError()

        # populate
        for index, item in enumerate(items):

            lower_index = 2 * index
            upper_index = lower_index + 1

            # lower edge
            edges[lower_index].is_upper_edge = False
            edges[lower_index].value = item.box.lower.get_index(axis)

            # upper edge
            edges[upper_index].is_upper_edge = True
            edges[upper_index].value = item.box.upper.get_index(axis)

        # sort
        qsort(<void *> edges, count, sizeof(edge), _edge_compare)

        # return
        num_edges[0] = count
        edges_ptr[0] = edges

    cdef void _free_edges(self, edge **edges_ptr):
        """
        Free allocated edge array.

        :param edges_ptr: Pointer to array of edges.
        """

        PyMem_Free(edges_ptr[0])

    cdef BoundingBox3D _get_lower_bounds(self, BoundingBox3D bounds, double split, int32_t axis):
        """
        Returns the lower box generated when the node bounding box is split.

        :param bounds: A BoundingBox3D defining the node bounds.
        :param split: The value along the axis at which to split.
        :param axis: The axis to split along.
        :return: A bounding box defining the lower bounds.
        """

        cdef Point3D upper
        upper = bounds.upper.copy()
        upper.set_index(axis, split)
        return new_boundingbox3d(bounds.lower.copy(), upper)

    cdef BoundingBox3D _get_upper_bounds(self, BoundingBox3D bounds, double split, int32_t axis):
        """
        Returns the upper box generated when the node bounding box is split.

        :param bounds: A BoundingBox3D defining the node bounds.
        :param split: The value along the axis at which to split.
        :param axis: The axis to split along.
        :return: A bounding box defining the upper bounds.
        """

        cdef Point3D lower
        lower = bounds.lower.copy()
        lower.set_index(axis, split)
        return new_boundingbox3d(lower, bounds.upper.copy())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int32_t _new_leaf(self, list items):
        """
        Adds a new leaf node to the kd-Tree and populates it.

        :param items: The items to add to the leaf node.
        :return: The id (index) of the generated node.
        """

        cdef int32_t id, count, index

        count = len(items)

        id = self._new_node()
        self._nodes[id].type = LEAF
        self._nodes[id].count = count
        if count > 0:
            self._nodes[id].items = <int32_t *> PyMem_Malloc(sizeof(int32_t) * count)
            if not self._nodes[id].items:
                raise MemoryError()

            for index in range(count):
                self._nodes[id].items[index] = (<Item3D> items[index]).id

        return id

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int32_t _new_branch(self, tuple split_solution, int32_t depth):
        """
        Adds a new branch node to the kd-Tree and populates it.

        :param split_solution: A tuple containing the split solution.
        :param depth: The current tree depth.
        :return: The id (index) of the generated node.
        """

        cdef:
            int32_t id, upper_id
            int32_t axis
            double split
            list lower_items, upper_items
            BoundingBox3D lower_bounds,  upper_bounds

        id = self._new_node()

        # unpack split solution
        axis, split, lower_items, lower_bounds, upper_items, upper_bounds = split_solution

        # recursively build lower and upper nodes
        # the lower node is always the next node in the list
        # the upper node may be an arbitrary distance along the list
        # we store the upper node id in count for future evaluation
        self._build(lower_items, lower_bounds, depth + 1)
        upper_id = self._build(upper_items, upper_bounds, depth + 1)

        # WARNING: Don't "optimise" this code by writing self._nodes[id].count = self._build(...)
        # it appears that the self._nodes[id] is de-referenced *before* the call to _build() and
        # subsequent assignment to count. If a realloc occurs during the execution of the build
        # call, the de-referenced address will become stale and access with cause a segfault.
        # This was a forking *NIGHTMARE* to debug!
        self._nodes[id].count = upper_id
        self._nodes[id].type = axis
        self._nodes[id].split = split

        return id

    cdef int32_t _new_node(self):
        """
        Adds a new, empty node to the kd-Tree.

        :return: The id (index) of the generated node.
        """

        cdef:
            kdnode *new_nodes = NULL
            int32_t id, new_size

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

    cpdef bint is_contained(self, Point3D point):
        """
        Traverses the kd-Tree to identify if the point is contained by an any item.

        :param point: A Point3D object.
        :return: True if the point lies inside an item, false otherwise.
        """

        return self._is_contained(point)

    cdef bint _is_contained(self, Point3D point):
        """
        Starts contains traversal of the kd-Tree.

        :param point: A Point3D object.
        :return: True if the point lies inside an item, false otherwise.
        """

        # exit early if point is not inside bounds of the kd-Tree
        if not self.bounds.contains(point):
            return False

        # start search
        return self._is_contained_node(ROOT_NODE, point)

    cdef bint _is_contained_node(self, int32_t id, Point3D point):
        """
        Dispatches contains point look-ups to the relevant node handler.

        :param id: Index of node in node array.
        :param point: Point3D to evaluate.
        :return: True if the point lies inside an item, false otherwise.
        """

        if self._nodes[id].type == LEAF:
            return self._is_contained_leaf(id, point)
        else:
            return self._is_contained_branch(id, point)

    cdef bint _is_contained_branch(self, int32_t id, Point3D point):
        """
        Locates the kd-Tree node containing the point.

        :param id: Index of node in node array.
        :param point: Point3D to evaluate.
        :return: True if the point lies inside an item, false otherwise.
        """

        cdef:
            int32_t axis
            double split
            int32_t lower_id, upper_id

        # unpack branch kdnode
        # notes:
        #  * the branch type enumeration is the same as axis index
        #  * the lower_id is always the next node in the array
        #  * the upper_id is stored in the count attribute
        axis = self._nodes[id].type
        split = self._nodes[id].split
        lower_id = id + 1
        upper_id = self._nodes[id].count

        if point.get_index(axis) < split:
            return self._is_contained_node(lower_id, point)
        else:
            return self._is_contained_node(upper_id, point)

    cdef bint _is_contained_leaf(self, int32_t id, Point3D point):
        """
        Tests each item in the node to identify if they enclose the point.

        This is a virtual method and must be implemented in a derived class if
        the identification of an item enclosing a point is required. This method
        must return True is the point lies inside an item or False otherwise.

        Derived classes may need to wish to return additional information about
        the enclosing item(s). This can be done by setting object attributes
        prior to returning. Any attributes set when _is_contained_leaf() returns
        are guaranteed not to be further modified.

        :param id: Index of node in node array.
        :param point: Point3D to evaluate.
        :return: True if the point lies inside an item, false otherwise.
        """

        # virtual function that must be implemented by derived classes
        raise NotImplementedError("KDTree3DCore _is_contained_leaf() method not implemented.")

    cpdef bint trace(self, Ray ray):
        """
        Traverses the kd-Tree to find the first intersection with an item stored in the tree.

        This method returns True is an item is hit and False otherwise.

        :param ray: A Ray object.
        :return: True is an intersection occurs, false otherwise.
        """

        return self._trace(ray)

    cdef bint _trace(self, Ray ray):
        """
        Starts the ray traversal of the kd tree.

        :param ray: A Ray object.
        :return: True is a hit occurs, false otherwise.
        """

        cdef:
            bint hit
            double min_range, max_range

        # check tree bounds
        hit = self.bounds.intersect(ray, &min_range, &max_range)
        if not hit:
            return False

        # start exploration of kd-Tree
        return self._trace_node(ROOT_NODE, ray, min_range, max_range)

    cdef bint _trace_node(self, int32_t id, Ray ray, double min_range, double max_range):
        """
        Dispatches trace calculation to the relevant node handler.

        :param id: Index of node in node array.
        :param ray: Ray object.
        :param min_range: The minimum intersection search range.
        :param max_range: The maximum intersection search range.
        :return: True is a hit occurs, false otherwise.
        """

        if self._nodes[id].type == LEAF:
            return self._trace_leaf(id, ray, max_range)
        else:
            return self._trace_branch(id, ray, min_range, max_range)

    @cython.cdivision(True)
    cdef bint _trace_branch(self, int32_t id, Ray ray, double min_range, double max_range):
        """
        Traverses a kd-Tree branch node along the ray path.

        :param id: Index of node in node array.
        :param ray: Ray object.
        :param min_range: The minimum intersection search range.
        :param max_range: The maximum intersection search range.
        :return: True is a hit occurs, false otherwise.
        """

        cdef:
            int32_t axis
            double split
            bint below_split
            int32_t lower_id, upper_id
            double origin, direction
            double plane_distance
            int32_t near_id, far_id
            bint hit

        # unpack branch kdnode
        # notes:
        #  * the branch type enumeration is the same as axis index
        #  * the lower_id is always the next node in the array
        #  * the upper_id is store in the count attribute
        axis = self._nodes[id].type
        split = self._nodes[id].split
        lower_id = id + 1
        upper_id = self._nodes[id].count

        origin = ray.origin.get_index(axis)
        direction = ray.direction.get_index(axis)

        # is the ray propagating parallel to the split plane?
        if direction == 0:

            # a ray propagating parallel to the split plane
            if origin < split:
                return self._trace_node(lower_id, ray, min_range, max_range)
            else:
                return self._trace_node(upper_id, ray, min_range, max_range)

        else:

            # ray propagation is not parallel to split plane
            plane_distance = (split - origin) / direction

            # does the ray origin sit below the split
            below_split = origin < split or (origin == split and direction < 0)

            # identify the order in which the ray will interact with the nodes
            if below_split:
                near_id = lower_id
                far_id = upper_id
            else:
                near_id = upper_id
                far_id = lower_id

            # does ray only intersect with the near node?
            if plane_distance > max_range or plane_distance <= 0:
                return self._trace_node(near_id, ray, min_range, max_range)

            # does ray only intersect with the far node?
            if plane_distance < min_range:
                return self._trace_node(far_id, ray, min_range, max_range)

            # ray must intersect both nodes, try nearest node first
            # note: this could theoretically be an OR operation, but we don't
            # want to risk an optimiser inverting the logic (paranoia!)
            hit = self._trace_node(near_id, ray, min_range, plane_distance)
            if hit:
                return True
            else:
                return self._trace_node(far_id, ray, plane_distance, max_range)

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
        :return: True is a hit occurs, false otherwise.
        """

        # virtual function that must be implemented by derived classes
        raise NotImplementedError("KDTree3DCore _trace_leaf() method not implemented.")

    cpdef list items_containing(self, Point3D point):
        """
        Starts contains traversal of the kd-Tree.
        Traverses the kd-Tree to find the items that contain the specified point.

        :param point: A Point3D object.
        :return: A list of ids (indices) of the items containing the point
        """

        return self._items_containing(point)

    cdef list _items_containing(self, Point3D point):
        """
        Starts contains traversal of the kd-Tree.

        :param point: A Point3D object.
        :return: A list of ids (indices) of the items containing the point
        """

        # exit early if point is not inside bounds of the kd-Tree
        if not self.bounds.contains(point):
            return []

        # start search
        return self._items_containing_node(ROOT_NODE, point)

    cdef list _items_containing_node(self, int32_t id, Point3D point):
        """
        Dispatches contains point look-ups to the relevant node handler.

        :param id: Index of node in node array.
        :param point: Point3D to evaluate.
        :return: List of items containing the point.
        """

        if self._nodes[id].type == LEAF:
            return self._items_containing_leaf(id, point)
        else:
            return self._items_containing_branch(id, point)

    cdef list _items_containing_branch(self, int32_t id, Point3D point):
        """
        Locates the kd-Tree node containing the point.

        :param id: Index of node in node array.
        :param point: Point3D to evaluate.
        :return: List of items containing the point.
        """

        cdef:
            int32_t axis
            double split
            int32_t lower_id, upper_id

        # unpack branch kdnode
        # notes:
        #  * the branch type enumeration is the same as axis index
        #  * the lower_id is always the next node in the array
        #  * the upper_id is stored in the count attribute
        axis = self._nodes[id].type
        split = self._nodes[id].split
        lower_id = id + 1
        upper_id = self._nodes[id].count

        if point.get_index(axis) < split:
            return self._items_containing_node(lower_id, point)
        else:
            return self._items_containing_node(upper_id, point)

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

        # virtual function that must be implemented by derived classes
        raise NotImplementedError("KDTree3DCore _items_containing_leaf() method not implemented.")

    cdef void _reset(self):
        """
        Resets the kd-tree state, de-allocating all memory.
        """

        cdef:
            int32_t index
            kdnode *node

        # free all leaf node item arrays
        for index in range(self._next_node):
            if self._nodes[index].type == LEAF and self._nodes[index].count > 0:
                PyMem_Free(self._nodes[index].items)

        # free the nodes
        PyMem_Free(self._nodes)

        # reset
        self._nodes = NULL
        self._allocated_nodes = 0
        self._next_node = 0

    def __dealloc__(self):
        """
        Frees the memory allocated to store the kd-Tree.
        """

        self._reset()

    # def debug_print_all(self):
    #
    #     for i in range(self._next_node):
    #         self.debug_print_node(i)
    #
    # def debug_print_node(self, id):
    #
    #     if 0 <= id < self._next_node:
    #         if self._nodes[id].type == LEAF:
    #             print("id={} LEAF: count {}, contents: [".format(id, self._nodes[id].count), end="")
    #             for i in range(self._nodes[id].count):
    #                 print("{}".format(self._nodes[id].items[i]), end="")
    #                 if i < self._nodes[id].count - 1:
    #                     print(", ", end="")
    #             print("]")
    #         else:
    #             print("id={} BRANCH: axis {}, split {}, lower_id {}, upper_id {}".format(id, self._nodes[id].type, self._nodes[id].split, id+1, self._nodes[id].count))

    def save(self, file):

        cdef:
            int32_t id, item

        close = False

        # treat as a filename if a stream is not supplied
        if not isinstance(file, io.IOBase):
            file = open(file, mode="wb")
            close = True

        # write header
        file.write(struct.pack("<i", self._max_depth))
        file.write(struct.pack("<i", self._min_items))
        file.write(struct.pack("<d", self._hit_cost))
        file.write(struct.pack("<d", self._empty_bonus))

        # write bounds
        file.write(struct.pack("<d", self.bounds.lower.x))
        file.write(struct.pack("<d", self.bounds.lower.y))
        file.write(struct.pack("<d", self.bounds.lower.z))

        file.write(struct.pack("<d", self.bounds.upper.x))
        file.write(struct.pack("<d", self.bounds.upper.y))
        file.write(struct.pack("<d", self.bounds.upper.z))

        # write nodes
        file.write(struct.pack("<i", self._next_node))  # number of nodes
        for id in range(self._next_node):

            if self._nodes[id].type == LEAF:

                # leaf node
                file.write(struct.pack("<i", self._nodes[id].type))
                file.write(struct.pack("<i", self._nodes[id].count))
                for item in range(self._nodes[id].count):
                    file.write(struct.pack("<i", self._nodes[id].items[item]))

            else:

                # branch node
                file.write(struct.pack("<i", self._nodes[id].type))
                file.write(struct.pack("<d", self._nodes[id].split))
                file.write(struct.pack("<i", self._nodes[id].count))

        # if we opened a file, we should close it
        if close:
            file.close()

    def load(self, file):

        cdef:
            int32_t id, item

        # free existing nodes
        self._reset()

        # treat as a filename if a stream is not supplied
        close = False
        if not isinstance(file, io.IOBase):
            file = open(file, mode="rb")
            close = True

        # read header
        self._max_depth = self._read_int32(file)
        self._min_items = self._read_int32(file)
        self._hit_cost = self._read_double(file)
        self._empty_bonus = self._read_double(file)

        # read bounds
        self.bounds = BoundingBox3D(
            Point3D(
                self._read_double(file),
                self._read_double(file),
                self._read_double(file)
            ),
            Point3D(
                self._read_double(file),
                self._read_double(file),
                self._read_double(file)
            )
        )

        # read nodes
        self._next_node = self._read_int32(file)
        self._allocated_nodes = self._next_node

        # allocate nodes
        self._nodes = <kdnode *> PyMem_Malloc(sizeof(kdnode) * self._allocated_nodes)
        if not self._nodes:
            raise MemoryError()

        # load nodes
        for id in range(self._next_node):

            self._nodes[id].type = self._read_int32(file)
            if self._nodes[id].type == LEAF:

                # leaf node
                self._nodes[id].count = self._read_int32(file)
                if self._nodes[id].count > 0:

                    # allocate items
                    self._nodes[id].items = <int32_t *> PyMem_Malloc(sizeof(int32_t) * self._nodes[id].count)
                    if not self._nodes[id].items:
                        raise MemoryError()

                    # read items
                    for item in range(self._nodes[id].count):
                        self._nodes[id].items[item] = self._read_int32(file)

            else:

                # branch node
                self._nodes[id].split = self._read_double(file)
                self._nodes[id].count = self._read_int32(file)

        # if we opened a file, we should close it
        if close:
            file.close()

    cdef int32_t _read_int32(self, object file):
        return (<int32_t *> PyBytes_AsString(file.read(sizeof(int32_t))))[0]

    cdef double _read_double(self, object file):
        return (<double *> PyBytes_AsString(file.read(sizeof(double))))[0]


cdef class KDTree3D(KDTree3DCore):
    """
    Implements a 3D kd-tree for items with finite extents.

    This class cannot be used directly, it must be sub-classed. One or both of
    _trace_item() and _items_containing_item() must be implemented.

    :param items: A list of Items.
    :param max_depth: The maximum tree depth (automatic if set to 0, default is 0).
    :param min_items: The item count threshold for forcing creation of a new leaf node (default 1).
    :param hit_cost: The relative computational cost of item hit evaluations vs kd-tree traversal (default 20.0).
    :param empty_bonus: The bonus applied to node splits that generate empty leaves (default 0.2).
    """

    cdef bint _trace_leaf(self, int32_t id, Ray ray, double max_range):
        """
        Wraps the C-level API so users can derive a class from KDTree3D using Python.

        Converts the arguments to types accessible from Python and re-exposes
        _trace_leaf() as the Python accessible method _trace_items().

        :param id: Index of node in node array.
        :param ray: Ray object.
        :param max_range: The maximum intersection search range.
        :return: True is a hit occurs, false otherwise.
        """

        cdef:
            int32_t index
            list items

        # convert list of items in C-array into a list
        items = []
        for index in range(self._nodes[id].count):
            items.append(self._nodes[id].items[index])

        return self._trace_items(items, ray, max_range)

    cpdef bint _trace_items(self, list item_ids, Ray ray, double max_range):
        """
        Tests each item to identify if an intersection occurs.

        This is a virtual method and must be implemented in a derived class if
        ray intersections are to be identified. This method must return True
        if an intersection is found and False otherwise.

        Derived classes may need to return information about the intersection.
        This can be done by setting object attributes prior to returning True.
        The kd-Tree search algorithm stops as soon as the first leaf is
        identified that contains an intersection. Any attributes set when
        _trace_items() returns True are guaranteed not to be further modified.

        :param item_ids: List of item ids.
        :param ray: Ray object.
        :param max_range: The maximum intersection search range.
        :return: True is a hit occurs, false otherwise.
        """

        raise NotImplementedError("KDTree3D Virtual function _trace_items() has not been implemented.")


    cdef list _items_containing_leaf(self, int32_t id, Point3D point):
        """
        Wraps the C-level API so users can derive a class from KDTree3D using Python.

        Converts the arguments to types accessible from Python and re-exposes
        _items_containing_leaf() as the Python accessible method
        _items_containing_items().

        :param id: Index of node in node array.
        :param point: Point3D to evaluate.
        :return: List of nodes containing the point.
        """

        cdef:
            int32_t index
            list items

        # convert list of items in C-array into a list
        items = []
        for index in range(self._nodes[id].count):
            items.append(self._nodes[id].items[index])

        return self._items_containing_items(items, point)

    cpdef list _items_containing_items(self, list item_ids, Point3D point):
        """
        Tests each item in the list to identify if they enclose the point.

        This is a virtual method and must be implemented in a derived class if
        the identification of items enclosing a point is required. This method
        must return a list of ids for the items that enclose the point. If no
        items enclose the point, an empty list must be returned.

        Derived classes may need to wish to return additional information about
        the enclosing items. This can be done by setting object attributes
        prior to returning the list. Any attributes set when
        _items_containing_items() returns are guaranteed not to be further
        modified.

        :param item_ids: List of item ids.
        :param point: Point3D to evaluate.
        :return: List of ids of the items containing the point.
        """


        raise NotImplementedError("KDTree3D Virtual function _items_containing_items() has not been implemented.")