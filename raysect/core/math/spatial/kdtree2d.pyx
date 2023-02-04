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

from raysect.core.boundingbox cimport new_boundingbox2d
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


cdef class Item2D:
    """
    Item2D class. Represents an item to place into the kd-tree.

    The id should be a unique integer value identifying an external object.
    For example the id could be the index into an array of polygon objects.
    The id values are stored in the kd-tree and returned by the hit() or
    contains() methods.

    A bounding box associated with the item defines the spatial extent of the
    item along each axis. This data is used to place the items in the tree.

    :param id: An integer item id.
    :param box: A BoundingBox2D object defining the item's spatial extent.
    """

    def __init__(self, int32_t id, BoundingBox2D box):

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


cdef class KDTree2DCore:
    """
    Implements a 2D kd-tree for items with finite extents.

    This is a Cython abstract base class. It cannot be directly extended in
    Python due to the need to implement cdef methods _contains_leaf() and
     _hit_leaf(). Use the KDTree2D wrapper class if extending from Python.

    :param list items: A list of Items.
    :param int max_depth: The maximum tree depth (automatic if set to 0, default is 0).
    :param int min_items: The item count threshold for forcing creation of a new leaf node (default 1).
    :param double hit_cost: The relative computational cost of item hit evaluations vs kd-tree traversal (default 20.0).
    :param double empty_bonus: The bonus applied to node splits that generate empty leaves (default 0.2).
    """

    def __cinit__(self):

        self._nodes = NULL
        self._allocated_nodes = 0
        self._next_node = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, list items, int32_t max_depth=0, int32_t min_items=1, double hit_cost=20.0, double empty_bonus=0.2):

        cdef:
            Item2D item

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
        self.bounds = BoundingBox2D()
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

    cdef int32_t _build(self, list items, BoundingBox2D bounds, int32_t depth=0):
        """
        Extends the kd-Tree by creating a new node.

        Attempts to partition space for efficient traversal.

        :param items: A list of items.
        :param bounds: A BoundingBox2D defining the node bounds.
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
    cdef tuple _split(self, list items, BoundingBox2D bounds):
        """
        Attempts to locate a split solution that minimises the cost of traversing the node.

        The cost of the node traversal is evaluated using the Surface Area Heuristic (SAH) method.

        :param items: A list of items.
        :param bounds: A BoundingBox2D defining the node bounds.
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
            Item2D item

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
        for axis in [longest_axis, (longest_axis + 1) % 2]:

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

                    # calculate surface area of split surfaces
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
            Item2D item
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

    cdef BoundingBox2D _get_lower_bounds(self, BoundingBox2D bounds, double split, int32_t axis):
        """
        Returns the lower box generated when the node bounding box is split.

        :param bounds: A BoundingBox2D defining the node bounds.
        :param split: The value along the axis at which to split.
        :param axis: The axis to split along.
        :return: A bounding box defining the lower bounds.
        """

        cdef Point2D upper
        upper = bounds.upper.copy()
        upper.set_index(axis, split)
        return new_boundingbox2d(bounds.lower.copy(), upper)

    cdef BoundingBox2D _get_upper_bounds(self, BoundingBox2D bounds, double split, int32_t axis):
        """
        Returns the upper box generated when the node bounding box is split.

        :param bounds: A BoundingBox2D defining the node bounds.
        :param split: The value along the axis at which to split.
        :param axis: The axis to split along.
        :return: A bounding box defining the upper bounds.
        """

        cdef Point2D lower
        lower = bounds.lower.copy()
        lower.set_index(axis, split)
        return new_boundingbox2d(lower, bounds.upper.copy())

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
                self._nodes[id].items[index] = (<Item2D> items[index]).id

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
            BoundingBox2D lower_bounds,  upper_bounds

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

    cpdef bint is_contained(self, Point2D point):
        """
        Traverses the kd-Tree to identify if the point is contained by an any item.

        :param point: A Point2D object.
        :return: True if the point lies inside an item, false otherwise.
        """

        return self._is_contained(point)

    cdef bint _is_contained(self, Point2D point):
        """
        Starts contains traversal of the kd-Tree.

        :param point: A Point2D object.
        :return: True if the point lies inside an item, false otherwise.
        """

        # exit early if point is not inside bounds of the kd-Tree
        if not self.bounds.contains(point):
            return False

        # start search
        return self._is_contained_node(ROOT_NODE, point)

    cdef bint _is_contained_node(self, int32_t id, Point2D point):
        """
        Dispatches contains point look-ups to the relevant node handler.

        :param id: Index of node in node array.
        :param point: Point2D to evaluate.
        :return: True if the point lies inside an item, false otherwise.
        """

        if self._nodes[id].type == LEAF:
            return self._is_contained_leaf(id, point)
        else:
            return self._is_contained_branch(id, point)

    cdef bint _is_contained_branch(self, int32_t id, Point2D point):
        """
        Locates the kd-Tree node containing the point.

        :param id: Index of node in node array.
        :param point: Point2D to evaluate.
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

    cdef bint _is_contained_leaf(self, int32_t id, Point2D point):
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
        :param point: Point2D to evaluate.
        :return: True if the point lies inside an item, false otherwise.
        """

        # virtual function that must be implemented by derived classes
        raise NotImplementedError("KDTree2DCore _is_contained_leaf() method not implemented.")

    cpdef list items_containing(self, Point2D point):
        """
        Traverses the kd-Tree to find the items that contain the specified point.

        :param point: A Point2D object.
        :return: A list of ids (indices) of the items containing the point
        """

        return self._items_containing(point)

    cdef list _items_containing(self, Point2D point):
        """
        Starts contains traversal of the kd-Tree.

        :param point: A Point2D object.
        :return: A list of ids (indices) of the items containing the point
        """

        # exit early if point is not inside bounds of the kd-Tree
        if not self.bounds.contains(point):
            return []

        # start search
        return self._items_containing_node(ROOT_NODE, point)

    cdef list _items_containing_node(self, int32_t id, Point2D point):
        """
        Dispatches contains point look-ups to the relevant node handler.

        :param id: Index of node in node array.
        :param point: Point2D to evaluate.
        :return: List of items containing the point.
        """

        if self._nodes[id].type == LEAF:
            return self._items_containing_leaf(id, point)
        else:
            return self._items_containing_branch(id, point)

    cdef list _items_containing_branch(self, int32_t id, Point2D point):
        """
        Locates the kd-Tree node containing the point.

        :param id: Index of node in node array.
        :param point: Point2D to evaluate.
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

    cdef list _items_containing_leaf(self, int32_t id, Point2D point):
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
        :param point: Point2D to evaluate.
        :return: List of items containing the point.
        """

        # virtual function that must be implemented by derived classes
        raise NotImplementedError("KDTree2DCore _items_containing_leaf() method not implemented.")

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

        file.write(struct.pack("<d", self.bounds.upper.x))
        file.write(struct.pack("<d", self.bounds.upper.y))

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
        self.bounds = BoundingBox2D(
            Point2D(
                self._read_double(file),
                self._read_double(file),
            ),
            Point2D(
                self._read_double(file),
                self._read_double(file),
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


cdef class KDTree2D(KDTree2DCore):
    """
    Implements a 2D kd-tree for items with finite extents.

    This class cannot be used directly, it must be sub-classed. One or both of
    _hit_item() and _contains_item() must be implemented.

    :param list items: A list of Items.
    :param int max_depth: The maximum tree depth (automatic if set to 0, default is 0).
    :param int min_items: The item count threshold for forcing creation of a new leaf node (default 1).
    :param double hit_cost: The relative computational cost of item hit evaluations vs kd-tree traversal (default 20.0).
    :param double empty_bonus: The bonus applied to node splits that generate empty leaves (default 0.2).
    """

    cdef bint _is_contained_leaf(self, int32_t id, Point2D point):
        """
        Wraps the C-level API so users can derive a class from KDTree2D using Python.

        Converts the arguments to types accessible from Python and re-exposes
        _is_contained_leaf() as the Python accessible method _is_contained_items().

        :param id: Index of node in node array.
        :param point: Point2D to evaluate.
        :return: True if the point lies inside an item, false otherwise.
        """

        cdef:
            int32_t index
            list items

        # convert list of items in C-array into a list
        items = []
        for index in range(self._nodes[id].count):
            items.append(self._nodes[id].items[index])

        return self._hit_items(items, point)

    cpdef bint _is_contained_items(self, list item_ids, Point2D point):
        """
        Tests each item in the list to identify if any enclose the point.

        This is a virtual method and must be implemented in a derived class if
        the identification of an item enclosing a point is required. This method
        must return True is the point lies inside an item or False otherwise.

        Derived classes may need to wish to return additional information about
        the enclosing item(s). This can be done by setting object attributes
        prior to returning. Any attributes set when _hit_items() returns are
        guaranteed not to be further modified.

        :param item_ids: List of item ids.
        :param point: Point2D to evaluate.
        :return: True if the point lies inside an item, false otherwise.
        """

        raise NotImplementedError("KDTree2D Virtual function _is_contained_items() has not been implemented.")

    cdef list _items_containing_leaf(self, int32_t id, Point2D point):
        """
        Wraps the C-level API so users can derive a class from KDTree2D using Python.

        Converts the arguments to types accessible from Python and re-exposes
        _items_containing_leaf() as the Python accessible method _items_containing_items().

        :param id: Index of node in node array.
        :param point: Point2D to evaluate.
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

    cpdef list _items_containing_items(self, list item_ids, Point2D point):
        """
        Tests each item in the list to identify if they enclose the point.

        This is a virtual method and must be implemented in a derived class if
        the identification of items enclosing a point is required. This method
        must return a list of ids for the items that enclose the point. If no
        items enclose the point, an empty list must be returned.

        Derived classes may need to wish to return additional information about
        the enclosing items. This can be done by setting object attributes
        prior to returning the list. Any attributes set when _contains_items()
        returns are guaranteed not to be further modified.

        :param item_ids: List of item ids.
        :param point: Point2D to evaluate.
        :return: List of ids of the items containing the point.
        """

        raise NotImplementedError("KDTree2D Virtual function _items_containing_items() has not been implemented.")