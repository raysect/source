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
from libc.string cimport memcpy
cimport cython

# this number of nodes will be pre-allocated when the kd-tree is initially created
DEF INITIAL_NODE_COUNT = 128

# node types
DEF LEAF = -1
DEF X_AXIS = 0  # branch, x-axis split
DEF Y_AXIS = 1  # branch, y-axis split
DEF Z_AXIS = 2  # branch, z-axis split

# axis

cdef struct kdnode:
    int type            # LEAF, BRANCH_X, BRANCH_Y, BRANCH_Z
    double split
    int count           # upper index, item_count
    int *items


cdef class KDTree:

    cdef:
        kdnode *_nodes
        int _allocated_nodes
        int _next_node

    def __cinit__(self):

        self._nodes = NULL
        self._allocated_nodes = 0
        self._next_node = 0

    def __init__(self, items):

        self.build(X_AXIS, items, None, 0)
        # self.dump_info()

    def dump_info(self):

        for i in range(self._next_node):
            self.node_info(i)

    def node_info(self, id):

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

    cdef int build(self, int axis, list items, BoundingBox bounds, int depth):

        # come in blind
        # this function does a split test then creates appropriates node

        # it returns the id of it's node

        # DUMMY CODE
        is_leaf = len(items) <= 4
        lower_items = items[:len(items)/2]
        lower_bounds = None
        upper_items = items[len(items)/2:]
        upper_bounds = None
        split = <double> len(items) / 2

        if is_leaf:
            return self._new_leaf(items)
        else:
            return self._new_branch(axis, split, lower_items, lower_bounds, upper_items, upper_bounds, depth)

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

    cdef int _new_branch(self, int axis, double split, list lower_items, BoundingBox lower_bounds, list upper_items, BoundingBox upper_bounds, int depth):

        cdef:
            int id, upper_id

        id = self._new_node()

        # recursively build lower and upper nodes
        # the lower node is always the next node in the list
        # the upper node may be an arbitrary distance along the list
        # we store the upper node id in count for future evaluation
        self.build(axis, lower_items, lower_bounds, depth + 1)
        upper_id = self.build(axis, upper_items, upper_bounds, depth + 1)

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


