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

# void *calloc (size_t count, size_t eltsize)
#     void free (void *ptr)
#     void *malloc (size_t size)
#     void *realloc (void *ptr, size_t newsize)




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

        self._nodes = <kdnode *> PyMem_Malloc(sizeof(kdnode) * INITIAL_NODE_COUNT)
        if not self._nodes:
            raise MemoryError()

        self._allocated_nodes = INITIAL_NODE_COUNT
        self._next_node = 0

    def __init__(self, items):

        #self.build(X_AXIS, items, None, 0)
        self.testme(0, items)

    cdef void testme(self, int depth, int maxd):

        cdef:
            kdnode *b

        if depth >= maxd:
            return

        b = self._nodes
        self._new_leaf([1,2,3,4])
        self.testme(depth + 1, maxd)
        print(depth, <unsigned long> b, <unsigned long> self._nodes)

    def testme2(self, n):

        for i in range(n):
            self._new_leaf([1,2,3,4])



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
        if self._next_node >= self._allocated_nodes:

            # double allocated memory
            new_nodes = <kdnode *> PyMem_Realloc(self._nodes, sizeof(kdnode) * self._allocated_nodes * 2)
            # new_nodes = <kdnode *> PyMem_Malloc(sizeof(kdnode) * self._allocated_nodes * 2)
            if not new_nodes:
                raise MemoryError()

            # memcpy(new_nodes, self._nodes, sizeof(kdnode) * self._allocated_nodes)

            self._nodes = new_nodes
            self._allocated_nodes *= 2

        id = self._next_node
        self._next_node += 1
        return id

    cdef int _new_leaf(self, list items): # except -1:

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

    cdef int _new_branch(self, int axis, double split, list lower_items, BoundingBox lower_bounds, list upper_items, BoundingBox upper_bounds, int depth): # except -1:

        cdef:
            int id
            int upper_id

        id = self._new_node()


        # recursively build lower and upper nodes
        # the lower node is always the next node in the list
        # the upper node may be an arbitrary distance along the list
        # we store the upper node id in count for future evaluation
        self.build(axis, lower_items, lower_bounds, depth + 1)
        self._nodes[id].count = self.build(axis, upper_items, upper_bounds, depth + 1)
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





    #
    # def __init__(self):
    #     pass
    #
    # cdef _build_tree(self):
    #     self.nodes
    #
    # cdef _new_leaf(self):
    #     pass
    #
    # cdef _new_branch(self):
    #     pass







#
# cdef class _Edge:
#     """
#     Represents the upper or lower edge of an item's bounding box on a specified axis.
#     """
#
#     def __cinit__(self, _Item item, double extent, bint is_upper_edge):
#
#         self.item = item
#         self.is_upper_edge = is_upper_edge
#         self.value = extent
#
#     def __richcmp__(_Edge x, _Edge y, int operation):
#
#         if operation == 0:  # __lt__(), less than
#             # lower edge must always be encountered first
#             # break tie by ensuring lower extent sorted before upper edge
#             if x.value == y.value:
#                 if x.is_upper_edge:
#                     return False
#                 else:
#                     return True
#             return x.value < y.value
#         else:
#             return NotImplemented
#
#
# cdef class _Item:
#
#     def __cinit__(self, int index, BoundingBox bbox):
#
#         self.index = index
#         self.bbox = bbox
#
# # make a _Node wrapper around the internal KDTree array, or just pass structures?
#
#         self.lower_branch = _KDTreeNode()
#         self.upper_branch = _KDTreeNode()
#         self.triangles = None
#         self.axis = axis
#         self.split = split
#         self.is_leaf = False
#
# # lower node is always the next node, so it is implicit

# cdef struct branch_node:
#     bint type
#     int upper_node
#     int axis
#     double split
#
# cdef struct leaf_node:
#     bint type
#     int items_count
#     int *items

# cdef union node_a:
#     int flags
#     int item_count
#     int upper_index
#
# cdef union node_b:
#     int *items
#     double split

#   2<<
# AACCCCCC|CCCCCCCC|CCCCCCCC|CCCCCCCC|CCCCCCCC|CCCCCCCC|CCCCCCCC|CCCCCCCC|
# ........|........|........|........|........|........|........|........|




#
#
# cdef class KDTree:
#
#     def __init__(self):
#
#         # kdtree construction options
#         self.max_depth = -1
#         self.empty_bonus = 0.5
#         self.hit_cost = 20.0
#         self.min_items = 1
#
#         # default to an unoptimised single leaf tree
#
#         # ON BUILD
#         # if node is node null, free existing tree
#         # create space for an initial set of nodes (say 1024)
#         # obtain the first and trigg
#
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     cdef void build(self, _Node node, BoundingBox node_bounds, list items, int axis, int depth):
#
#         if depth == 0 or len(items) <= self.min_triangles:
#             self._become_leaf(node, items)
#             return
#
#         # attempt split with next axis
#         result = self._split(items, axis, node_bounds)
#
#         # split solution found?
#         if result is None:
#             self._become_leaf(node, items)
#         else:
#             split, lower_triangle_data, upper_triangle_data = result
#             self._become_branch(node, (axis + 1) % 3, split, lower_triangle_data, upper_triangle_data, node_bounds, depth)
#
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     @cython.cdivision(True)
#     cdef tuple _split(self, int axis, list triangle_data, BoundingBox node_bounds, double hit_cost):
#
#         cdef:
#             double split, cost, best_cost, best_split
#             bint is_leaf
#             int lower_triangle_count, upper_triangle_count
#             double recip_total_sa, lower_sa, upper_sa
#             list edges, lower_triangle_data, upper_triangle_data
#             _Edge edge
#             _Item data
#
#         # store cost of leaf as current best solution
#         best_cost = len(triangle_data) * hit_cost
#         best_split = 0
#         is_leaf = True
#
#         # cache reciprocal of node's surface area
#         recip_total_sa = 1.0 / node_bounds.surface_area()
#
#         # obtain sorted list of candidate edges along chosen axis
#         edges = self._build_edges(triangle_data, axis)
#
#         # cache triangle counts in lower and upper volumes for speed
#         lower_triangle_count = 0
#         upper_triangle_count = len(triangle_data)
#
#         # scan through candidate edges from lowest to highest
#         for edge in edges:
#
#             # update primitive counts for upper volume
#             # note: this occasionally creates invalid solutions if edges of
#             # boxes are coincident however the invalid solutions cost
#             # more than the valid solutions and will not be selected
#             if edge.is_upper_edge:
#                 upper_triangle_count -= 1
#
#             # a split on the node boundary serves no useful purpose
#             # only consider edges that lie inside the node bounds
#             split = edge.value
#             if node_bounds.lower.get_index(axis) < split < node_bounds.upper.get_index(axis):
#
#                 # calculate surface area of split volumes
#                 lower_sa = self._calc_lower_bounds(node_bounds, split, axis).surface_area()
#                 upper_sa = self._calc_upper_bounds(node_bounds, split, axis).surface_area()
#
#                 # calculate SAH cost
#                 cost = 1 + (lower_sa * lower_triangle_count + upper_sa * upper_triangle_count) * recip_total_sa * hit_cost
#
#                 # has a better split been found?
#                 if cost < best_cost:
#                     best_cost = cost
#                     best_split = split
#                     is_leaf = False
#
#             # update triangle counts for lower volume
#             # note: this occasionally creates invalid solutions if edges of
#             # boxes are coincident however the invalid solutions cost
#             # more than the valid solutions and will not be selected
#             if not edge.is_upper_edge:
#                 lower_triangle_count += 1
#
#         if is_leaf:
#             return None
#
#         # using cached values split triangles into two lists
#         # note the split boundary is defined as lying in the upper node
#         lower_triangle_data = []
#         upper_triangle_data = []
#         for data in triangle_data:
#
#             # is the triangle present in the lower node?
#             if data.lower_extent[axis] < best_split:
#                 lower_triangle_data.append(data)
#
#             # is the triangle present in the upper node?
#             if data.upper_extent[axis] >= best_split:
#                 upper_triangle_data.append(data)
#
#         return best_split, lower_triangle_data, upper_triangle_data
#
#     cdef void _become_leaf(self, list triangle_data):
#
#         cdef _TriangleData data
#
#         self.lower_branch = None
#         self.upper_branch = None
#         self.axis = NO_AXIS
#         self.split = 0
#         self.is_leaf = True
#
#         self.triangles = []
#         for data in triangle_data:
#             self.triangles.append(data.triangle)
#
#     cdef void _become_branch(self, int axis, double split, list lower_triangle_data, list upper_triangle_data, BoundingBox node_bounds, int depth, int min_triangles, double hit_cost):
#
#         # become a branch node
#         self.lower_branch = _KDTreeNode()
#         self.upper_branch = _KDTreeNode()
#         self.triangles = None
#         self.axis = axis
#         self.split = split
#         self.is_leaf = False
#
#         # continue expanding the tree inside the two volumes
#         self.lower_branch.build(self._calc_lower_bounds(node_bounds, split, axis),
#                                 lower_triangle_data, depth - 1, min_triangles, hit_cost, axis)
#
#         self.upper_branch.build(self._calc_upper_bounds(node_bounds, split, axis),
#                                 upper_triangle_data, depth - 1, min_triangles, hit_cost, axis)
#
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     cdef list _build_edges(self, list triangle_data, int axis):
#
#         cdef:
#             list edges
#             _TriangleData triangle
#
#         edges = []
#         for data in triangle_data:
#             edges.append(data.lower_edges[axis])
#             edges.append(data.upper_edges[axis])
#         edges.sort()
#
#         return edges
#
#     cdef BoundingBox _calc_lower_bounds(self, BoundingBox node_bounds, double split_value, int axis):
#
#         cdef Point upper
#         upper = node_bounds.upper.copy()
#         upper.set_index(axis, split_value)
#         return new_boundingbox(node_bounds.lower.copy(), upper)
#
#     cdef BoundingBox _calc_upper_bounds(self, BoundingBox node_bounds, double split_value, int axis):
#
#         cdef Point lower
#         lower = node_bounds.lower.copy()
#         lower.set_index(axis, split_value)
#         return new_boundingbox(lower, node_bounds.upper.copy())
#
#     cdef tuple hit(self, Ray ray, double min_range, double max_range):
#
#         if self.is_leaf:
#             return self._hit_leaf(ray, max_range)
#         else:
#             return self._hit_branch(ray, min_range, max_range)
#
#     @cython.cdivision(True)
#     cdef inline tuple _hit_branch(self, Ray ray, double min_range, double max_range):
#
#         cdef:
#             double origin, direction
#             tuple lower_intersection, upper_intersection, intersection
#             double plane_distance
#             _KDTreeNode near, far
#
#         origin = ray.origin.get_index(self.axis)
#         direction = ray.direction.get_index(self.axis)
#
#         # is the ray propagating parallel to the split plane?
#         if direction == 0:
#
#             # a ray propagating parallel to the split plane will only ever interact with one of the nodes
#             if origin < self.split:
#                 return self.lower_branch.hit(ray, min_range, max_range)
#             else:
#                 return self.upper_branch.hit(ray, min_range, max_range)
#
#         else:
#
#             # ray propagation is not parallel to split plane
#             plane_distance = (self.split - origin) / direction
#
#             # identify the order in which the ray will interact with the nodes
#             if origin < self.split:
#                 near = self.lower_branch
#                 far = self.upper_branch
#             elif origin > self.split:
#                 near = self.upper_branch
#                 far = self.lower_branch
#             else:
#                 # degenerate case, note split plane lives in upper branch
#                 if direction >= 0:
#                     near = self.upper_branch
#                     far = self.lower_branch
#                 else:
#                     near = self.lower_branch
#                     far = self.upper_branch
#
#             # IN LOCAL SPACE THIS SPLIT IS ON THE AXIS
#             # does ray only intersect with the near node?
#             if plane_distance > max_range or plane_distance <= 0:
#                 return near.hit(ray, min_range, max_range)
#
#             # does ray only intersect with the far node?
#             if plane_distance < min_range:
#                 return far.hit(ray, min_range, max_range)
#
#             # ray must intersect both nodes, try nearest node first
#             intersection = near.hit(ray, min_range, plane_distance)
#             if intersection is not None:
#                 return intersection
#
#             intersection = far.hit(ray, plane_distance, max_range)
#             return intersection
#
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     cdef inline tuple _hit_leaf(self, Ray ray, double max_range):
#         """
#         Find the closest triangle-ray intersection.
#         """
#
#         cdef:
#             double distance
#             tuple intersection, closest_intersection, ray_transform
#             Triangle triangle, closest_triangle
#             double t, u, v, w
#
#         #print(len(self.triangles))
#
#         # find the closest triangle-ray intersection with initial search distance limited by node and ray limits
#         closest_intersection = None
#         distance = min(ray.max_distance, max_range)
#         for triangle in self.triangles:
#             # TODO: move transform calc outside the tree - it is common
#             ray_transform = self._calc_rayspace_transform(ray)
#             intersection = self._hit_triangle(triangle, ray_transform, ray)
#             if intersection is not None and intersection[0] <= distance:
#                 distance = intersection[0]
#                 closest_intersection = intersection
#                 closest_triangle = triangle
#
#         if closest_intersection is None:
#             return None
#
#         t, u, v, w = closest_intersection
#         return closest_triangle, t, u, v, w
#
#     @cython.cdivision(True)
#     cdef tuple _calc_rayspace_transform(self, Ray ray):
#
#         # This code is a Python port of the code listed in appendix A of
#         #  "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald,
#         #  Journal of Computer Graphics Techniques (2013), Vol.2, No. 1
#
#         cdef:
#             int ix, iy, iz
#             double rdz
#             double sx, sy, sz
#
#         # to minimise numerical error cycle the direction components so the largest becomes the z-component
#         if fabs(ray.direction.x) > fabs(ray.direction.y) and fabs(ray.direction.x) > fabs(ray.direction.z):
#
#             # x dimension largest
#             ix, iy, iz = 1, 2, 0
#
#         elif fabs(ray.direction.y) > fabs(ray.direction.x) and fabs(ray.direction.y) > fabs(ray.direction.z):
#
#             # y dimension largest
#             ix, iy, iz = 2, 0, 1
#
#         else:
#
#             # z dimension largest
#             ix, iy, iz = 0, 1, 2
#
#         # if the z component is negative, swap x and y to restore the handedness of the space
#         rdz = ray.direction.get_index(iz)
#         if rdz < 0.0:
#             ix, iy = iy, ix
#
#         # calculate shear transform
#         sz = 1.0 / rdz
#         sx = ray.direction.get_index(ix) * sz
#         sy = ray.direction.get_index(iy) * sz
#
#         return ix, iy, iz, sx, sy, sz
#
#     @cython.cdivision(True)
#     cdef tuple _hit_triangle(self, Triangle triangle, tuple ray_transform, Ray ray):
#
#         # This code is a Python port of the code listed in appendix A of
#         #  "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald,
#         #  Journal of Computer Graphics Techniques (2013), Vol.2, No. 1
#
#         cdef:
#             int ix, iy, iz
#             double sx, sy, sz
#             Point v1, v2, v3
#             double v1z, v2z, v3z
#             double x1, x2, x3, y1, y2, y3
#             double t, u, v, w
#             double det, det_reciprocal
#
#         # unpack ray transform
#         ix, iy, iz, sx, sy, sz = ray_transform
#
#         # center coordinate space on ray origin
#         v1 = new_point(triangle.v1.x - ray.origin.x, triangle.v1.y - ray.origin.y, triangle.v1.z - ray.origin.z)
#         v2 = new_point(triangle.v2.x - ray.origin.x, triangle.v2.y - ray.origin.y, triangle.v2.z - ray.origin.z)
#         v3 = new_point(triangle.v3.x - ray.origin.x, triangle.v3.y - ray.origin.y, triangle.v3.z - ray.origin.z)
#
#         # cache z components to avoid repeated lookups
#         v1z = v1.get_index(iz)
#         v2z = v2.get_index(iz)
#         v3z = v3.get_index(iz)
#
#         # transform vertices by shearing and scaling space so the ray points along the +ve z axis
#         # we can now discard the z-axis and work with the 2D projection of the triangle in x and y
#         x1 = v1.get_index(ix) - sx * v1z
#         x2 = v2.get_index(ix) - sx * v2z
#         x3 = v3.get_index(ix) - sx * v3z
#
#         y1 = v1.get_index(iy) - sy * v1z
#         y2 = v2.get_index(iy) - sy * v2z
#         y3 = v3.get_index(iy) - sy * v3z
#
#         # calculate scaled barycentric coordinates
#         u = x3 * y2 - y3 * x2
#         v = x1 * y3 - y1 * x3
#         w = x2 * y1 - y2 * x1
#
#         # # catch cases where there is insufficient numerical accuracy to resolve the subsequent edge tests
#         # if u == 0.0 or v == 0.0 or w == 0.0:
#         #     # TODO: add a higher precision (128bit) fallback calculation to make this watertight
#
#         # perform edge tests
#         if (u < 0.0 or v < 0.0 or w < 0.0) and (u > 0.0 or v > 0.0 or w > 0.0):
#             return None
#
#         # calculate determinant
#         det = u + v + w
#
#         # if determinant is zero the ray is parallel to the face
#         if det == 0.0:
#             return None
#
#         # calculate z coordinates for the transform vertices, we need the z component to calculate the hit distance
#         z1 = sz * v1z
#         z2 = sz * v2z
#         z3 = sz * v3z
#         t = u * z1 + v * z2 + w * z3
#
#         # is hit distance within ray limits
#         if det > 0.0:
#             if t < 0.0 or t > ray.max_distance * det:
#                 return None
#         else:
#             if t > 0.0 or t < ray.max_distance * det:
#                 return None
#
#         # normalise barycentric coordinates and hit distance
#         det_reciprocal = 1.0 / det
#         u *= det_reciprocal
#         v *= det_reciprocal
#         w *= det_reciprocal
#         t *= det_reciprocal
#
#         return t, u, v, w
#
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     cdef list contains(self, Point point):
#
#         pass
#
#         # cdef:
#         #     BoundPrimitive primitive
#         #     list enclosing_primitives
#         #     double location
#         #
#         # if self.is_leaf:
#         #
#         #     enclosing_primitives = []
#         #     for primitive in self.primitives:
#         #         if primitive.contains(point):
#         #             enclosing_primitives.append(primitive.primitive)
#         #     return enclosing_primitives
#         #
#         # else:
#         #
#         #     location = point.get_index(self.axis)
#         #     if location < self.split:
#         #         return self.lower_branch.contains(point)
#         #     else:
#         #         return self.upper_branch.contains(point)