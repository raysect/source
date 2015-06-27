# cython: language_level=3
# cython: profile=True

# Copyright (c) 2015, Dr Alex Meakins, Raysect Project
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

from raysect.core.math.affinematrix cimport AffineMatrix
from raysect.core.math.normal cimport Normal, new_normal
from raysect.core.math.point cimport Point, new_point
from raysect.core.classes cimport Material, Intersection, new_intersection
from raysect.core.acceleration.boundingbox cimport BoundingBox
from libc.math cimport fabs, log, ceil
cimport cython

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-9

# extent padding for the triangles in the kd-Tree
DEF EDGE_LOWER_PADDING = 0.99999
DEF EDGE_UPPER_PADDING = 1.00001

# kd-Tree axis definitions
DEF X_AXIS = 0
DEF Y_AXIS = 1
DEF Z_AXIS = 2
DEF NO_AXIS = 3

# additional ray distance to avoid re-hitting the same surface point
DEF EPSILON = 1e-9

"""
Requirements:
* tri-poly mesh support
* option to set mesh closed or open (code will assume user is not an idiot), open mesh means contains() always reports False
* normal interpolation option (smoothing)

Development plan for mesh:

1) initial prototype
* implement watertight triangle intersection, ignoring normal interpolation for now - just use polygon normal
* implement a brute force (list based) search for closest poly and next poly -
* meshes are always open (i.e skip implementation of contains())

2) 2nd pass
* add kdtree to optimise hit and contains

3) release
* add smoothing parameter and add normal interpolation
* add open/closed mesh support (implement contains using a surface intersection count)


Notes:
The ray-triangle intersection is a partial implementation of the algorithm described in:
    "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald, Journal of Computer Graphics Techniques (2013), Vol.2, No. 1

As implemented, the algorithm is not fully watertight due to the use of double precision throughout. At present, there is no appeal to
higher precision to resolve cases when the edge tests result in a degenerate solution. This should only occur when a mesh contains
extremely small triangles that are being tested against a ray with an origin far from the mesh.
"""

cdef class Triangle:

    def __init__(self, Point v1 not None, Point v2 not None, Point v3 not None,
                 Normal n1=None, Normal n2=None, Normal n3=None):

        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self._calc_face_normal()

        # if any of the vertex normals is missing, disable interpolation
        if n1 is None or n2 is None or n3 is None:

            self._smoothing_enabled = False
            self.n1 = None
            self.n2 = None
            self.n3 = None

        else:

            self._smoothing_enabled = True
            self.n1 = n1.normalise()
            self.n2 = n2.normalise()
            self.n3 = n3.normalise()

    cdef Normal _calc_face_normal(self):
        """
        Calculate the triangles face normal from the vertices.

        The triangle face normal direction is defined by the right hand screw
        rule. When looking at the triangle from the back face, the vertices
        will be ordered in a clockwise fashion and the normal will be pointing
        away from the observer.
        """

        cdef:
            Vector a, b, c

        a = self.v1.vector_to(self.v2)
        b = self.v1.vector_to(self.v3)
        c = a.cross(b).normalise()
        self.face_normal = new_normal(c.x, c.y, c.z)

    cpdef Normal interpolate_normal(self, double u, double v, double w, bint smoothing=True):
        """
        Returns the surface normal for the specified barycentric coordinate.

        The result is undefined if u, v or w are outside the range [0, 1].
        If smoothing is disabled the result will be the face normal.

        :param u: Barycentric U coordinate.
        :param v: Barycentric V coordinate.
        :param w: Barycentric W coordinate.
        :return The surface normal at the specified coordinate.
        """

        if smoothing and self._smoothing_enabled:
            return new_normal(
                u * self.n1.x + v * self.n2.x + w * self.n3.x,
                u * self.n1.y + v * self.n2.y + w * self.n3.y,
                u * self.n1.z + v * self.n2.z + w * self.n3.z,
            )
        else:
            return self.face_normal

    def lower_extent(self, axis):
        """
        Returns the lowest extent of the triangle along the specified axis.
        """

        return min(self.v1[axis], self.v2[axis], self.v3[axis])

    def upper_extent(self, axis):
        """
        Returns the upper extent of the triangle along the specified axis.
        """

        return max(self.v1[axis], self.v2[axis], self.v3[axis])

    # def side(self, p):
    #     """
    #     Returns which side of the face the point lies on.
    #
    #     The front of the triangle is defined as the side towards which the face normal is pointing.
    #     Everywhere else is considered to liw behind the triangle. A point lying on the plane in
    #     which the triangle lies is considered to be behind the triangle.
    #
    #     :return: Returns True if the point lies in front of the triangle, False otherwise
    #     """
    #     pass


cdef class Mesh(Primitive):

    def __init__(self, list triangles, bint smoothing=True, object parent=None, AffineMatrix transform not None=AffineMatrix(), Material material not None=Material(), unicode name not None=""):

        super().__init__(parent, transform, material, name)

        self.triangles = triangles
        self.smoothing = smoothing

        self.kdtree_max_depth = -1
        self.kdtree_min_triangles = 1

        # TODO: calculate or measure this, relative cost of a triangle hit
        # calculation compared to a kdtree split traversal
        self.kdtree_hit_cost = 20.0

        # construct a bounding box that contains all the triangles in the mesh
        self._build_local_bbox()

        # build the kd-Tree
        self._build_kdtree()

    cdef object _build_kdtree(self):
        """
        Rebuilds the kd-Tree acceleration structure with the list of triangles.
        """

        cdef int max_depth

        self._kdtree = _Node()

        # default max tree depth is set to the value suggested in "Physically Based Rendering From Theory to
        # Implementation 2nd Edition", Matt Phar and Greg Humphreys, Morgan Kaufmann 2010, p232
        if self.kdtree_max_depth <= 0:
            max_depth = <int> ceil(8 + 1.3 * log(len(self.triangles)))
            print(max_depth, "max_depth")
        else:
            max_depth = self.kdtree_max_depth

        # calling build on the root node triggers a recursive rebuild of the tree
        self._kdtree.build(self._local_bbox, self.triangles, max_depth, self.kdtree_min_triangles, self.kdtree_hit_cost)

    cdef object _build_local_bbox(self):
        """
        Builds a local space bounding box that encloses all the supplied triangles.
        """

        cdef:
            Triangle triangle
            BoundingBox box

        self._local_bbox = BoundingBox()
        for triangle in self.triangles:
            self._local_bbox.extend(triangle.v1, BOX_PADDING)
            self._local_bbox.extend(triangle.v2, BOX_PADDING)
            self._local_bbox.extend(triangle.v3, BOX_PADDING)


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
            double t, u, v, w

        local_ray = Ray(
            ray.origin.transform(self.to_local()),
            ray.direction.transform(self.to_local()),
            ray.max_distance
        )

        # unpacking manually is marginally faster...
        intersection = self._local_bbox.full_intersection(local_ray)
        hit = intersection[0]
        min_range = intersection[1]
        max_range = intersection[2]
        if not hit:
            return None

        intersection = self._kdtree.hit(local_ray, min_range, max_range)
        if intersection is None:
            return None

        triangle, t, u, v, w = intersection

        hit_point = local_ray.origin + local_ray.direction * t
        inside_point = hit_point - triangle.face_normal * EPSILON
        outside_point = hit_point + triangle.face_normal * EPSILON
        normal = triangle.interpolate_normal(u, v, w, self.smoothing)
        exiting = local_ray.direction.dot(triangle.face_normal) > 0.0
        return Intersection(ray, t, self,
                            hit_point, inside_point, outside_point,
                            normal, exiting, self.to_local(), self.to_root())

    cpdef Intersection next_intersection(self):
        """
        Virtual method - to be implemented by derived classes.

        Returns the next intersection of the ray with the primitive along the
        ray path.

        This method may only be called following a call to hit(). If the ray
        has further intersections with the primitive, these may be obtained by
        repeatedly calling the next_intersection() method. Each call to
        next_intersection() will return the next ray-primitive intersection
        along the ray's path. If no further intersections are found or
        intersections lie outside the ray parameters then next_intersection()
        will return None.

        If any geometric elements of the primitive, ray and/or scenegraph are
        altered between a call to hit() and calls to next_intersection() the
        data returned by next_intersection() may be invalid. Primitives may
        cache data to accelerate next_intersection() calls which will be
        invalidated by geometric alterations to the scene. If the scene is
        altered the data returned by next_intersection() is undefined.
        """

        raise NotImplementedError("Primitive surface has not been defined. Virtual method next_intersection() has not been implemented.")

    cpdef bint contains(self, Point p) except -1:
        """
        Virtual method - to be implemented by derived classes.

        Must returns True if the Point lies within the boundary of the surface
        defined by the Primitive. False is returned otherwise.
        """

        return False

    cpdef BoundingBox bounding_box(self):
        """
        Virtual method - to be implemented by derived classes.

        When the primitive is connected to a scenegraph containing a World
        object at its root, this method should return a bounding box that
        fully encloses the primitive's surface (plus a small margin to
        avoid numerical accuracy problems). The bounding box must be defined in
        the world's coordinate space.

        If this method is called when the primitive is not connected to a
        scenegraph with a World object at its root, it must throw a TypeError
        exception.
        """

        bbox = BoundingBox()
        for triangle in self.triangles:
            bbox.extend(triangle.v1.transform(self.to_root()), BOX_PADDING)
            bbox.extend(triangle.v2.transform(self.to_root()), BOX_PADDING)
            bbox.extend(triangle.v3.transform(self.to_root()), BOX_PADDING)
        return bbox


# cdef class _TriangleData






cdef class _Edge:
    """
    Represents the upper or lower edge of a triangle's bounding box on a specified axis.
    """

    def __init__(self, Triangle triangle, int axis, bint is_upper_edge):

        self.triangle = triangle
        self.is_upper_edge = is_upper_edge

        # value is padded by a small margin as the watertight hit algorithm requires conservative bounds
        if is_upper_edge:
            self.value = EDGE_UPPER_PADDING * triangle.upper_extent(axis)
        else:
            self.value = EDGE_LOWER_PADDING * triangle.lower_extent(axis)

    def __richcmp__(_Edge x, _Edge y, int operation):

        if operation == 0:  # __lt__(), less than
            # lower edge must always be encountered first
            # break tie by ensuring lower extent sorted before upper edge
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
        self.triangles = []
        self.axis = NO_AXIS
        self.split = 0
        self.is_leaf = False

    cdef object build(self, BoundingBox node_bounds, list triangles, int depth, int min_triangles, double hit_cost, int last_axis=X_AXIS):

        cdef:
            int axis
            bint is_leaf
            double split
            list edges, lower_triangles, upper_triangles
            Triangle triangle

        if depth == 0 or len(triangles) < min_triangles:
            # print("FORCED LEAF with {} triangles, depth={}".format(len(triangles), depth))
            self._become_leaf(triangles)
            return

        # attempt split with next axis
        axis = last_axis + 1
        if axis > Z_AXIS:
            axis = X_AXIS

        is_leaf, split = self._select_split(triangles, axis, node_bounds, hit_cost)

        # no split solution found?
        if is_leaf:
            # print("LEAF with {} triangles, depth={}".format(len(triangles), depth))
            self._become_leaf(triangles)
            return

        # print(depth, best_axis, "cost {}%".format(100 * best_cost/(len(triangles) * hit_cost)), (best_split - node_bounds.lower[best_axis]) / (node_bounds.upper[best_axis] - node_bounds.lower[best_axis]))

        # using cached values split triangles into two lists
        # note the split boundary is defined as lying in the upper node
        lower_triangles = []
        upper_triangles = []
        for triangle in triangles:

            # is the triangle present in the lower node?
            if triangle.lower_extent(axis) < split:
                lower_triangles.append(triangle)

            # is the triangle present in the upper node?
            if triangle.upper_extent(axis) >= split:
                upper_triangles.append(triangle)

        # become a branch node
        self.lower_branch = _Node()
        self.upper_branch = _Node()
        self.triangles = None
        self.axis = axis
        self.split = split
        self.is_leaf = False

        # continue expanding the tree inside the two volumes
        self.lower_branch.build(self._calc_lower_bounds(node_bounds, split, axis),
                                lower_triangles, depth - 1, min_triangles, hit_cost, axis)

        self.upper_branch.build(self._calc_upper_bounds(node_bounds, split, axis),
                                upper_triangles, depth - 1, min_triangles, hit_cost, axis)

    cdef tuple _select_split(self, list triangles, int axis, BoundingBox node_bounds, double hit_cost):

        cdef:
            double split, cost, best_cost, best_split
            bint is_leaf
            int lower_triangle_count, upper_triangle_count
            double recip_total_sa, lower_sa, upper_sa
            list edges
            _Edge edge

        # store cost of leaf as current best solution
        best_cost = len(triangles) * hit_cost
        best_split = 0
        is_leaf = True

        # cache reciprocal of node's surface area
        recip_total_sa = 1.0 / node_bounds.surface_area()

        # obtain sorted list of candidate edges along chosen axis
        edges = self._build_edges(triangles, axis)

        # cache triangle counts in lower and upper volumes for speed
        lower_triangle_count = 0
        upper_triangle_count = len(triangles)

        # scan through candidate edges from lowest to highest
        for edge in edges:

            # update primitive counts for upper volume
            # note: this occasionally creates invalid solutions if edges of
            # boxes are coincident however the invalid solutions cost
            # more than the valid solutions and will not be selected
            if edge.is_upper_edge:
                upper_triangle_count -= 1

            # a split on the node boundary serves no useful purpose
            # only consider edges that lie inside the node bounds
            split = edge.value
            if node_bounds.lower.get_index(axis) < split < node_bounds.upper.get_index(axis):

                # calculate surface area of split volumes
                lower_sa = self._calc_lower_bounds(node_bounds, split, axis).surface_area()
                upper_sa = self._calc_upper_bounds(node_bounds, split, axis).surface_area()

                # calculate SAH cost
                cost = 1 + (lower_sa * lower_triangle_count + upper_sa * upper_triangle_count) * recip_total_sa * hit_cost

                # has a better split been found?
                if cost < best_cost:
                    best_cost = cost
                    best_split = split
                    is_leaf = False

            # update triangle counts for lower volume
            # note: this occasionally creates invalid solutions if edges of
            # boxes are coincident however the invalid solutions cost
            # more than the valid solutions and will not be selected
            if not edge.is_upper_edge:
                lower_triangle_count += 1

        return is_leaf, best_split

    cdef void _become_leaf(self, list triangles):

        self.lower_branch = None
        self.upper_branch = None
        self.triangles = triangles
        self.axis = NO_AXIS
        self.split = 0
        self.is_leaf = True

    cdef list _build_edges(self, list triangles, int axis):

        cdef:
            list edges
            Triangle triangle

        edges = []
        for triangle in triangles:
            edges.append(_Edge(triangle, axis, is_upper_edge=False))
            edges.append(_Edge(triangle, axis, is_upper_edge=True))
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

    cdef tuple hit(self, Ray ray, double min_range, double max_range):

        if self.is_leaf:
            return self._hit_leaf(ray, max_range)
        else:
            return self._hit_branch(ray, min_range, max_range)

    @cython.cdivision(True)
    cdef inline tuple _hit_branch(self, Ray ray, double min_range, double max_range):

        cdef:
            double origin, direction
            tuple lower_intersection, upper_intersection, intersection
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
            else:
                near = self.upper_branch
                far = self.lower_branch

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
    cdef inline tuple _hit_leaf(self, Ray ray, double max_range):
        """
        Find the closest triangle-ray intersection.
        """

        cdef:
            double distance
            tuple intersection, closest_intersection, ray_transform
            Triangle triangle, closest_triangle
            double t, u, v, w

        # find the closest triangle-ray intersection with initial search distance limited by node and ray limits
        closest_intersection = None
        distance = min(ray.max_distance, max_range)
        for triangle in self.triangles:
            # TODO: move transform calc outside the tree - it is common
            ray_transform = self._calc_rayspace_transform(ray)
            intersection = self._hit_triangle(triangle, ray_transform, ray)
            if intersection is not None and intersection[0] <= distance:
                distance = intersection[0]
                closest_intersection = intersection
                closest_triangle = triangle

        if closest_intersection is None:
            return None

        t, u, v, w = closest_intersection
        return closest_triangle, t, u, v, w

    @cython.cdivision(True)
    cdef tuple _calc_rayspace_transform(self, Ray ray):

        # This code is a Python port of the code listed in appendix A of
        #  "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald,
        #  Journal of Computer Graphics Techniques (2013), Vol.2, No. 1

        cdef:
            int ix, iy, iz
            double rdz
            double sx, sy, sz

        # to minimise numerical error cycle the direction components so the largest becomes the z-component
        if fabs(ray.direction.x) > fabs(ray.direction.y) and fabs(ray.direction.x) > fabs(ray.direction.z):

            # x dimension largest
            ix, iy, iz = 1, 2, 0

        elif fabs(ray.direction.y) > fabs(ray.direction.x) and fabs(ray.direction.y) > fabs(ray.direction.z):

            # y dimension largest
            ix, iy, iz = 2, 0, 1

        else:

            # z dimension largest
            ix, iy, iz = 0, 1, 2

        # if the z component is negative, swap x and y to restore the handedness of the space
        rdz = ray.direction.get_index(iz)
        if rdz < 0.0:
            ix, iy = iy, ix

        # calculate shear transform
        sz = 1.0 / rdz
        sx = ray.direction.get_index(ix) * sz
        sy = ray.direction.get_index(iy) * sz

        return ix, iy, iz, sx, sy, sz

    @cython.cdivision(True)
    cdef tuple _hit_triangle(self, Triangle triangle, tuple ray_transform, Ray ray):

        # This code is a Python port of the code listed in appendix A of
        #  "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald,
        #  Journal of Computer Graphics Techniques (2013), Vol.2, No. 1

        cdef:
            int ix, iy, iz
            double sx, sy, sz
            Point v1, v2, v3
            double v1z, v2z, v3z
            double x1, x2, x3, y1, y2, y3
            double t, u, v, w
            double det, det_reciprocal

        # unpack ray transform
        ix, iy, iz, sx, sy, sz = ray_transform

        # center coordinate space on ray origin
        v1 = new_point(triangle.v1.x - ray.origin.x, triangle.v1.y - ray.origin.y, triangle.v1.z - ray.origin.z)
        v2 = new_point(triangle.v2.x - ray.origin.x, triangle.v2.y - ray.origin.y, triangle.v2.z - ray.origin.z)
        v3 = new_point(triangle.v3.x - ray.origin.x, triangle.v3.y - ray.origin.y, triangle.v3.z - ray.origin.z)

        # cache z components to avoid repeated lookups
        v1z = v1.get_index(iz)
        v2z = v2.get_index(iz)
        v3z = v3.get_index(iz)

        # transform vertices by shearing and scaling space so the ray points along the +ve z axis
        # we can now discard the z-axis and work with the 2D projection of the triangle in x and y
        x1 = v1.get_index(ix) - sx * v1z
        x2 = v2.get_index(ix) - sx * v2z
        x3 = v3.get_index(ix) - sx * v3z

        y1 = v1.get_index(iy) - sy * v1z
        y2 = v2.get_index(iy) - sy * v2z
        y3 = v3.get_index(iy) - sy * v3z

        # calculate scaled barycentric coordinates
        u = x3 * y2 - y3 * x2
        v = x1 * y3 - y1 * x3
        w = x2 * y1 - y2 * x1

        # # catch cases where there is insufficient numerical accuracy to resolve the subsequent edge tests
        # if u == 0.0 or v == 0.0 or w == 0.0:
        #     # TODO: add a higher precision (128bit) fallback calculation to make this watertight

        # perform edge tests
        if (u < 0.0 or v < 0.0 or w < 0.0) and (u > 0.0 or v > 0.0 or w > 0.0):
            return None

        # calculate determinant
        det = u + v + w

        # if determinant is zero the ray is parallel to the face
        if det == 0.0:
            return None

        # calculate z coordinates for the transform vertices, we need the z component to calculate the hit distance
        z1 = sz * v1z
        z2 = sz * v2z
        z3 = sz * v3z
        t = u * z1 + v * z2 + w * z3

        # is hit distance within ray limits
        if det > 0.0:
            if t < 0.0 or t > ray.max_distance * det:
                return None
        else:
            if t > 0.0 or t < ray.max_distance * det:
                return None

        # normalise barycentric coordinates and hit distance
        det_reciprocal = 1.0 / det
        u *= det_reciprocal
        v *= det_reciprocal
        w *= det_reciprocal
        t *= det_reciprocal

        return t, u, v, w

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list contains(self, Point point):

        pass

        # cdef:
        #     BoundPrimitive primitive
        #     list enclosing_primitives
        #     double location
        #
        # if self.is_leaf:
        #
        #     enclosing_primitives = []
        #     for primitive in self.primitives:
        #         if primitive.contains(point):
        #             enclosing_primitives.append(primitive.primitive)
        #     return enclosing_primitives
        #
        # else:
        #
        #     location = point.get_index(self.axis)
        #     if location < self.split:
        #         return self.lower_branch.contains(point)
        #     else:
        #         return self.upper_branch.contains(point)

