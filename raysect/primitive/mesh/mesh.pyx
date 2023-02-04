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

from numpy import array, float32, int32, zeros
from raysect.core cimport Primitive, AffineMatrix3D, Normal3D, new_normal3d, Point3D, new_point3d, Vector3D, new_vector3d, Material, Ray, new_ray, Intersection, new_intersection, BoundingBox3D, new_boundingbox3d
from raysect.core.math.spatial cimport KDTree3DCore, Item3D
from libc.math cimport fabs
from numpy cimport float32_t, int32_t, uint8_t
from cpython.bytes cimport PyBytes_AsString
cimport cython

"""
The ray-triangle intersection used for the Mesh primitive is an implementation of the algorithm described in:
    "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald, Journal of Computer Graphics Techniques (2013), Vol.2, No. 1
"""

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-6

# additional ray distance to avoid re-hitting the same surface point
DEF EPSILON = 1e-6

# convenience defines
DEF X = 0
DEF Y = 1
DEF Z = 2

DEF U = 0
DEF V = 1
DEF W = 2
DEF T = 3

DEF V1 = 0
DEF V2 = 1
DEF V3 = 2
DEF N1 = 3
DEF N2 = 4
DEF N3 = 5

DEF NO_INTERSECTION = -1

# raysect mesh format constants
DEF RSM_VERSION_MAJOR = 1
DEF RSM_VERSION_MINOR = 0


cdef class MeshIntersection(Intersection):
    """
    Describes the result of a ray-primitive intersection with a Mesh primitive.

    :param Ray ray: The incident ray object (world space).
    :param double ray_distance: The distance of the intersection along the ray path.
    :param Primitive primitive: The intersected primitive object.
    :param Point3D hit_point: The point of intersection between the ray and the primitive (primitive local space).
    :param Point3D inside_point: The interior ray launch point (primitive local space).
    :param Point3D outside_point: The exterior ray launch point (primitive local space).
    :param Normal3D normal: The surface normal (primitive local space)
    :param bool exiting: True if the ray is exiting the surface, False otherwise.
    :param AffineMatrix3D world_to_primitive: A world to primitive local transform matrix.
    :param AffineMatrix3D primitive_to_world: A primitive local to world transform matrix.

    :ivar bool exiting: True if the ray is exiting the surface, False otherwise.
    :ivar Point3D hit_point: The point of intersection between the ray and the primitive
      (primitive local space).
    :ivar Point3D inside_point: The interior ray launch point (primitive local space).
    :ivar Normal3D normal: The surface normal (primitive local space).
    :ivar Point3D outside_point: The exterior ray launch point (primitive local space).
    :ivar Primitive primitive: The primitive object that was intersected by the Ray.
    :ivar AffineMatrix3D primitive_to_world: The primitive's local to world transform matrix.
    :ivar Ray ray: The incident ray object (world space).
    :ivar double ray_distance: The distance of the intersection along the ray path.
    :ivar AffineMatrix3D world_to_primitive: A world to primitive local transform matrix.
    :ivar int triangle: The index of the triangle intersected.
    :ivar float u: The barycentric coordinate U of the intersection.
    :ivar float v: The barycentric coordinate V of the intersection.
    :ivar float w: The barycentric coordinate W of the intersection.
    """

    def __init__(self, Ray ray, double ray_distance, Primitive primitive,
                 Point3D hit_point, Point3D inside_point, Point3D outside_point,
                 Normal3D normal, bint exiting, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                 int32_t triangle, float u, float v, float w):

        self._construct(ray, ray_distance, primitive, hit_point, inside_point, outside_point, normal, exiting, world_to_primitive, primitive_to_world)
        self.triangle = triangle
        self.u = u
        self.v = v
        self.w = w

    def __repr__(self):

        return "MeshIntersection({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
            self.ray, self.ray_distance, self.primitive,
            self.hit_point, self.inside_point, self.outside_point,
            self.normal, self.exiting,
            self.world_to_primitive, self.primitive_to_world,
            self.triangle, self.u, self.v, self.w)


# TODO: fire exceptions if degenerate triangles are found and tolerant mode is not enabled (the face normal call will fail @ normalisation)
# TODO: tidy up the internal storage of triangles - separate the triangle reference arrays for vertices, normals etc...
# TODO: the following code really is a bit opaque, needs a general tidy up
# TODO: move load/save code to C?
cdef class MeshData(KDTree3DCore):
    """
    Holds the mesh data and acceleration structures.

    The Mesh primitive is a thin wrapper around a MeshData object. This
    arrangement simplifies mesh instancing and the load/dump methods.

    :param object vertices: A list/array or triangle vertices with shape Nx3,
      where N is the number of vertices.
    :param object triangles: A list/array of triangles with shape Nx3 or Nx6
      where N is the number of triangles in the mesh. For each triangle there
      must be three integers identifying the triangle's vertices in the vertices
      array. If vertex normals are present then three additional integers
      specify the triangle's vertex normals in the normals array.
    :param object normals: Optional array of triangle normals (default=None).
    :param bool smoothing: Turns on smoothing of triangle surface normals when
      calculating ray intersections (default=True).
    :param bool closed: Whether this mesh should be treated as a closed surface,
      i.e. no holes. (default=True)
    :param bool tolerant: Toggles filtering out of degenerate triangles
      (default=True).
    :param bool flip_normals: Inverts the direction of the surface normals (default=False).
    :param int max_depth: Maximum kd-Tree depth for this mesh (automatic if set to
      0, default=0).
    :param int min_items: The item count threshold for forcing creation of a
      new leaf node in the kdTree (default=1).
    :param double hit_cost: The relative computational cost of item hit evaluations
      vs kd-tree traversal (default=20.0).
    :param double empty_bonus: The bonus applied to node splits that generate empty
      kd-Tree leaves (default=0.2).
    """

    def __init__(self, object vertices, object triangles, object normals=None, bint smoothing=True,
                 bint closed=True, bint tolerant=True, bint flip_normals=False,
                 int max_depth=0, int min_items=1, double hit_cost=20.0, double empty_bonus=0.2):

        self.smoothing = smoothing
        self.closed = closed

        # convert to numpy arrays for internal use
        vertices = array(vertices, dtype=float32)
        triangles = array(triangles, dtype=int32)
        if normals is not None:
            vertex_normals = array(normals, dtype=float32)
        else:
            vertex_normals = None

        # check dimensions are correct
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError("The vertex array must have dimensions Nx3.")

        if vertex_normals is not None:

            if len(vertex_normals.shape) != 2 or vertex_normals.shape[1] != 3:
                raise ValueError("The normal array must have dimensions Nx3.")

            if len(triangles.shape) != 2 or triangles.shape[1] != 6:
                raise ValueError("The triangle array must have dimensions Nx6.")

        else:

            if len(triangles.shape) != 2 or triangles.shape[1] != 3:
                raise ValueError("The triangle array must have dimensions Nx3.")

        # check triangles contains only valid indices
        invalid = (triangles[:, 0:3] < 0) | (triangles[:, 0:3] >= vertices.shape[0])
        if invalid.any():
            raise ValueError("The triangle array references non-existent vertices.")

        if vertex_normals is not None:
            invalid = (triangles[:, 3:6] < 0) | (triangles[:, 3:6] >= vertex_normals.shape[0])
            if invalid.any():
                raise ValueError("The triangle array references non-existent normals.")

        # assign to internal attributes
        self._vertices = vertices
        self._vertex_normals = vertex_normals
        self._triangles = triangles

        # assign to memory views
        self.vertices_mv = vertices
        self.vertex_normals_mv = vertex_normals
        self.triangles_mv = triangles

        # initial hit data
        self._u = -1.0
        self._v = -1.0
        self._w = -1.0
        self._t = INFINITY
        self._i = NO_INTERSECTION

        # filter out degenerate triangles if we are being tolerant
        if tolerant:
            self._filter_triangles()

        # flip normals if requested
        if flip_normals:
            self._flip_normals()

        # generate face normals
        self._generate_face_normals()

        # kd-Tree init requires the triangle's id (it's index here) and bounding box
        items = []
        for i in range(self.triangles_mv.shape[0]):
            items.append(Item3D(i, self._generate_bounding_box(i)))

        super().__init__(items, max_depth, min_items, hit_cost, empty_bonus)

    def __getstate__(self):
        state = io.BytesIO()
        self.save(state)
        return state.getvalue()

    def __setstate__(self, state):
        self.load(io.BytesIO(state))

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @property
    def vertices(self):
        return self._vertices.copy()

    @property
    def triangles(self):
        return self._triangles.copy()

    @property
    def vertex_normals(self):
        if self._vertex_normals is None:
            return None
        return self._vertex_normals.copy()

    @property
    def face_normals(self):
        return self._face_normals.copy()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Point3D vertex(self, int index):
        """
        Returns the specified vertex.
        
        :param index: The vertex index.
        :return: A Point3D object. 
        """

        if index < 0 or index >= self.vertices_mv.shape[0]:
            raise ValueError('Vertex index is out of range: [0, {}].'.format(self.vertices_mv.shape[0]))

        return new_point3d(
            self.vertices_mv[index, X],
            self.vertices_mv[index, Y],
            self.vertices_mv[index, Z]
        )

    cpdef ndarray triangle(self, int index):
        """
        Returns the specified triangle.
        
        The returned data will either be a 3 or 6 element numpy array. The 
        first three element are the triangle's vertex indices. If present, the
        last three elements are the triangle's vertex normal indices.
        
        :param index: The triangle index.
        :return: A numpy array. 
        """

        if index < 0 or index >= self.vertices_mv.shape[0]:
            raise ValueError('Triangle index is out of range: [0, {}].'.format(self.triangles_mv.shape[0]))

        return self._triangles[index, :].copy()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Normal3D vertex_normal(self, int index):
        """
        Returns the specified vertex normal.
        
        :param index: The vertex normal's index.
        :return: A Normal3D object. 
        """

        if self._vertex_normals is None:
            raise ValueError('Mesh does not contain vertex normals.')

        if index < 0 or index >= self.vertex_normals_mv.shape[0]:
            raise ValueError('Vertex normal index is out of range: [0, {}].'.format(self.vertex_normals_mv.shape[0]))

        return new_normal3d(
            self.vertex_normals_mv[index, X],
            self.vertex_normals_mv[index, Y],
            self.vertex_normals_mv[index, Z]
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Normal3D face_normal(self, int index):
        """
        Returns the specified face normal.
        
        :param index: The face normal's index.
        :return: A Normal3D object. 
        """

        if index < 0 or index >= self.face_normals_mv.shape[0]:
            raise ValueError('Face normal index is out of range: [0, {}].'.format(self.face_normals_mv.shape[0]))

        return new_normal3d(
            self.face_normals_mv[index, X],
            self.face_normals_mv[index, Y],
            self.face_normals_mv[index, Z]
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef object _filter_triangles(self):

        cdef:
            int32_t i, valid
            int32_t i1, i2, i3
            Point3D p1, p2, p3
            Vector3D v1, v2, v3

        # scan triangles and make valid triangles contiguous
        valid = 0
        for i in range(self.triangles_mv.shape[0]):

            i1 = self.triangles_mv[i, V1]
            i2 = self.triangles_mv[i, V2]
            i3 = self.triangles_mv[i, V3]

            p1 = new_point3d(self.vertices_mv[i1, X], self.vertices_mv[i1, Y], self.vertices_mv[i1, Z])
            p2 = new_point3d(self.vertices_mv[i2, X], self.vertices_mv[i2, Y], self.vertices_mv[i2, Z])
            p3 = new_point3d(self.vertices_mv[i3, X], self.vertices_mv[i3, Y], self.vertices_mv[i3, Z])

            # the cross product of two edge vectors of a degenerate triangle
            # (where 2 or more vertices are coincident or lie on the same line)
            # is zero
            v1 = p1.vector_to(p2)
            v2 = p1.vector_to(p3)
            v3 = v1.cross(v2)
            if v3.get_length() == 0.0:

                # triangle is degenerate, skip
                continue

            # shift triangles
            self.triangles_mv[valid, :] = self.triangles_mv[i, :]
            valid += 1

        # reslice array to contain only valid triangles
        self.triangles_mv = self.triangles_mv[:valid, :]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef object _flip_normals(self):
        """
        Flip the face orientation of this mesh.
        """

        if self._vertex_normals is None:

            for i in range(self.triangles_mv.shape[0]):
                self.triangles_mv[i, 0], self.triangles_mv[i, 2] = self.triangles_mv[i, 2], self.triangles_mv[i, 0]

        else:

            for i in range(self.triangles_mv.shape[0]):
                self.triangles_mv[i, 0], self.triangles_mv[i, 2] = self.triangles_mv[i, 2], self.triangles_mv[i, 0]
                self.triangles_mv[i, 3], self.triangles_mv[i, 5] = self.triangles_mv[i, 5], self.triangles_mv[i, 3]

            for i in range(self.vertex_normals_mv.shape[0]):
                self.vertex_normals_mv[i, X] = -self.vertex_normals_mv[i, X]
                self.vertex_normals_mv[i, Y] = -self.vertex_normals_mv[i, Y]
                self.vertex_normals_mv[i, Z] = -self.vertex_normals_mv[i, Z]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef object _generate_face_normals(self):
        """
        Calculate the triangles face normals from the vertices.

        The triangle face normal direction is defined by the right hand screw
        rule. When looking at the triangle from the back face, the vertices
        will be ordered in a clockwise fashion and the normal will be pointing
        away from the observer.
        """

        cdef:
            int32_t i
            int32_t i1, i2, i3
            Point3D p1, p2, p3
            Vector3D v1, v2, v3

        self._face_normals = zeros((self.triangles_mv.shape[0], 3), dtype=float32)
        self.face_normals_mv = self._face_normals
        for i in range(self.face_normals_mv.shape[0]):

            i1 = self.triangles_mv[i, V1]
            i2 = self.triangles_mv[i, V2]
            i3 = self.triangles_mv[i, V3]

            p1 = new_point3d(self.vertices_mv[i1, X], self.vertices_mv[i1, Y], self.vertices_mv[i1, Z])
            p2 = new_point3d(self.vertices_mv[i2, X], self.vertices_mv[i2, Y], self.vertices_mv[i2, Z])
            p3 = new_point3d(self.vertices_mv[i3, X], self.vertices_mv[i3, Y], self.vertices_mv[i3, Z])

            v1 = p1.vector_to(p2)
            v2 = p1.vector_to(p3)
            v3 = v1.cross(v2).normalise()

            self.face_normals_mv[i, X] = v3.x
            self.face_normals_mv[i, Y] = v3.y
            self.face_normals_mv[i, Z] = v3.z

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef BoundingBox3D _generate_bounding_box(self, int32_t i):
        """
        Generates a bounding box for the specified triangle.

        A small degree of padding is added to the bounding box to provide the
        conservative bounds required by the watertight mesh algorithm.

        :param i: Triangle array index.
        :return: A BoundingBox3D object.
        """

        cdef:
            int32_t i1, i2, i3
            BoundingBox3D bbox

        i1 = self.triangles_mv[i, V1]
        i2 = self.triangles_mv[i, V2]
        i3 = self.triangles_mv[i, V3]

        bbox = new_boundingbox3d(
            new_point3d(
                min(self.vertices_mv[i1, X], self.vertices_mv[i2, X], self.vertices_mv[i3, X]),
                min(self.vertices_mv[i1, Y], self.vertices_mv[i2, Y], self.vertices_mv[i3, Y]),
                min(self.vertices_mv[i1, Z], self.vertices_mv[i2, Z], self.vertices_mv[i3, Z]),
            ),
            new_point3d(
                max(self.vertices_mv[i1, X], self.vertices_mv[i2, X], self.vertices_mv[i3, X]),
                max(self.vertices_mv[i1, Y], self.vertices_mv[i2, Y], self.vertices_mv[i3, Y]),
                max(self.vertices_mv[i1, Z], self.vertices_mv[i2, Z], self.vertices_mv[i3, Z]),
            ),
        )

        # The bounding box and triangle vertices may not align following coordinate
        # transforms in the water tight mesh algorithm, therefore a small bit of padding
        # is added to avoid numerical representation issues.
        bbox.pad(max(BOX_PADDING, bbox.largest_extent() * BOX_PADDING))

        return bbox

    cpdef bint trace(self, Ray ray):

        # reset hit data
        self._u = -1.0
        self._v = -1.0
        self._w = -1.0
        self._t = INFINITY
        self._i = NO_INTERSECTION

        self._calc_rayspace_transform(ray)
        return self._trace(ray)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _trace_leaf(self, int32_t id, Ray ray, double max_range):

        cdef:
            float hit_data[4]
            int32_t count, item, index
            double distance
            double u, v, w, t
            int32_t triangle, closest_triangle

        # unpack leaf data
        count = self._nodes[id].count

        # find the closest triangle-ray intersection with initial search distance limited by node and ray limits
        # closest_triangle is initialised with an illegal value so a non-intersection can be detected
        distance = min(ray.max_distance, max_range)
        closest_triangle = NO_INTERSECTION
        for item in range(count):

            # dereference the triangle
            triangle = self._nodes[id].items[item]

            # test for intersection
            if self._hit_triangle(triangle, ray, hit_data):

                t = hit_data[T]
                if t < distance:

                    distance = t
                    closest_triangle = triangle
                    u = hit_data[U]
                    v = hit_data[V]
                    w = hit_data[W]

        if closest_triangle == NO_INTERSECTION:
            return False

        # update intersection data
        self._u = u
        self._v = v
        self._w = w
        self._t = distance
        self._i = closest_triangle

        return True

    @cython.cdivision(True)
    cdef void _calc_rayspace_transform(self, Ray ray):

        # This code is a Python port of the code listed in appendix A of
        #  "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald,
        #  Journal of Computer Graphics Techniques (2013), Vol.2, No. 1

        cdef:
            int32_t ix, iy, iz
            float rdz
            float sx, sy, sz

        # to minimise numerical error cycle the direction components so the largest becomes the z-component
        if fabs(ray.direction.x) > fabs(ray.direction.y) and fabs(ray.direction.x) > fabs(ray.direction.z):

            # x dimension largest
            ix, iy, iz = Y, Z, X

        elif fabs(ray.direction.y) > fabs(ray.direction.x) and fabs(ray.direction.y) > fabs(ray.direction.z):

            # y dimension largest
            ix, iy, iz = Z, X, Y

        else:

            # z dimension largest
            ix, iy, iz = X, Y, Z

        # if the z component is negative, swap x and y to restore the handedness of the space
        rdz = ray.direction.get_index(iz)
        if rdz < 0.0:
            ix, iy = iy, ix

        # calculate shear transform
        sz = 1.0 / rdz
        sx = ray.direction.get_index(ix) * sz
        sy = ray.direction.get_index(iy) * sz

        # store ray transform
        self._ix = ix
        self._iy = iy
        self._iz = iz

        self._sx = sx
        self._sy = sy
        self._sz = sz

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef bint _hit_triangle(self, int32_t i, Ray ray, float[4] hit_data):

        # This code is a Python port of the code listed in appendix A of
        #  "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald,
        #  Journal of Computer Graphics Techniques (2013), Vol.2, No. 1

        cdef:
            int32_t i1, i2, i3
            int32_t ix, iy, iz
            float sx, sy, sz
            float[3] v1, v2, v3
            float x1, x2, x3
            float y1, y2, y3
            float z1, z2, z3
            float t, u, v, w
            float det, det_reciprocal

        # obtain vertex ids
        i1 = self.triangles_mv[i, V1]
        i2 = self.triangles_mv[i, V2]
        i3 = self.triangles_mv[i, V3]

        # center coordinate space on ray origin
        v1[X] = self.vertices_mv[i1, X] - ray.origin.x
        v1[Y] = self.vertices_mv[i1, Y] - ray.origin.y
        v1[Z] = self.vertices_mv[i1, Z] - ray.origin.z

        v2[X] = self.vertices_mv[i2, X] - ray.origin.x
        v2[Y] = self.vertices_mv[i2, Y] - ray.origin.y
        v2[Z] = self.vertices_mv[i2, Z] - ray.origin.z

        v3[X] = self.vertices_mv[i3, X] - ray.origin.x
        v3[Y] = self.vertices_mv[i3, Y] - ray.origin.y
        v3[Z] = self.vertices_mv[i3, Z] - ray.origin.z

        # obtain ray transform
        ix = self._ix
        iy = self._iy
        iz = self._iz

        sx = self._sx
        sy = self._sy
        sz = self._sz

        # transform vertices by shearing and scaling space so the ray points along the +ve z axis
        # we can now discard the z-axis and work with the 2D projection of the triangle in x and y
        x1 = v1[ix] - sx * v1[iz]
        x2 = v2[ix] - sx * v2[iz]
        x3 = v3[ix] - sx * v3[iz]

        y1 = v1[iy] - sy * v1[iz]
        y2 = v2[iy] - sy * v2[iz]
        y3 = v3[iy] - sy * v3[iz]

        # calculate scaled barycentric coordinates
        u = x3 * y2 - y3 * x2
        v = x1 * y3 - y1 * x3
        w = x2 * y1 - y2 * x1

        # catch cases where there is insufficient numerical accuracy to resolve the subsequent edge tests
        if u == 0.0 or v == 0.0 or w == 0.0:
            u = <float> (<double> x3 * <double> y2 - <double> y3 * <double> x2)
            v = <float> (<double> x1 * <double> y3 - <double> y1 * <double> x3)
            w = <float> (<double> x2 * <double> y1 - <double> y2 * <double> x1)

        # perform edge tests
        if (u < 0.0 or v < 0.0 or w < 0.0) and (u > 0.0 or v > 0.0 or w > 0.0):
            return False

        # calculate determinant
        det = u + v + w

        # if determinant is zero the ray is parallel to the face
        if det == 0.0:
            return False

        # calculate z coordinates for the transform vertices, we need the z component to calculate the hit distance
        z1 = sz * v1[iz]
        z2 = sz * v2[iz]
        z3 = sz * v3[iz]
        t = u * z1 + v * z2 + w * z3

        # is hit distance within ray limits
        if det > 0.0:
            if t < 0.0 or t > ray.max_distance * det:
                return False
        else:
            if t > 0.0 or t < ray.max_distance * det:
                return False

        # normalise barycentric coordinates and hit distance
        det_reciprocal = 1.0 / det
        hit_data[U] = u * det_reciprocal
        hit_data[V] = v * det_reciprocal
        hit_data[W] = w * det_reciprocal
        hit_data[T] = t * det_reciprocal

        return True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Intersection calc_intersection(self, Ray ray):

        cdef:
            double t
            int32_t triangle
            Point3D hit_point, inside_point, outside_point
            Normal3D face_normal, normal
            bint exiting

        # on a hit the kd-tree populates attributes containing the intersection data
        t = self._t
        triangle = self._i

        if triangle == NO_INTERSECTION:
            return None

        # generate intersection description
        face_normal = new_normal3d(
            self.face_normals_mv[triangle, X],
            self.face_normals_mv[triangle, Y],
            self.face_normals_mv[triangle, Z]
        )
        hit_point = new_point3d(
            ray.origin.x + ray.direction.x * t,
            ray.origin.y + ray.direction.y * t,
            ray.origin.z + ray.direction.z * t
        )
        inside_point = new_point3d(
            hit_point.x - face_normal.x * EPSILON,
            hit_point.y - face_normal.y * EPSILON,
            hit_point.z - face_normal.z * EPSILON
        )
        outside_point = new_point3d(
            hit_point.x + face_normal.x * EPSILON,
            hit_point.y + face_normal.y * EPSILON,
            hit_point.z + face_normal.z * EPSILON
        )
        normal = self._intersection_normal()
        exiting = ray.direction.dot(face_normal) > 0.0

        return new_mesh_intersection(
            ray, t, None,
            hit_point, inside_point, outside_point,
            normal, exiting, None, None,
            self._i, self._u, self._v, self._w

        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef Normal3D _intersection_normal(self):
        """
        Returns the surface normal for the last triangle hit.

        The result is undefined if this method is called when a triangle has
        not been hit (u, v or w are outside the range [0, 1]). If smoothing is
        disabled the result will be the face normal.

        :return: The surface normal at the specified coordinate.
        """

        cdef int32_t n1, n2, n3

        if self.smoothing and self.vertex_normals_mv is not None:

            n1 = self.triangles_mv[self._i, N1]
            n2 = self.triangles_mv[self._i, N2]
            n3 = self.triangles_mv[self._i, N3]

            return new_normal3d(
                self._u * self.vertex_normals_mv[n1, X] + self._v * self.vertex_normals_mv[n2, X] + self._w * self.vertex_normals_mv[n3, X],
                self._u * self.vertex_normals_mv[n1, Y] + self._v * self.vertex_normals_mv[n2, Y] + self._w * self.vertex_normals_mv[n3, Y],
                self._u * self.vertex_normals_mv[n1, Z] + self._v * self.vertex_normals_mv[n2, Z] + self._w * self.vertex_normals_mv[n3, Z]
            ).normalise()

        else:

            return new_normal3d(
                self.face_normals_mv[self._i, X],
                self.face_normals_mv[self._i, Y],
                self.face_normals_mv[self._i, Z]
            ).normalise()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef bint contains(self, Point3D p):
        """
        Tests if a point is contained by the mesh.

        Note, this method assumes the mesh is closed. Any open/closed mesh test
        must be performed externally (this is generally quicker as coordinate
        transforms etc... can be skipped if the mesh is open).

        :param p: Local space Point3D.
        :return: True if mesh contains point, False otherwise.
        """

        cdef Ray ray

        # fire ray along z axis, if it encounters a polygon it inspects the orientation of the face
        # if the face is outwards, then the ray was spawned inside the mesh
        # this assumes the mesh has all face normals facing outwards from the mesh interior
        ray = new_ray(p, new_vector3d(0, 0, 1), INFINITY)

        # search for closest triangle intersection
        if not self.trace(ray):
            return False

        # inspect the Z component of the triangle face normal to identify orientation
        # this is an optimised version of ray.direction.dot(face_normal) as we know ray only propagating in Z
        return self.face_normals_mv[self._i, Z] > 0.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef BoundingBox3D bounding_box(self, AffineMatrix3D to_world):
        """
        Returns a bounding box that encloses the mesh.

        The box is padded by a small margin to reduce the risk of numerical
        accuracy problems between the mesh and box representations following
        coordinate transforms.

        :param to_world: Local to world space transform matrix.
        :return: A BoundingBox3D object.
        """

        cdef:
            int32_t i
            BoundingBox3D bbox
            Point3D vertex

        # TODO: padding should really be a function of mesh extent
        # convert vertices to world space and grow a bounding box around them
        bbox = BoundingBox3D()
        for i in range(self.vertices_mv.shape[0]):
            vertex = new_point3d(self.vertices_mv[i, X], self.vertices_mv[i, Y], self.vertices_mv[i, Z])
            bbox.extend(vertex.transform(to_world), BOX_PADDING)

        return bbox

    # TODO: this code is forking horrible - need to split the triangle array into separate components (for a start!)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def save(self, object file):
        """
        Save the mesh's kd-Tree representation to a binary Raysect mesh file (.rsm).

        :param object file: File stream or string file name to save state.
        """

        cdef:
            int32_t i, j
            float32_t[:, ::1] vertices
            float32_t[:, ::1] vertex_normals
            int32_t[:, ::1] triangles

        close = False

        # treat as a filename if a stream is not supplied
        if not isinstance(file, io.IOBase):
            file = open(file, mode="wb")
            close = True

        # hold local references to avoid repeated memory view object checks
        vertices = self.vertices_mv
        vertex_normals = self.vertex_normals_mv
        triangles = self.triangles_mv

        # write header
        file.write(b"RSM")
        file.write(struct.pack("<B", RSM_VERSION_MAJOR))
        file.write(struct.pack("<B", RSM_VERSION_MINOR))

        # mesh setting flags
        file.write(struct.pack("<?", self.smoothing))
        file.write(struct.pack("<?", self.closed))
        file.write(struct.pack("<?", True))    # kdtree in file (hardcoded for now, will be an option)

        # item counts
        file.write(struct.pack("<i", vertices.shape[0]))

        if self.vertex_normals_mv is not None:
            file.write(struct.pack("<i", vertex_normals.shape[0]))
        else:
            file.write(struct.pack("<i", 0))

        file.write(struct.pack("<i", triangles.shape[0]))

        # write vertices
        for i in range(vertices.shape[0]):
            for j in range(3):
                file.write(struct.pack("<f", vertices[i, j]))

        # write normals
        if vertex_normals is not None:
            for i in range(vertex_normals.shape[0]):
                for j in range(3):
                    file.write(struct.pack("<f", vertex_normals[i, j]))

        # triangles
        width = 3
        if vertex_normals is not None:
            # we have vertex normals for each triangle
            width += 3

        for i in range(triangles.shape[0]):
            for j in range(width):
                file.write(struct.pack("<i", triangles[i, j]))

        # write kd-tree
        super().save(file)

        # if we opened a file, we should close it
        if close:
            file.close()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def load(self, object file):
        """
        Load a mesh with its kd-Tree representation from Raysect mesh binary file (.rsm).

        :param object file: File stream or string file name to save state.
        """

        cdef:
            int32_t i, j

        close = False

        # treat as a filename if a stream is not supplied
        if not isinstance(file, io.IOBase):
            file = open(file, mode="rb")
            close = True

        # read and check header
        identifier = file.read(3)
        major_version = self._read_uint8(file)
        minor_version = self._read_uint8(file)

        # validate
        if identifier != b"RSM":
            raise ValueError("Specified file is not a Raysect mesh file.")

        if major_version != RSM_VERSION_MAJOR or minor_version != RSM_VERSION_MINOR:
            raise ValueError("Unsupported Raysect mesh version.")

        # mesh setting flags
        self.smoothing = self._read_bool(file)
        self.closed = self._read_bool(file)
        _ = self._read_bool(file)    # kdtree option, ignore for now (to be implemented)

        # item counts
        num_vertices = self._read_int32(file)
        num_vertex_normals = self._read_int32(file)
        num_triangles = self._read_int32(file)

        # read vertices
        self._vertices = zeros((num_vertices, 3), dtype=float32)
        self.vertices_mv = self._vertices
        for i in range(num_vertices):
            for j in range(3):
                self.vertices_mv[i, j] = self._read_float(file)

        # read vertex normals
        if num_vertex_normals > 0:
            self._vertex_normals = zeros((num_vertex_normals, 3), dtype=float32)
            self.vertex_normals_mv = self._vertex_normals
            for i in range(num_vertex_normals):
                for j in range(3):
                    self.vertex_normals_mv[i, j] = self._read_float(file)

        else:
            self._vertex_normals = None
            self.vertex_normals_mv = None

        # read triangles
        width = 3
        if num_vertex_normals > 0:
            # we have vertex normals for each triangle
            width += 3

        self._triangles = zeros((num_triangles, width), dtype=int32)
        self.triangles_mv = self._triangles
        for i in range(num_triangles):
            for j in range(width):
                self.triangles_mv[i, j] = self._read_int32(file)

        # read kdtree
        super().load(file)

        # generate face normals
        self._generate_face_normals()

        # initial hit data
        self._u = -1.0
        self._v = -1.0
        self._w = -1.0
        self._t = INFINITY
        self._i = NO_INTERSECTION

        # if we opened a file, we should close it
        if close:
            file.close()

    @classmethod
    def from_file(cls, file):
        """
        Load a mesh with its kd-Tree representation from Raysect mesh binary file (.rsm).

        :param object file: File stream or string file name to save state.
        """

        m = MeshData.__new__(MeshData)
        m.load(file)
        return m

    cdef uint8_t _read_uint8(self, object file):
        return (<uint8_t *> PyBytes_AsString(file.read(sizeof(uint8_t))))[0]

    cdef bint _read_bool(self, object file):
        return self._read_uint8(file) != 0

    cdef double _read_float(self, object file):
        return (<float *> PyBytes_AsString(file.read(sizeof(float))))[0]


cdef class Mesh(Primitive):
    """
    This primitive defines a polyhedral surface with triangular faces.

    To define a new mesh, a list of vertices and triangles must be supplied.
    A set of vertex normals, used for smoothing calculations may also be
    provided.

    The mesh vertices are supplied as an Nx3 list/array of floating point
    values. For each Vertex, x, y and z coordinates must be supplied. e.g.

        vertices = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], ...]

    Vertex normals are similarly defined. Note that vertex normals must be
    correctly normalised.

    The triangle array is either Mx3 or Mx6 - Mx3 if only vertices are defined
    or Mx6 if both vertices and vertex normals are defined. Triangles are
    defined by indexing into the vertex and vertex normal arrays. i.e:

        triangles = [[v1, v2, v3, n1, n2, n3], ...]

    where v1, v2, v3 are the vertex array indices specifying the triangle's
    vertices and n1, n2, n3 are the normal array indices specifying the
    triangle's surface normals at each vertex location. Where normals are
    not defined, n1, n2 and n3 are omitted.

    The mesh may be an open surface (which does not enclose a volume) or a
    closed surface (which defines a volume). The nature of the mesh must be
    specified using the closed argument. If closed is True (default) then the
    mesh must be watertight and the face normals must be facing so they point
    out of the volume. If the mesh is open then closed must be set to False.
    Incorrectly setting the closed argument may result in undefined behaviour,
    depending on the application of the ray-tracer.

    If vertex normals are defined for some or all of the triangles of the mesh
    then normal interpolation may be enabled for the mesh. For optical models
    this will result in a (suitably defined) mesh appearing smooth rather than
    faceted. If the triangles do not have vertex normals defined, the smoothing
    argument is ignored.

    An alternate option for creating a new mesh is to create an instance of an
    existing mesh. An instance is a "clone" of the original mesh. Mesh instances
    hold references to the internal data of the target mesh, they are therefore
    very memory efficient (particularly for detailed meshes) compared to
    creating a new mesh from scratch. A new instance of a mesh can be created
    using the instance() method.

    If a mesh contains degenerate triangles (common for meshes generated from
    CAD models), enable tolerant mode to automatically remove them during mesh
    initialisation. A degenerate triangle is one where two or more vertices are
    coincident or all the vertices lie on the same line. Degenerate triangles
    will produce rendering error if encountered even though they are
    "infinitesimally" thin. A ray can still intersect them if they perfectly
    align as the triangle edges are treated as part of the triangle surface).

    The kdtree_* arguments are tuning parameters for the kd-tree construction.
    For more information see the documentation of KDTree3D. The default values
    should result in efficient construction of the mesh's internal kd-tree.
    Generally there is no need to modify these parameters unless the memory
    used by the kd-tree must be controlled. This may occur if very large meshes
    are used.

    :param object vertices: An N x 3 list of vertices.
    :param object triangles: An M x 3 or N x 6 list of vertex/normal indices
      defining the mesh triangles.
    :param object normals: An K x 3 list of vertex normals or None (default=None).
    :param bool smoothing: True to enable normal interpolation (default=True).
    :param bool closed: True is the mesh defines a closed volume (default=True).
    :param bool tolerant: Mesh will automatically correct meshes with degenerate
      triangles if set to True (default=True).
    :param bool flip_normals: Inverts the direction of the surface normals (default=False).
    :param int kdtree_max_depth: The maximum tree depth (automatic if set to 0, default=0).
    :param int kdtree_min_items: The item count threshold for forcing creation of
      a new leaf node (default=1).
    :param double kdtree_hit_cost: The relative computational cost of item hit
      evaluations vs kd-tree traversal (default=20.0).
    :param double kdtree_empty_bonus: The bonus applied to node splits that
      generate empty leaves (default=0.2).
    :param Node parent: Attaches the mesh to the specified scene-graph
      node (default=None).
    :param AffineMatrix3D transform: The co-ordinate transform between
      the mesh and its parent (default=unity matrix).
    :param Material material: The surface/volume material
      (default=Material() instance).
    :param str name: A human friendly name to identity the mesh in the
      scene-graph (default="").

    :ivar MeshData data: A class instance containing all the mesh data.
    """

    # TODO: calculate or measure triangle hit cost vs split traversal
    def __init__(self, object vertices, object triangles, object normals=None,
                 bint smoothing=True, bint closed=True, bint tolerant=True, bint flip_normals=False,
                 int kdtree_max_depth=-1, int kdtree_min_items=1, double kdtree_hit_cost=5.0,
                 double kdtree_empty_bonus=0.25, object parent=None,
                 AffineMatrix3D transform=None, Material material=None, str name=None):

        super().__init__(parent, transform, material, name)

        if vertices is None or triangles is None:
            raise ValueError("Vertices and triangle arrays must be supplied if the mesh is not configured to be an instance.")

        # build the kd-Tree
        self.data = MeshData(vertices, triangles, normals=normals, smoothing=smoothing, closed=closed,
                             tolerant=tolerant, flip_normals=flip_normals, max_depth=kdtree_max_depth,
                             min_items=kdtree_min_items, hit_cost=kdtree_hit_cost, empty_bonus=kdtree_empty_bonus)

        # initialise next intersection search
        self._seek_next_intersection = False
        self._next_world_ray = None
        self._next_local_ray = None
        self._ray_distance = 0

    cpdef object instance(self, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):

        cdef Mesh mesh = Mesh.__new__(Mesh)
        super(Mesh, mesh).__init__(parent, transform, material, name)

        # copy the kd-Tree
        mesh.data = self.data

        # initialise next intersection search
        self._seek_next_intersection = False
        self._next_world_ray = None
        self._next_local_ray = None
        self._ray_distance = 0

        return mesh

    cpdef Intersection hit(self, Ray ray):
        """
        Returns the first intersection with the mesh surface.

        If an intersection occurs this method will return an Intersection
        object. The Intersection object will contain the details of the
        ray-surface intersection, such as the surface normal and intersection
        point.

        If no intersection occurs None is returned.

        :param ray: A world-space ray.
        :return: An Intersection or None.
        """

        cdef Ray local_ray

        local_ray = new_ray(
            ray.origin.transform(self.to_local()),
            ray.direction.transform(self.to_local()),
            ray.max_distance
        )

        # reset accumulated ray distance (used by next_intersection)
        self._ray_distance = 0

        # do we hit the mesh?
        if self.data.trace(local_ray):
            return self._process_intersection(ray, local_ray)

        # there was no intersection so disable next intersection search
        self._seek_next_intersection = False

        return None

    cpdef Intersection next_intersection(self):
        """
        Returns the next intersection of the ray with the mesh along the ray
        path.

        This method may only be called following a call to hit(). If the ray
        has further intersections with the mesh, these may be obtained by
        repeatedly calling the next_intersection() method. Each call to
        next_intersection() will return the next ray-mesh intersection
        along the ray's path. If no further intersections are found or
        intersections lie outside the ray parameters then next_intersection()
        will return None.

        :return: An Intersection or None.
        """

        if self._seek_next_intersection:

            # do we hit the mesh again?
            if self.data.trace(self._next_local_ray):
                return self._process_intersection(self._next_world_ray, self._next_local_ray)

            # there was no intersection so disable further searching
            self._seek_next_intersection = False

        return None

    cdef Intersection _process_intersection(self, Ray world_ray, Ray local_ray):

        cdef:
            Intersection intersection

        # obtain intersection details from the kd-tree
        intersection = self.data.calc_intersection(local_ray)

        # enable next intersection search and cache the local ray for the next intersection calculation
        # we must shift the new origin past the last intersection
        self._seek_next_intersection = True
        self._next_world_ray = world_ray
        self._next_local_ray = new_ray(
            new_point3d(
                intersection.hit_point.x + local_ray.direction.x * EPSILON,
                intersection.hit_point.y + local_ray.direction.y * EPSILON,
                intersection.hit_point.z + local_ray.direction.z * EPSILON
            ),
            local_ray.direction,
            local_ray.max_distance - intersection.ray_distance - EPSILON
        )

        # for next intersection calculations the ray local origin is moved past the last intersection point so
        # we therefore need to add the additional distance between the local ray origin and the original ray origin.
        intersection.ray_distance += self._ray_distance

        # ray origin is shifted to avoid self intersection, account for this in subsequent intersections
        self._ray_distance = intersection.ray_distance + EPSILON

        # fill in missing intersection information
        intersection.primitive = self
        intersection.ray = world_ray
        intersection.world_to_primitive = self.to_local()
        intersection.primitive_to_world = self.to_root()

        return intersection

    cpdef bint contains(self, Point3D p) except -1:
        """
        Identifies if the point lies in the volume defined by the mesh.

        If a mesh is open, this method will always return False.

        This method will fail if the face normals of the mesh triangles are not
        oriented to be pointing out of the volume surface.

        :param p: The point to test.
        :return: True if the point lies in the volume, False otherwise.
        """

        # avoid unnecessary transform by checking closed state early
        if not self.data.closed:
            return False

        p = p.transform(self.to_local())
        return self.data.contains(p)

    cpdef BoundingBox3D bounding_box(self):
        """
        Returns a world space bounding box that encloses the mesh.

        The box is padded by a small margin to reduce the risk of numerical
        accuracy problems between the mesh and box representations following
        coordinate transforms.

        :return: A BoundingBox3D object.
        """

        return self.data.bounding_box(self.to_root())

    def save(self, object file):
        """
        Saves the mesh to the specified file object or filename.

        The mesh in written in RaySect Mesh (RSM) format. The RSM format
        contains the mesh geometry and the mesh acceleration structures.

        :param file: File object or string path.

        .. code-block:: pycon

            >>> mesh
            <raysect.primitive.mesh.mesh.Mesh at 0x7f2c09eac2e8>
            >>> mesh.save("my_mesh.rsm")

        """

        # todo: keeping this here until I re add the kdtree parameter
        # """
        # Saves the mesh to the specified file object or filename.
        #
        # The mesh in written in RaySect Mesh (RSM) format. The RSM format
        # contains the mesh geometry and (optionally) the mesh acceleration
        # structures.
        #
        # By default, the mesh kdtree acceleration structure is stored alongside
        # the mesh geometry. If this is not desired, for instance if storage
        # space is a premium, the kdtree can be omitted by setting the kdtree
        # argument to False. The kdtree will be recalculated when the mesh is
        # next loaded (as is performed when other mesh formats are imported).
        #
        # :param file: File object or string path.
        # """

        # hand over to the mesh data object
        self.data.save(file)

    def load(self, object file):
        """
        Loads the mesh specified by a file object or filename.

        The mesh must be stored in a RaySect Mesh (RSM) format file. RSM files
        are created with the Mesh save() method.

        :param file: File object or string path.
        """

        # rebuild internal state
        self.data = MeshData.from_file(file)
        self._seek_next_intersection = False
        self._next_world_ray = None
        self._next_local_ray = None
        self._ray_distance = 0

    @classmethod
    def from_file(cls, object file, object parent=None,
                  AffineMatrix3D transform=AffineMatrix3D(),
                  Material material=Material(), unicode name=""):
        """
        Instances a new Mesh using data from a file object or filename.

        The mesh must be stored in a RaySect Mesh (RSM) format file. RSM files
        are created with the Mesh save() method.

        :param object file: File object or string path.
        :param Node parent: Attaches the mesh to the specified scene-graph node.
        :param AffineMatrix3D transform: The co-ordinate transform between the mesh and its parent.
        :param Material material: The surface/volume material.
        :param str name: A human friendly name to identity the mesh in the scene-graph.

        .. code-block:: pycon

            >>> from raysect.optical import World, translate, rotate, ConstantSF, Sellmeier, Dielectric
            >>> from raysect.primitive import Mesh
            >>>
            >>> world = World()
            >>>
            >>> diamond = Dielectric(Sellmeier(0.3306, 4.3356, 0.0, 0.1750**2, 0.1060**2, 0.0), ConstantSF(1.0))
            >>>
            >>> mesh = Mesh.from_file("my_mesh.rsm", parent=world, material=diamond,
            >>>                       transform=translate(0, 0, 0)*rotate(165, 0, 0))

        """

        m = Mesh.__new__(Mesh)
        super(Mesh, m).__init__(parent, transform, material, name)
        m.load(file)
        return m


