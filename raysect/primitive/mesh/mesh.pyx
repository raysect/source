# cython: language_level=3

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

from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.math.affinematrix cimport AffineMatrix
from raysect.core.math.normal cimport Normal, new_normal
from raysect.core.math.point cimport Point, new_point
from raysect.core.math.vector cimport Vector, new_vector
from raysect.core.math.kdtree cimport KDTreeCore, Item
from raysect.core.classes cimport Material, Intersection, Ray, new_intersection, new_ray
from raysect.core.acceleration.boundingbox cimport BoundingBox, new_boundingbox
from libc.math cimport fabs, log, ceil
from numpy import array, float32, int64, zeros
from numpy cimport ndarray, float32_t, int64_t
import io
import pickle
cimport cython

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-6

# additional ray distance to avoid re-hitting the same surface point
DEF EPSILON = 1e-6

# handy defines
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

"""
Notes:
The ray-triangle intersection is a partial implementation of the algorithm described in:
    "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald, Journal of Computer Graphics Techniques (2013), Vol.2, No. 1

As implemented, the algorithm is not fully watertight due to the use of double precision throughout. At present, there is no appeal to
higher precision to resolve cases when the edge tests result in a degenerate solution. This should only occur when a mesh contains
extremely small triangles that are being tested against a ray with an origin far from the mesh.
"""

cdef class MeshKDTree(KDTreeCore):

    cdef:
        float32_t[:, ::1] vertices
        float32_t[:, ::1] vertex_normals
        float32_t[:, ::1] face_normals
        int64_t[:, ::1] triangles
        public bint smoothing
        int _ix, _iy, _iz
        float _sx, _sy, _sz
        float _u, _v, _w, _t
        int _i

    def __init__(self, object vertices, object triangles, object normals=None, bint smoothing=True, int max_depth=0, int min_items=1, double hit_cost=20.0, double empty_bonus=0.2):

        self.smoothing = smoothing

        # convert to numpy arrays for internal use
        vertices = array(vertices, dtype=float32)
        triangles = array(triangles, dtype=int64)
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

        # ensure vertex normals are normalised
        #if vertex_normals is not None:
            # TODO: write me

        # assign to memory views
        self.vertices = vertices
        self.vertex_normals = vertex_normals
        self.triangles = triangles

        # initial hit data
        self._u = -1.0
        self._v = -1.0
        self._w = -1.0
        self._t = INFINITY
        self._i = NO_INTERSECTION

        # generate face normals
        self._generate_face_normals()

        # kd-Tree init requires the triangle's id (it's index here) and bounding box
        items = []
        for i in range(self.triangles.shape[0]):
            items.append(Item(i, self._generate_bounding_box(i)))

        super().__init__(items, max_depth, min_items, hit_cost, empty_bonus)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _generate_face_normals(self):
        """
        Calculate the triangles face normals from the vertices.

        The triangle face normal direction is defined by the right hand screw
        rule. When looking at the triangle from the back face, the vertices
        will be ordered in a clockwise fashion and the normal will be pointing
        away from the observer.
        """

        cdef:
            float32_t[:, ::1] vertices
            int64_t[:, ::1] triangles
            int i
            int i1, i2, i3
            Point p1, p2, p3
            Vector v1, v2, v3

        # assign locally to avoid repeated memory view validity checks
        vertices = self.vertices
        triangles = self.triangles

        self.face_normals = zeros((self.triangles.shape[0], 3), dtype=float32)
        for i in range(self.face_normals.shape[0]):

            i1 = triangles[i, V1]
            i2 = triangles[i, V2]
            i3 = triangles[i, V3]

            p1 = new_point(vertices[i1, X], vertices[i1, Y], vertices[i1, Z])
            p2 = new_point(vertices[i2, X], vertices[i2, Y], vertices[i2, Z])
            p3 = new_point(vertices[i3, X], vertices[i3, Y], vertices[i3, Z])

            v1 = p1.vector_to(p2)
            v2 = p1.vector_to(p3)
            v3 = v1.cross(v2).normalise()

            self.face_normals[i, X] = v3.x
            self.face_normals[i, Y] = v3.y
            self.face_normals[i, Z] = v3.z

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef BoundingBox _generate_bounding_box(self, int i):
        """
        Generates a bounding box for the specified triangle.

        A small degree of padding is added to the bounding box to provide the
        conservative bounds required by the watertight mesh algorithm.

        :param i: Triangle array index.
        :return: A BoundingBox object.
        """

        cdef:
            float32_t[:, ::1] vertices
            int64_t[:, ::1] triangles
            int i1, i2, i3
            BoundingBox bbox

        # assign locally to avoid repeated memory view validity checks
        vertices = self.vertices
        triangles = self.triangles

        i1 = triangles[i, V1]
        i2 = triangles[i, V2]
        i3 = triangles[i, V3]

        bbox = new_boundingbox(
            new_point(
                min(vertices[i1, X], vertices[i2, X], vertices[i3, X]),
                min(vertices[i1, Y], vertices[i2, Y], vertices[i3, Y]),
                min(vertices[i1, Z], vertices[i2, Z], vertices[i3, Z]),
            ),
            new_point(
                max(vertices[i1, X], vertices[i2, X], vertices[i3, X]),
                max(vertices[i1, Y], vertices[i2, Y], vertices[i3, Y]),
                max(vertices[i1, Z], vertices[i2, Z], vertices[i3, Z]),
            ),
        )
        bbox.pad(max(BOX_PADDING, bbox.largest_extent() * BOX_PADDING))

        return bbox

    # def __getstate__(self):
    #     """Encodes state for pickling."""
    #
    #     return self.triangles, super().__getstate__()
    #
    # def __setstate__(self, state):
    #     """Decodes state for pickling."""
    #
    #     self.triangles, base_state = state
    #     super().__setstate__(base_state)

    cpdef bint hit(self, Ray ray):

        # reset hit data
        self._u = -1.0
        self._v = -1.0
        self._w = -1.0
        self._t = INFINITY
        self._i = NO_INTERSECTION

        self._calc_rayspace_transform(ray)
        return self._hit(ray)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _hit_leaf(self, int id, Ray ray, double max_range):

        cdef:
            float hit_data[4]
            int count, item, index
            double distance
            double u, v, w, t
            int triangle, closest_triangle

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
            int ix, iy, iz
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
    cdef bint _hit_triangle(self, int i, Ray ray, float[4] hit_data):

        # This code is a Python port of the code listed in appendix A of
        #  "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald,
        #  Journal of Computer Graphics Techniques (2013), Vol.2, No. 1

        cdef:
            float32_t[:, ::1] vertices
            int64_t[:, ::1] triangles
            int i1, i2, i3
            int ix, iy, iz
            float sx, sy, sz
            float[3] v1, v2, v3
            float x1, x2, x3
            float y1, y2, y3
            float z1, z2, z3
            float t, u, v, w
            float det, det_reciprocal

        # assign locally to avoid repeated memory view validity checks
        vertices = self.vertices
        triangles = self.triangles

        # obtain vertex ids
        i1 = triangles[i, V1]
        i2 = triangles[i, V2]
        i3 = triangles[i, V3]

        # center coordinate space on ray origin
        v1[X] = vertices[i1, X] - ray.origin.x
        v1[Y] = vertices[i1, Y] - ray.origin.y
        v1[Z] = vertices[i1, Z] - ray.origin.z

        v2[X] = vertices[i2, X] - ray.origin.x
        v2[Y] = vertices[i2, Y] - ray.origin.y
        v2[Z] = vertices[i2, Z] - ray.origin.z

        v3[X] = vertices[i3, X] - ray.origin.x
        v3[Y] = vertices[i3, Y] - ray.origin.y
        v3[Z] = vertices[i3, Z] - ray.origin.z

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
    cpdef Intersection calc_intersection(self, Ray ray):

        cdef:
            double t
            int triangle
            Point hit_point, inside_point, outside_point
            Normal face_normal, normal
            bint exiting

        # on a hit the kd-tree populates attributes containing the intersection data
        t = self._t
        triangle = self._i

        if triangle == NO_INTERSECTION:
            return None

        # generate intersection description
        face_normal = new_normal(
            self.face_normals[triangle, X],
            self.face_normals[triangle, Y],
            self.face_normals[triangle, Z]
        )
        hit_point = new_point(
            ray.origin.x + ray.direction.x * t,
            ray.origin.y + ray.direction.y * t,
            ray.origin.z + ray.direction.z * t
        )
        inside_point = new_point(
            hit_point.x - face_normal.x * EPSILON,
            hit_point.y - face_normal.y * EPSILON,
            hit_point.z - face_normal.z * EPSILON
        )
        outside_point = new_point(
            hit_point.x + face_normal.x * EPSILON,
            hit_point.y + face_normal.y * EPSILON,
            hit_point.z + face_normal.z * EPSILON
        )
        normal = self._intersection_normal()
        exiting = ray.direction.dot(face_normal) > 0.0

        return new_intersection(
            ray, t, None,
            hit_point, inside_point, outside_point,
            normal, exiting, None, None
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Normal _intersection_normal(self):
        """
        Returns the surface normal for the last triangle hit.

        The result is undefined if this method is called when a triangle has
        not been hit (u, v or w are outside the range [0, 1]). If smoothing is
        disabled the result will be the face normal.

        :return: The surface normal at the specified coordinate.
        """

        cdef:
            int64_t[:, ::1] triangles
            float32_t[:, ::1] vertex_normals
            float32_t[:, ::1] face_normals
            int n1, n2, n3

        # assign locally to avoid repeated memory view validity checks
        vertex_normals = self.vertex_normals

        if self.smoothing and vertex_normals is not None:

            # assign locally to avoid repeated memory view validity checks
            triangles = self.triangles

            n1 = triangles[self._i, N1]
            n2 = triangles[self._i, N2]
            n3 = triangles[self._i, N3]

            return new_normal(
                self._u * vertex_normals[n1, X] + self._v * vertex_normals[n2, X] + self._w * vertex_normals[n3, X],
                self._u * vertex_normals[n1, Y] + self._v * vertex_normals[n2, Y] + self._w * vertex_normals[n3, Y],
                self._u * vertex_normals[n1, Z] + self._v * vertex_normals[n2, Z] + self._w * vertex_normals[n3, Z]
            )

        else:

            # assign locally to avoid repeated memory view validity checks
            face_normals = self.face_normals

            return new_normal(
                face_normals[self._i, X],
                face_normals[self._i, Y],
                face_normals[self._i, Z]
            )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef BoundingBox bounding_box(self, AffineMatrix to_world):
        """
        Returns a bounding box that encloses the mesh.

        The box is padded by a small margin to reduce the risk of numerical
        accuracy problems between the mesh and box representations following
        coordinate transforms.

        :param to_world: Local to world space transform matrix.
        :return: A BoundingBox object.
        """

        cdef:
            float32_t[:, ::1] vertices
            int i
            BoundingBox bbox
            Point vertex

        # assign locally to avoid repeated memory view validity checks
        vertices = self.vertices

        # TODO: padding should really be a function of mesh extent
        # convert vertices to world space and grow a bounding box around them
        bbox = BoundingBox()
        for i in range(vertices.shape[0]):
            vertex = new_point(vertices[i, X], vertices[i, Y], vertices[i, Z])
            bbox.extend(vertex.transform(to_world), BOX_PADDING)

        return bbox


cdef class Mesh(Primitive):
    """
    This primitive defines a polyhedral surface with triangular faces.

    To define a mesh, a list of Triangle objects must be supplied.

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
    existing mesh. An instance is a "clone" of the original mesh. Instances
    hold references to the internal data of the target mesh, they are therefore
    very memory efficient (particularly for detailed meshes) compared to
    creating a new mesh from scratch. If instance is set, it takes precedence
    over any other mesh creation settings.

    The kdtree_* arguments are tuning parameters for the kd-tree construction.
    For more information see the documentation of KDTree. The default values
    should result in efficient construction of the mesh's internal kd-tree.
    Generally there is no need to modify these parameters unless the memory
    used by the kd-tree must be controlled. This may occur if very large meshes
    are used.

    # :param triangles: A list of Triangles defining the mesh.
    #
    #
    :param smoothing: True to enable normal interpolation, False to disable.
    :param closed: True is the mesh defines a closed volume, False otherwise.
    :param instance: The Mesh to become an instance of.
    :param kdtree_max_depth: The maximum tree depth (automatic if set to 0, default is 0).
    :param kdtree_min_items: The item count threshold for forcing creation of a new leaf node (default 1).
    :param kdtree_hit_cost: The relative computational cost of item hit evaluations vs kd-tree traversal (default 20.0).
    :param kdtree_empty_bonus: The bonus applied to node splits that generate empty leaves (default 0.2).
    :param parent: Attaches the mesh to the specified scene-graph node.
    :param transform: The co-ordinate transform between the mesh and its parent.
    :param material: The surface/volume material.
    :param name: A human friendly name to identity the mesh in the scene-graph.
    :return:
    """

    cdef:
        MeshKDTree _kdtree
        bint closed
        bint _seek_next_intersection
        Ray _next_world_ray
        Ray _next_local_ray
        double _ray_distance

    # TODO: calculate or measure triangle hit cost vs split traversal
    def __init__(self, object vertices, object triangles, object normals=None, bint smoothing=True, bint closed=True, Mesh instance=None, int kdtree_max_depth=-1, int kdtree_min_items=1, double kdtree_hit_cost=5.0, double kdtree_empty_bonus=0.25, object parent=None, AffineMatrix transform not None=AffineMatrix(), Material material not None=Material(), unicode name not None=""):

        super().__init__(parent, transform, material, name)

        if instance:

            # hold references to internal data of the specified mesh
            self.closed = instance.closed
            self._kdtree = instance._kdtree

        else:

            self.closed = closed

            # build the kd-Tree
            self._kdtree = MeshKDTree(vertices, triangles, normals, smoothing, kdtree_max_depth, kdtree_min_items, kdtree_hit_cost, kdtree_empty_bonus)

        # initialise next intersection search
        self._seek_next_intersection = False
        self._next_world_ray = None
        self._next_local_ray = None
        self._ray_distance = 0

    # property triangles:
    #
    #     def __get__(self):
    #
    #         # return a copy to prevent users altering the list
    #         # if the list size is altered it could cause a segfault
    #         return self._kdtree.triangles.copy()

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
        if self._kdtree.hit(local_ray):
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
            if self._kdtree.hit(self._next_local_ray):
                return self._process_intersection(self._next_world_ray, self._next_local_ray)

            # there was no intersection so disable further searching
            self._seek_next_intersection = False

        return None

    cdef Intersection _process_intersection(self, Ray world_ray, Ray local_ray):

        cdef:
            Intersection intersection

        # obtain intersection details from the kd-tree
        intersection = self._kdtree.calc_intersection(local_ray)

        # enable next intersection search and cache the local ray for the next intersection calculation
        # we must shift the new origin past the last intersection
        self._seek_next_intersection = True
        self._next_world_ray = world_ray
        self._next_local_ray = new_ray(
            new_point(
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
        intersection.to_local = self.to_local()
        intersection.to_world = self.to_root()

        return intersection

    # TODO: add an option to use an intersection count algorithm for meshes that have bad face normal orientations
    cpdef bint contains(self, Point p) except -1:
        """
        Identifies if the point lies in the volume defined by the mesh.

        If a mesh is open, this method will always return False.

        This method will fail if the face normals of the mesh triangles are not
        oriented to be pointing out of the volume surface.

        :param p: The point to test.
        :return: True if the point lies in the volume, False otherwise.
        """

        # cdef:
        #     Ray ray
        #     bint hit
        #     double min_range, max_range
        #     Triangle triangle
        #     double t, u, v, w
        #
        # if not self.closed:
        #     return False
        #
        # # fire ray along z axis, if it encounters a polygon it inspects the orientation of the face
        # # if the face is outwards, then the ray was spawned inside the mesh
        # # this assumes the mesh has all face normals facing outwards from the mesh interior
        # ray = new_ray(
        #     p.transform(self.to_local()),
        #     new_vector(0, 0, 1),
        #     INFINITY
        # )
        #
        # # search for closest triangle intersection
        # if not self._kdtree.hit(ray):
        #     return False
        #
        # triangle, t, u, v, w = self._kdtree.hit_intersection
        # return triangle.face_normal.dot(ray.direction) > 0.0
        return False

    cpdef BoundingBox bounding_box(self):
        """
        Returns a world space bounding box that encloses the mesh.

        The box is padded by a small margin to reduce the risk of numerical
        accuracy problems between the mesh and box representations following
        coordinate transforms.

        :return: A BoundingBox object.
        """

        return self._kdtree.bounding_box(self.to_root())


    # cpdef dump(self, file):
    #     """
    #     Writes the mesh data to the specified file descriptor or filename.
    #
    #     This method can be used as part of a caching system to avoid the
    #     computational cost of building a mesh's kd-tree. The kd-tree is stored
    #     with the mesh data and is restored when the mesh is loaded.
    #
    #     This method may be supplied with a file object or a string path.
    #
    #     :param file: File object or string path.
    #     """
    #
    #     state = (self._kdtree, self.smoothing, self.closed)
    #
    #     if isinstance(file, io.BytesIO):
    #          pickle.dump(state, file)
    #     else:
    #         with open(file, mode="wb") as f:
    #             pickle.dump(state, f)
    #
    # cpdef load(self, file):
    #     """
    #     Reads the mesh data from the specified file descriptor or filename.
    #
    #     This method can be used as part of a caching system to avoid the
    #     computational cost of building a mesh's kd-tree. The kd-tree is stored
    #     with the mesh data and is restored when the mesh is loaded.
    #
    #     This method may be supplied with a file object or a string path.
    #
    #     :param file: File object or string path.
    #     """
    #
    #     if isinstance(file, io.BytesIO):
    #          state = pickle.load(file)
    #     else:
    #         with open(file, mode="rb") as f:
    #             state = pickle.load(f)
    #
    #     self._kdtree, self.smoothing, self.closed = state
    #     self._seek_next_intersection = False
