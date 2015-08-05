# cython: language_level=3
# cython: profile=False

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
from raysect.core.math.kdtree cimport KDTreeCore, Item, kdnode
from raysect.core.classes cimport Material, Intersection, Ray, new_intersection, new_ray
from raysect.core.acceleration.boundingbox cimport BoundingBox, new_boundingbox
from libc.math cimport fabs, log, ceil
import io
import pickle
cimport cython

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-9

# additional ray distance to avoid re-hitting the same surface point
DEF EPSILON = 1e-9

"""
Notes:
The ray-triangle intersection is a partial implementation of the algorithm described in:
    "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald, Journal of Computer Graphics Techniques (2013), Vol.2, No. 1

As implemented, the algorithm is not fully watertight due to the use of double precision throughout. At present, there is no appeal to
higher precision to resolve cases when the edge tests result in a degenerate solution. This should only occur when a mesh contains
extremely small triangles that are being tested against a ray with an origin far from the mesh.
"""

cdef class Triangle:

    cdef:
        readonly Point v1, v2, v3
        readonly Normal n1, n2, n3
        readonly Normal face_normal
        bint _smoothing_enabled

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

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.v1, self.v2, self.v3, self.n1, self.n2, self.n3, self.face_normal, self._smoothing_enabled

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.v1, self.v2, self.v3, self.n1, self.n2, self.n3, self.face_normal, self._smoothing_enabled = state

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

    @cython.cdivision(True)
    cpdef Point centre_point(self):
        """
        Returns the centre point (centroid) of the triangle.
        :return: Point object.
        """

        return new_point(
            (self.v1.x + self.v2.x + self.n3.x) / 3,
            (self.v1.y + self.v2.y + self.n3.y) / 3,
            (self.v1.z + self.v2.z + self.n3.z) / 3
        )

    cpdef Normal interpolate_normal(self, double u, double v, double w, bint smoothing=True):
        """
        Returns the surface normal for the specified barycentric coordinate.

        The result is undefined if u, v or w are outside the range [0, 1].
        If smoothing is disabled the result will be the face normal.

        :param u: Barycentric U coordinate.
        :param v: Barycentric V coordinate.
        :param w: Barycentric W coordinate.
        :return: The surface normal at the specified coordinate.
        """

        if smoothing and self._smoothing_enabled:
            return new_normal(
                u * self.n1.x + v * self.n2.x + w * self.n3.x,
                u * self.n1.y + v * self.n2.y + w * self.n3.y,
                u * self.n1.z + v * self.n2.z + w * self.n3.z,
            )
        else:
            return self.face_normal

    cpdef double lower_extent(self, int axis):
        """
        Returns the lowest extent of the triangle along the specified axis.
        """

        return min(self.v1.get_index(axis),
                   self.v2.get_index(axis),
                   self.v3.get_index(axis))

    cpdef double upper_extent(self, int axis):
        """
        Returns the upper extent of the triangle along the specified axis.
        """

        return max(self.v1.get_index(axis),
                   self.v2.get_index(axis),
                   self.v3.get_index(axis))

    cpdef BoundingBox bounding_box(self):
        """
        Returns a bounding box enclosing the triangle.

        The box is defined in the triangle's coordinate system. A small degree
        of padding is added to the bounding box to provide the conservative
        bounds required by the watertight mesh algorithm.

        :return: A BoundingBox object.
        """

        bbox = new_boundingbox(
            Point(
                min(self.v1.x, self.v2.x, self.v3.x),
                min(self.v1.y, self.v2.y, self.v3.y),
                min(self.v1.z, self.v2.z, self.v3.z),
            ),
            Point(
                max(self.v1.x, self.v2.x, self.v3.x),
                max(self.v1.y, self.v2.y, self.v3.y),
                max(self.v1.z, self.v2.z, self.v3.z),
            ),
        )
        bbox.pad(bbox.largest_extent() * BOX_PADDING)
        return bbox


cdef class MeshKDTree(KDTreeCore):

    cdef:
        list triangles
        tuple _hit_ray_transform
        readonly tuple hit_intersection

    def __init__(self, list triangles, int max_depth=0, int min_items=1, double hit_cost=20.0, double empty_bonus=0.2):

        self.triangles = triangles
        self._hit_ray_transform = None
        self.hit_intersection = None

        # kd-Tree init requires the triangle's id (it's index here) and bounding box
        items = []
        for id, triangle in enumerate(triangles):
            items.append(Item(id, triangle.bounding_box()))

        super().__init__(items, max_depth, min_items, hit_cost, empty_bonus)

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.triangles, super().__getstate__()

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.triangles, base_state = state
        super().__setstate__(base_state)

    cpdef bint hit(self, Ray ray):

        self.hit_intersection = None
        self._calc_rayspace_transform(ray)
        return self._hit(ray)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _hit_leaf(self, int id, Ray ray, double max_range):

        cdef:
            int count, item, index
            double distance
            double t, u, v, w
            Triangle triangle, closest_triangle
            tuple intersection, closest_intersection

        # unpack leaf data
        count = self._nodes[id].count

        # find the closest triangle-ray intersection with initial search distance limited by node and ray limits
        distance = min(ray.max_distance, max_range)
        closest_intersection = None
        closest_triangle = None
        for item in range(count):

            # dereference the triangle
            index = self._nodes[id].items[item]
            triangle = self.triangles[index]

            # test for intersection
            intersection = self._hit_triangle(triangle, ray)
            if intersection is not None and intersection[0] < distance:
                distance = intersection[0]
                closest_triangle = triangle
                closest_intersection = intersection

        if closest_intersection is None:
            return False

        t, u, v, w = closest_intersection
        self.hit_intersection = closest_triangle, t, u, v, w
        return True

    @cython.cdivision(True)
    cdef void _calc_rayspace_transform(self, Ray ray):

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

        # store ray transform
        self._hit_ray_transform = ix, iy, iz, sx, sy, sz

    @cython.cdivision(True)
    cdef tuple _hit_triangle(self, Triangle triangle, Ray ray):

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
        ix, iy, iz, sx, sy, sz = self._hit_ray_transform

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

    :param triangles: A list of Triangles defining the mesh.
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
        readonly bint smoothing
        readonly bint closed
        bint _seek_next_intersection
        Ray _next_world_ray
        Ray _next_local_ray

    # TODO: calculate or measure triangle hit cost vs split traversal
    def __init__(self, list triangles=None, bint smoothing=True, bint closed=True, Mesh instance=None, int kdtree_max_depth=-1, int kdtree_min_items=1, double kdtree_hit_cost=5.0, double kdtree_empty_bonus=0.25, object parent=None, AffineMatrix transform not None=AffineMatrix(), Material material not None=Material(), unicode name not None=""):

        super().__init__(parent, transform, material, name)

        if instance:
            # hold references to internal data of the specified mesh
            self.smoothing = instance.smoothing
            self.closed = instance.closed
            self._kdtree = instance._kdtree

        else:

            if triangles is None:
                triangles = []

            self.smoothing = smoothing
            self.closed = closed

            # build the kd-Tree
            self._kdtree = MeshKDTree(triangles, kdtree_max_depth, kdtree_min_items, kdtree_hit_cost, kdtree_empty_bonus)

        # initialise next intersection search
        self._seek_next_intersection = False
        self._next_world_ray = None
        self._next_local_ray = None

    property triangles:

        def __get__(self):

            # return a copy to prevent users altering the list
            # if the list size is altered it could cause a segfault
            return self._kdtree.triangles.copy()

    cpdef Intersection hit(self, Ray ray):
        """
        Returns the first intersection with the mesh surface.

        If an intersection occurs this method will return an Intersection
        object. The Intersection object will contain the details of the
        ray-surface intersection, sucah as the surface normal and intersection
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
            Triangle triangle
            double t, u, v, w
            Point hit_point, inside_point, outside_point
            Normal normal
            bint exiting

        # on a hit the kd-tree populates an attribute containing the intersection data, unpack it
        triangle, t, u, v, w = self._kdtree.hit_intersection

        # generate intersection description
        hit_point = local_ray.origin + local_ray.direction * t
        inside_point = hit_point - triangle.face_normal * EPSILON
        outside_point = hit_point + triangle.face_normal * EPSILON
        normal = triangle.interpolate_normal(u, v, w, self.smoothing)
        exiting = local_ray.direction.dot(triangle.face_normal) > 0.0

        # enable next intersection search and cache the local ray for the next intersection calculation
        # we must shift the new origin past the last intersection
        self._seek_next_intersection = True
        self._next_world_ray = world_ray
        self._next_local_ray = new_ray(
            hit_point + local_ray.direction * EPSILON,
            local_ray.direction,
            local_ray.max_distance - t - EPSILON
        )

        return new_intersection(
            world_ray, t, self,
            hit_point, inside_point, outside_point,
            normal, exiting, self.to_local(), self.to_root()
        )

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

        cdef:
            Ray ray
            bint hit
            double min_range, max_range
            Triangle triangle
            double t, u, v, w

        if not self.closed:
            return False

        # fire ray along z axis, if it encounters a polygon it inspects the orientation of the face
        # if the face is outwards, then the ray was spawned inside the mesh
        # this assumes the mesh has all face normals facing outwards from the mesh interior
        ray = new_ray(
            p.transform(self.to_local()),
            new_vector(0, 0, 1),
            INFINITY
        )

        # search for closest triangle intersection
        if not self._kdtree.hit(ray):
            return False

        triangle, t, u, v, w = self._kdtree.hit_intersection
        return triangle.face_normal.dot(ray.direction) > 0.0

    cpdef BoundingBox bounding_box(self):
        """
        Returns a world space bounding box that encloses the mesh.

        The box is padded by a small margin to reduce the risk of numerical
        accuracy problems between the mesh and box representations following
        coordinate transforms.

        :return: A BoundingBox object.
        """

        cdef:
            BoundingBox bbox
            Triangle triangle

        # TODO: reconsider the padding - the padding should a multiple of max extent, not a fixed value
        bbox = BoundingBox()
        for triangle in self._kdtree.triangles:
            bbox.extend(triangle.v1.transform(self.to_root()), BOX_PADDING)
            bbox.extend(triangle.v2.transform(self.to_root()), BOX_PADDING)
            bbox.extend(triangle.v3.transform(self.to_root()), BOX_PADDING)
        return bbox

    cpdef dump(self, file):
        """
        Writes the mesh data to the specified file descriptor or filename.

        This method can be used as part of a caching system to avoid the
        computational cost of building a mesh's kd-tree. The kd-tree is stored
        with the mesh data and is restored when the mesh is loaded.

        This method may be supplied with a file object or a string path.

        :param file: File object or string path.
        """

        state = (self._kdtree, self.smoothing, self.closed)

        if isinstance(file, io.BytesIO):
             pickle.dump(state, file)
        else:
            with open(file, mode="wb") as f:
                pickle.dump(state, f)

    cpdef load(self, file):
        """
        Reads the mesh data from the specified file descriptor or filename.

        This method can be used as part of a caching system to avoid the
        computational cost of building a mesh's kd-tree. The kd-tree is stored
        with the mesh data and is restored when the mesh is loaded.

        This method may be supplied with a file object or a string path.

        :param file: File object or string path.
        """

        if isinstance(file, io.BytesIO):
             state = pickle.load(file)
        else:
            with open(file, mode="rb") as f:
                state = pickle.load(f)

        self._kdtree, self.smoothing, self.closed = state
        self._seek_next_intersection = False

