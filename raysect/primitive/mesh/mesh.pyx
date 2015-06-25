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

from raysect.core.math.affinematrix cimport AffineMatrix
from raysect.core.math.normal cimport Normal
from raysect.core.math.point cimport Point, new_point
from raysect.core.classes cimport Material, Intersection, new_intersection
from raysect.core.acceleration.boundingbox cimport BoundingBox
from libc.math cimport fabs
cimport cython

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-9

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

    def _calc_face_normal(self):
        """
        Calculate the triangles face normal from the vertices.

        The triangle face normal direction is defined by the right hand screw
        rule. When looking at the triangle from the back face, the vertices
        will be ordered in a clockwise fashion and the normal will be pointing
        away from the observer.
        """

        a = self.v1.vector_to(self.v2)
        b = self.v1.vector_to(self.v3)
        self.face_normal = Normal(*a.cross(b).normalise())

    def interpolate_normal(self, u, v, w, smoothing=True):
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
            return u * self.n1 + v * self.n2 + w * self.n3
        else:
            return self.face_normal


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

    cpdef Intersection hit(self, Ray ray):
        """
        Virtual method - to be implemented by derived classes.

        Calculates the closest intersection of the Ray with the Primitive
        surface, if such an intersection exists.

        If a hit occurs an Intersection object must be returned, otherwise None
        is returned. The intersection object holds the details of the
        intersection including the point of intersection, surface normal and
        the objects involved in the intersection.
        """

        local_ray = Ray(ray.origin.transform(self.to_local()),
                        ray.direction.transform(self.to_local()))

        ix, iy, iz, sx, sy, sz = self._calc_rayspace_transform(local_ray)

        closest = None
        ray_distance = ray.max_distance
        for triangle in self.triangles:
            result = self._hit_triangle(triangle, ix, iy, iz, sx, sy, sz, local_ray)
            if result is not None:
                t, u, v, w = result
                if t < ray_distance:
                    closest = triangle
                    ray_distance = t
                    cu, cv, cw = u, v, w

        if closest is None:
            return None

        hit_point = local_ray.origin + local_ray.direction * ray_distance
        inside_point = hit_point - closest.face_normal * EPSILON
        outside_point = hit_point + closest.face_normal * EPSILON
        normal = closest.interpolate_normal(cu, cv, cw, self.smoothing)
        exiting = local_ray.direction.dot(closest.face_normal) > 0.0
        return Intersection(ray, ray_distance, self,
                            hit_point, inside_point, outside_point,
                            normal, exiting, self.to_local(), self.to_root())

    @cython.cdivision(True)
    cdef tuple _calc_rayspace_transform(self, Ray ray):

        # This code is a Python port of the code listed in appendix A of
        #  "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald,
        #  Journal of Computer Graphics Techniques (2013), Vol.2, No. 1

        cdef:
            int ix, iy, iz
            double rdx, rdy, rdz
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

        rdx = ray.direction.get_index(ix)
        rdy = ray.direction.get_index(iy)
        rdz = ray.direction.get_index(iz)

        # if the z component is negative, swap x and y to restore the handedness of the space
        if rdz < 0.0:
            ix, iy = iy, ix

        # calculate shear transform
        sz = 1 / rdz
        sx = rdx * sz
        sy = rdy * sz

        return ix, iy, iz, sx, sy, sz

    @cython.cdivision(True)
    cdef tuple _hit_triangle(self, Triangle triangle, int ix, int iy, int iz, double sx, double sy, double sz, Ray ray):

        # This code is a Python port of the code listed in appendix A of
        #  "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald,
        #  Journal of Computer Graphics Techniques (2013), Vol.2, No. 1

        cdef:
            Point v1, v2, v3
            double v1z, v2z, v3z
            double x1, x2, x3, y1, y2, y3
            double t, u, v, w
            double det, det_reciprocal

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
            bbox.extend(Point(*triangle.v1).transform(self.to_root()), BOX_PADDING)
            bbox.extend(Point(*triangle.v2).transform(self.to_root()), BOX_PADDING)
            bbox.extend(Point(*triangle.v3).transform(self.to_root()), BOX_PADDING)
        return bbox


