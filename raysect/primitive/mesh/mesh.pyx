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
from raysect.core.math.point cimport Point
from raysect.core.classes cimport Material, new_intersection
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
* add smoothing parameter and add normal interpolation
* add open/closed mesh support (implement contains using a surface intersection count)

3) release
* add kdtree to optimise hit and contains

Notes:
The ray-triangle intersection is a partial implementation of the algorithm described in:
    "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald, Journal of Computer Graphics Techniques (2013), Vol.2, No. 1

As implemented, the algorithm is not fully watertight due to the use of double precision throughout. At present, there is no appeal to
higher precision to resolve cases when the edge tests result in a degenerate solution. This should only occur when a mesh contains
extremely small triangles that are being tested against a ray with an origin far from the mesh.
"""

cdef class _Triangle:

    cdef:
        readonly Point v1, v2, v3
        readonly Normal n1, n2, n3
        readonly Normal face_normal

    def __init__(self, Point v1 not None, Point v2 not None, Point v3 not None,
                 Normal n1 not None, Normal n2 not None, Normal n3 not None):

        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self._calc_face_normal()

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

    def hit(self, ray):

        # This code is a Python port of the code listed in appendix A of
        #  "Watertight Ray/Triangle Intersection", S.Woop, C.Benthin, I.Wald,
        #  Journal of Computer Graphics Techniques (2013), Vol.2, No. 1

        # this code assumes ray is in local co-ordinates

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
        if ray.direction[iz] < 0.0:
            ix, iy = iy, ix

        # calculate shear transform
        sx = ray.direction[ix] / ray.direction[iz]
        sy = ray.direction[iy] / ray.direction[iz]
        sz = 1.0 / ray.direction[iz]

        # center coordinate space on ray origin
        v1 = Point(self.v1.x - ray.origin.x, self.v1.y - ray.origin.y, self.v1.z - ray.origin.z)
        v2 = Point(self.v2.x - ray.origin.x, self.v2.y - ray.origin.y, self.v2.z - ray.origin.z)
        v3 = Point(self.v3.x - ray.origin.x, self.v3.y - ray.origin.y, self.v3.z - ray.origin.z)

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
        z1 = sz * v1[iz]
        z2 = sz * v2[iz]
        z3 = sz * v3[iz]
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

    # cdef bint side(self, Point p):
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



# todo: get/set attributes must return copies of arrays to protect internals of the mesh object
cdef class Mesh(Primitive):

    def __init__(self, object vertices, object polygons, object parent=None, AffineMatrix transform not None=AffineMatrix(), Material material not None=Material(), unicode name not None=""):

        super().__init__(parent, transform, material, name)

        # convert to internal numpy arrays and obtain memory views

        # validate
            # check vertices
            # check normals are valid
            # check polygons are not dangling

        pass

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

        raise NotImplementedError("Primitive surface has not been defined. Virtual method hit() has not been implemented.")


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

        When the primitive is connected to a scenegrpah containing a World
        object at its root, this method should return a bounding box that
        fully encloses the primitive's surface (plus a small margin to
        avoid numerical accuracy problems). The bounding box must be defined in
        the world's coordinate space.

        If this method is called when the primitive is not connected to a
        scenegraph with a World object at its root, it must throw a TypeError
        exception.
        """

        raise NotImplementedError("Primitive surface has not been defined. Virtual method bounding_box() has not been implemented.")

