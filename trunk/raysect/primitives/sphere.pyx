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


cimport cython
from raysect.core.math.point cimport new_point
from raysect.core.math.normal cimport new_normal, Normal
from raysect.core.math.vector cimport Vector
from libc.math cimport sqrt

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-9

# suggested ray distance to avoid rehitting the same surface point
DEF EPSILON = 1e-9

cdef class Sphere(Primitive):

    def __init__(self, double radius = 1.0, object parent = None, AffineMatrix transform not None = AffineMatrix(), Material material not None = Material(), unicode name not None= ""):

        super().__init__(parent, transform, material, name)

        if radius < 0.0:

            raise ValueError("Sphere radius cannot be less than zero.")

        self._radius = radius

    property radius:

        def __get__(self):

            return self._radius

        def __set__(self, double radius):

            if radius == self._radius:

                return

            if radius < 0.0:

                raise ValueError("Sphere radius cannot be less than zero.")

            self._radius = radius

            # any geometry caching in the root node is now invalid, inform root
            self.root._change(self)

    cpdef Intersection hit(self, Ray ray):

        cdef Point origin, hit_point
        cdef Normal normal
        cdef Vector direction
        cdef double a, b, c, d, q, t0, t1, temp, t_closest
        cdef Intersection intersection

        # convert ray parameters to local space
        origin = ray.origin.transform(self.to_local())
        direction = ray.direction.transform(self.to_local())

        # coefficients of quadratic equation and discriminant
        a = (direction.x * direction.x
           + direction.y * direction.y
           + direction.z * direction.z)

        b = 2 * (direction.x * origin.x
               + direction.y * origin.y
               + direction.z * origin.z)

        c = (origin.x * origin.x
           + origin.y * origin.y
           + origin.z * origin.z
           - self._radius * self._radius)

        d = b*b - 4*a*c

        # ray misses sphere if there are no real roots of the quadratic
        if d < 0:

            return None

        # calculate intersection distances using method described in the book:
        # "Physically Based Rendering - 2nd Edition", Elsevier 2010
        # this method is more numerically stable than the usual root equation
        if b < 0:

            q = -0.5 * (b - sqrt(d))

        else:

            q = -0.5 * (b + sqrt(d))

        with cython.cdivision(True):

            t0 = q / a
            t1 = c / q

        # ensure t0 is always smaller than t1
        if t0 > t1:

            # swap
            temp = t0
            t0 = t1
            t1 = temp

        # test the intersection points inside the ray search range
        if (t0 > ray.max_distance) or (t1 < ray.min_distance):

            return None

        if t0 >= ray.min_distance:

            t_closest = t0

        elif t1 <= ray.max_distance:

            t_closest = t1

        else:

            return None

        # point of surface intersection in local space
        hit_point = new_point(origin.x + t_closest * direction.x,
                              origin.y + t_closest * direction.y,
                              origin.z + t_closest * direction.z)

        # normal is normalised vector from sphere origin to hit_point
        normal = new_normal(hit_point.x, hit_point.y, hit_point.z)
        normal = normal.normalise()

        intersection = Intersection()
        intersection.primitive = self
        intersection.hit_point = hit_point
        intersection.surface_normal = normal
        intersection.ray = ray
        intersection.ray_distance = t_closest
        intersection.ray_epsilon = EPSILON

        return intersection

    cpdef bint inside(self, Point p) except -1:

        cdef Point local_point
        cdef double distance_sqr

        # convert world space point to local space
        local_point = p.transform(self.to_local())

        # calculate squared distance of the point from the sphere origin
        distance_sqr = (local_point.x * local_point.x
                      + local_point.y * local_point.y
                      + local_point.z * local_point.z)

        # point is outside sphere if point distance is greater than the radius
        # compare squares to avoid costly square root calculation
        if distance_sqr > self._radius * self._radius:

            return False

        return True

    cpdef BoundingBox bounding_box(self):

        cdef double extent
        cdef Point origin, lower, upper

        # obtain sphere origin in world space
        origin = new_point(0, 0, 0).transform(self.to_root())

        # calculate upper and lower corners of box
        extent = self._radius + BOX_PADDING
        lower = new_point(origin.x - extent, origin.y - extent, origin.z - extent)
        upper = new_point(origin.x + extent, origin.y + extent, origin.z + extent)

        return BoundingBox(self, lower, upper)
