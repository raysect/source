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

# TODO: add docstrings

from libc.math cimport fabs
cimport cython

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

cdef class BoundingBox:
    """
    Axis aligned bounding box.

    Represents a bounding box around a primitive's surface. The points defining
    the lower and upper corners of the box must be specified in world space.

    Axis aligned bounding box ray intersections are extremely fast to evaluate
    compared to intersections with more general geometry. Prior to testing a
    primitives hit() method the hit() method of the bounding box is called. If
    the bounding box is not hit, then the expensive primitive hit() method is
    avoided.

    Combined with a spatial subdivision acceleration structure, the cost of ray-
    primitive evaluations can be heavily reduced (O(n) -> O(log n)).

    For optimal speed the bounding box is aligned with the world space axes. As
    rays are propagated in world space, co-ordinate transforms can be avoided.
    """

    def __init__(self, Primitive primitive not None, Point lower not None, Point upper not None):

        if lower.x > upper.x or lower.y > upper.y or lower.z > upper.z:

            raise ValueError("The lower point coordinates must be less than or equal to the upper point coordinates.")

        self.lower = lower
        self.upper = upper
        self.primitive = primitive

    property lower:

        def __get__(self):

            return self.lower

        def __set__(self, Point value not None):

            self.lower = value

    property upper:

        def __get__(self):

            return self.upper

        def __set__(self, Point value not None):

            self.upper = value

    property primitive:

        def __get__(self):

            return self.primitive

        def __set__(self, Primitive value not None):

            self.primitive = value

    cpdef bint hit(self, Ray ray):

        cdef double front_intersection, back_intersection

        # set initial ray-slab intersection search range
        front_intersection = -INFINITY
        back_intersection = INFINITY

        # evaluate ray-slab intersection for x, y and z dimensions and update the intersection positions
        self._slab(ray.origin.x, ray.direction.x, self.lower.x, self.upper.x, &front_intersection, &back_intersection)
        self._slab(ray.origin.y, ray.direction.y, self.lower.y, self.upper.y, &front_intersection, &back_intersection)
        self._slab(ray.origin.z, ray.direction.z, self.lower.z, self.upper.z, &front_intersection, &back_intersection)

        # does ray intersect box?
        if front_intersection > back_intersection:
            return False

        # are both intersections behind ray origin?
        if (front_intersection < 0.0) and (back_intersection < 0.0):
            return False

        return True

    cdef inline void _slab(self, double origin, double direction, double lower, double upper, double *front_intersection, double *back_intersection):

        cdef double reciprocal, tmin, tmax

        if direction != 0.0:

            # calculate intersections with slab planes
            with cython.cdivision(True):
                reciprocal = 1.0 / direction

            if direction > 0:

                # calculate length along ray path of intersections
                tmin = (lower - origin) * reciprocal
                tmax = (upper - origin) * reciprocal

            else:

                # calculate length along ray path of intersections
                tmin = (upper - origin) * reciprocal
                tmax = (lower - origin) * reciprocal

        else:

            # ray is not propagating along this axis so limits are infinite
            if origin < lower:

                tmin = -INFINITY
                tmax = -INFINITY

            elif origin > upper:

                tmin = INFINITY
                tmax = INFINITY

            else:

                tmin = -INFINITY
                tmax = INFINITY

        # calculate slab intersection overlap, store closest dimension and intersected place
        if tmin > front_intersection[0]:
            front_intersection[0] = tmin

        if tmax < back_intersection[0]:
            back_intersection[0] = tmax

    cpdef bint inside(self, Point point):

        # point is inside box if it is inside all slabs
        if (point.x < self.lower.x) or (point.x > self.upper.x):

            return False

        if (point.y < self.lower.y) or (point.y > self.upper.y):

            return False

        if (point.z < self.lower.z) or (point.z > self.upper.z):

            return False

        return True