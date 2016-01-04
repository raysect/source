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

cimport cython
from raysect.core.math.point cimport new_point3d, new_point2d

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

# axis defines
DEF X_AXIS = 0
DEF Y_AXIS = 1
DEF Z_AXIS = 2


cdef class BoundingBox3D:
    """
    Axis-aligned bounding box.

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

    def __init__(self, Point3D lower=None, Point3D upper=None):

        # initialise to a null box if called without both initial points
        if lower is None or upper is None:
            self.lower = new_point3d(INFINITY, INFINITY, INFINITY)
            self.upper = new_point3d(-INFINITY, -INFINITY, -INFINITY)
        else:
            if lower.x > upper.x or lower.y > upper.y or lower.z > upper.z:
                raise ValueError("The lower point coordinates must be less than or equal to the upper point coordinates.")
            self.lower = lower
            self.upper = upper

    def __repr__(self):

        return "BoundingBox3D({}, {})".format(self.lower, self.upper)

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.lower, self.upper

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.lower, self.upper = state

    property lower:

        def __get__(self):
            return self.lower

        def __set__(self, Point3D value not None):
            self.lower = value

    property upper:

        def __get__(self):
            return self.upper

        def __set__(self, Point3D value not None):
            self.upper = value

    cpdef bint hit(self, Ray ray):

        cdef:
            double front_intersection, back_intersection

        return self.intersect(ray, &front_intersection, &back_intersection)

    cpdef tuple full_intersection(self, Ray ray):

        cdef:
            double front_intersection, back_intersection
            bint hit

        hit = self.intersect(ray, &front_intersection, &back_intersection)

        return hit, front_intersection, back_intersection

    cdef inline bint intersect(self, Ray ray, double *front_intersection, double *back_intersection):

        # set initial ray-slab intersection search range
        front_intersection[0] = -INFINITY
        back_intersection[0] = INFINITY

        # evaluate ray-slab intersection for x, y and z dimensions and update the intersection positions
        self._slab(ray.origin.x, ray.direction.x, self.lower.x, self.upper.x, front_intersection, back_intersection)
        self._slab(ray.origin.y, ray.direction.y, self.lower.y, self.upper.y, front_intersection, back_intersection)
        self._slab(ray.origin.z, ray.direction.z, self.lower.z, self.upper.z, front_intersection, back_intersection)

        # does ray intersect box?
        if front_intersection[0] > back_intersection[0]:
            return False

        # are both intersections behind ray origin?
        if (front_intersection[0] < 0.0) and (back_intersection[0] < 0.0):
            return False
        return True

    @cython.cdivision(True)
    cdef inline void _slab(self, double origin, double direction, double lower, double upper, double *front_intersection, double *back_intersection):

        cdef double reciprocal, tmin, tmax

        if direction != 0.0:

            # calculate intersections with slab planes
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

    cpdef bint contains(self, Point3D point):

        # point is inside box if it is inside all slabs
        if (point.x < self.lower.x) or (point.x > self.upper.x):
            return False
        if (point.y < self.lower.y) or (point.y > self.upper.y):
            return False
        if (point.z < self.lower.z) or (point.z > self.upper.z):
            return False
        return True

    cpdef object union(self, BoundingBox3D box):

        self.lower.x = min(self.lower.x, box.lower.x)
        self.lower.y = min(self.lower.y, box.lower.y)
        self.lower.z = min(self.lower.z, box.lower.z)

        self.upper.x = max(self.upper.x, box.upper.x)
        self.upper.y = max(self.upper.y, box.upper.y)
        self.upper.z = max(self.upper.z, box.upper.z)

    cpdef object extend(self, Point3D point, double padding=0.0):

        self.lower.x = min(self.lower.x, point.x - padding)
        self.lower.y = min(self.lower.y, point.y - padding)
        self.lower.z = min(self.lower.z, point.z - padding)

        self.upper.x = max(self.upper.x, point.x + padding)
        self.upper.y = max(self.upper.y, point.y + padding)
        self.upper.z = max(self.upper.z, point.z + padding)

    cpdef double surface_area(self):

        cdef double dx, dy, dz

        dx = self.upper.x - self.lower.x
        dy = self.upper.y - self.lower.y
        dz = self.upper.z - self.lower.z

        return 2 * (dx * dy + dx * dz + dy * dz)

    cpdef double volume(self):

        return (self.upper.x - self.lower.x) * (self.upper.y - self.lower.y) * (self.upper.z - self.lower.z)

    cpdef list vertices(self):

        return [
            new_point3d(self.lower.x, self.lower.y, self.lower.z),
            new_point3d(self.lower.x, self.lower.y, self.upper.z),
            new_point3d(self.lower.x, self.upper.y, self.lower.z),
            new_point3d(self.lower.x, self.upper.y, self.upper.z),
            new_point3d(self.upper.x, self.lower.y, self.lower.z),
            new_point3d(self.upper.x, self.lower.y, self.upper.z),
            new_point3d(self.upper.x, self.upper.y, self.lower.z),
            new_point3d(self.upper.x, self.upper.y, self.upper.z),
        ]

    cpdef double extent(self, int axis) except *:

        if axis == X_AXIS:
            return max(0.0, self.upper.x - self.lower.x)
        elif axis == Y_AXIS:
            return max(0.0, self.upper.y - self.lower.y)
        elif axis == Z_AXIS:
            return max(0.0, self.upper.z - self.lower.z)
        else:
            raise ValueError("Axis must be in the range [0, 2].")

    cpdef int largest_axis(self):

        cdef:
            int largest_axis
            double largest_extent, extent

        largest_axis = X_AXIS
        largest_extent = self.extent(X_AXIS)

        extent = self.extent(Y_AXIS)
        if extent > largest_extent:
            largest_axis = Y_AXIS
            largest_extent = extent

        extent = self.extent(Z_AXIS)
        if extent > largest_extent:
            largest_axis = Z_AXIS
            largest_extent = extent

        return largest_axis

    cpdef double largest_extent(self):

        return max(self.extent(X_AXIS), self.extent(Y_AXIS), self.extent(Z_AXIS))

    cpdef object pad(self, double padding):

        self.lower.x = self.lower.x - padding
        self.lower.y = self.lower.y - padding
        self.lower.z = self.lower.z - padding

        self.upper.x = self.upper.x + padding
        self.upper.y = self.upper.y + padding
        self.upper.z = self.upper.z + padding


cdef class BoundingBox2D:
    """
    Axis-aligned 2D bounding box.
    """

    def __init__(self, Point2D lower=None, Point2D upper=None):

        # initialise to a null box if called without both initial points
        if lower is None or upper is None:
            self.lower = new_point2d(INFINITY, INFINITY)
            self.upper = new_point2d(-INFINITY, -INFINITY)
        else:
            if lower.x > upper.x or lower.y > upper.y:
                raise ValueError("The lower point coordinates must be less than or equal to the upper point coordinates.")
            self.lower = lower
            self.upper = upper

    def __repr__(self):

        return "BoundingBox2D({}, {})".format(self.lower, self.upper)

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.lower, self.upper

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.lower, self.upper = state

    property lower:

        def __get__(self):
            return self.lower

        def __set__(self, Point2D value not None):
            self.lower = value

    property upper:

        def __get__(self):
            return self.upper

        def __set__(self, Point2D value not None):
            self.upper = value

    cpdef bint contains(self, Point2D point):

        # point is inside box if it is inside all slabs
        if (point.x < self.lower.x) or (point.x > self.upper.x):
            return False
        if (point.y < self.lower.y) or (point.y > self.upper.y):
            return False
        return True

    cpdef object union(self, BoundingBox2D box):

        self.lower.x = min(self.lower.x, box.lower.x)
        self.lower.y = min(self.lower.y, box.lower.y)

        self.upper.x = max(self.upper.x, box.upper.x)
        self.upper.y = max(self.upper.y, box.upper.y)

    cpdef object extend(self, Point2D point, double padding=0.0):

        self.lower.x = min(self.lower.x, point.x - padding)
        self.lower.y = min(self.lower.y, point.y - padding)

        self.upper.x = max(self.upper.x, point.x + padding)
        self.upper.y = max(self.upper.y, point.y + padding)

    cpdef double surface_area(self):

        return (self.upper.x - self.lower.x) * (self.upper.y - self.lower.y)

    cpdef list vertices(self):

        return [
            new_point2d(self.lower.x, self.lower.y),
            new_point2d(self.lower.x, self.upper.y),
            new_point2d(self.upper.x, self.lower.y),
            new_point2d(self.upper.x, self.upper.y),
        ]

    cpdef double extent(self, int axis) except *:

        if axis == X_AXIS:
            return max(0.0, self.upper.x - self.lower.x)
        elif axis == Y_AXIS:
            return max(0.0, self.upper.y - self.lower.y)
        else:
            raise ValueError("Axis must be in the range [0, 1].")

    cpdef int largest_axis(self):

        cdef:
            int largest_axis
            double largest_extent, extent

        largest_axis = X_AXIS
        largest_extent = self.extent(X_AXIS)

        extent = self.extent(Y_AXIS)
        if extent > largest_extent:
            largest_axis = Y_AXIS
            largest_extent = extent

        return largest_axis

    cpdef double largest_extent(self):

        return max(self.extent(X_AXIS), self.extent(Y_AXIS))

    cpdef object pad(self, double padding):

        self.lower.x = self.lower.x - padding
        self.lower.y = self.lower.y - padding

        self.upper.x = self.upper.x + padding
        self.upper.y = self.upper.y + padding
