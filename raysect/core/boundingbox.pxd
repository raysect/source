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

from raysect.core.boundingsphere cimport BoundingSphere3D
from raysect.core.math cimport Point3D, Point2D
from raysect.core.ray cimport Ray


cdef class BoundingBox3D:

    cdef Point3D lower
    cdef Point3D upper

    cdef Point3D get_centre(self)

    cpdef bint hit(self, Ray ray)

    cpdef tuple full_intersection(self, Ray ray)

    cdef bint intersect(self, Ray ray, double *front_intersection, double *back_intersection)

    cdef void _slab(self, double origin, double direction, double lower, double upper, double *front_intersection, double *back_intersection) nogil

    cpdef bint contains(self, Point3D point)

    cpdef object union(self, BoundingBox3D box)

    cpdef object extend(self, Point3D point, double padding=*)

    cpdef double surface_area(self)

    cpdef double volume(self)

    cpdef list vertices(self)

    cpdef double extent(self, int axis) except -1

    cpdef int largest_axis(self)

    cpdef double largest_extent(self)

    cpdef object pad(self, double padding)

    cpdef object pad_axis(self, int axis, double padding)

    cpdef BoundingSphere3D enclosing_sphere(self)


cdef inline BoundingBox3D new_boundingbox3d(Point3D lower, Point3D upper):
    """
    BoundingBox3D factory function.

    Creates a new BoundingBox3D object with less overhead than the equivalent
    Python call. This function is callable from cython only.
    """

    cdef BoundingBox3D v
    v = BoundingBox3D.__new__(BoundingBox3D)
    v.lower = lower
    v.upper = upper
    return v


cdef class BoundingBox2D:

    cdef Point2D lower
    cdef Point2D upper

    cpdef bint contains(self, Point2D point)

    cpdef object union(self, BoundingBox2D box)

    cpdef object extend(self, Point2D point, double padding=*)

    cpdef double surface_area(self)

    cpdef list vertices(self)

    cpdef double extent(self, int axis) except -1

    cpdef int largest_axis(self)

    cpdef double largest_extent(self)

    cpdef object pad(self, double padding)

    cpdef object pad_axis(self, int axis, double padding)


cdef inline BoundingBox2D new_boundingbox2d(Point2D lower, Point2D upper):
    """
    BoundingBox2D factory function.

    Creates a new BoundingBox2D object with less overhead than the equivalent
    Python call. This function is callable from cython only.
    """

    cdef BoundingBox2D b
    b = BoundingBox2D.__new__(BoundingBox2D)
    b.lower = lower
    b.upper = upper
    return b
