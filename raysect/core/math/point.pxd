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

from raysect.core.math._vec3 cimport _Vec3
from raysect.core.math.vector cimport Vector2D, Vector3D
from raysect.core.math.affinematrix cimport AffineMatrix3D

cdef class Point3D:

    cdef public double x, y, z

    cpdef Vector3D vector_to(self, Point3D p)

    cpdef double distance_to(self, Point3D p)

    cpdef Point3D transform(self, AffineMatrix3D m)

    cdef Point3D add(self, _Vec3 v)

    cdef Point3D sub(self, _Vec3 v)

    cpdef Point3D copy(self)

    cdef double get_index(self, int index) nogil

    cdef void set_index(self, int index, double value) nogil


cdef inline Point3D new_point3d(double x, double y, double z):
    """
    Point3D factory function.

    Creates a new Point3D object with less overhead than the equivalent Python
    call. This function is callable from cython only.
    """

    cdef Point3D v
    v = Point3D.__new__(Point3D)
    v.x = x
    v.y = y
    v.z = z
    return v


cdef class Point2D:

    cdef public double x, y

    cpdef Vector2D vector_to(self, Point2D p)

    cpdef double distance_to(self, Point2D p)

    # cpdef Point3D transform(self, AffineMatrix3D m)

    cdef Point2D add(self, Vector2D v)

    cdef Point2D sub(self, Vector2D v)

    cpdef Point2D copy(self)

    cdef double get_index(self, int index) nogil

    cdef void set_index(self, int index, double value) nogil


cdef inline Point2D new_point2d(double x, double y):
    """
    Point2D factory function.

    Creates a new Point2D object with less overhead than the equivalent Python
    call. This function is callable from cython only.
    """

    cdef Point2D a
    a = Point2D.__new__(Point2D)
    a.x = x
    a.y = y
    return a
