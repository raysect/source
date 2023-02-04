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
from raysect.core.math.affinematrix cimport AffineMatrix3D

cdef class Vector3D(_Vec3):

    cpdef Vector3D cross(self, _Vec3 v)

    cpdef Vector3D normalise(self)

    cpdef Vector3D transform(self, AffineMatrix3D m)

    cdef Vector3D neg(self)

    cdef Vector3D add(self, _Vec3 v)

    cdef Vector3D sub(self, _Vec3 v)

    cdef Vector3D mul(self, double m)

    cdef Vector3D div(self, double m)

    cpdef Vector3D copy(self)

    cpdef Vector3D orthogonal(self)

    cpdef Vector3D lerp(self, Vector3D b, double t)

    cpdef Vector3D slerp(self, Vector3D b, double t)


cdef inline Vector3D new_vector3d(double x, double y, double z):
    """
    Vector3D factory function.

    Creates a new Vector3D object with less overhead than the equivalent Python
    call. This function is callable from cython only.
    """

    cdef Vector3D v
    v = Vector3D.__new__(Vector3D)
    v.x = x
    v.y = y
    v.z = z
    return v


cdef class Vector2D:

    cdef public double x, y

    cpdef double dot(self, Vector2D v)

    cdef double get_length(self) nogil

    cdef object set_length(self, double v)

    cdef double get_index(self, int index) nogil

    cdef void set_index(self, int index, double value) nogil

    cpdef double cross(self, Vector2D v)

    cpdef Vector2D normalise(self)

    # cpdef Vector2D transform(self, AffineMatrix2D m):

    cdef Vector2D neg(self)

    cdef Vector2D add(self, Vector2D v)

    cdef Vector2D sub(self, Vector2D v)

    cdef Vector2D mul(self, double m)

    cdef Vector3D div(self, double d)

    cpdef Vector2D copy(self)

    cpdef Vector2D orthogonal(self)


cdef inline Vector2D new_vector2d(double x, double y):
    """
    Vector2D factory function.

    Creates a new Vector2D object with less overhead than the equivalent Python
    call. This function is callable from cython only.
    """

    cdef Vector2D v
    v = Vector2D.__new__(Vector2D)
    v.x = x
    v.y = y
    return v
