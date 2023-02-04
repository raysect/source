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

from raysect.core.math.vector cimport Vector3D
from raysect.core.math.affinematrix cimport AffineMatrix3D


cdef class Quaternion:

    cdef public double x, y, z, s

    cpdef Quaternion copy(self)

    cpdef Quaternion conjugate(self)

    cpdef Quaternion inverse(self)

    cpdef Quaternion normalise(self)

    cpdef bint is_unit(self, double tolerance=*)

    cpdef Quaternion transform(self, AffineMatrix3D m)

    cpdef AffineMatrix3D as_matrix(self)

    cpdef Quaternion quaternion_to(self, Quaternion q)

    cdef Quaternion neg(self)

    cdef Quaternion add(self, Quaternion q)

    cdef Quaternion sub(self, Quaternion q)

    cdef Quaternion mul_quaternion(self, Quaternion q)

    cdef Quaternion mul_scalar(self, double d)

    cdef Quaternion div_quaternion(self, Quaternion q)

    cdef Quaternion div_scalar(self, double d)

    cdef Vector3D get_axis(self)

    cdef double get_angle(self)

    cdef double get_length(self) nogil

    cdef object set_length(self, double v)


cdef inline Quaternion new_quaternion(double x, double y, double z, double s):
    """
    Quaternion factory function.

    Creates a new Quaternion object with less overhead than the equivalent Python
    call. This function is callable from cython only.
    """

    cdef Quaternion q
    q = Quaternion.__new__(Quaternion)
    q.x = x
    q.y = y
    q.z = z
    q.s = s
    return q


cdef Quaternion new_quaternion_from_matrix(AffineMatrix3D matrix)


cdef Quaternion new_quaternion_from_axis_angle(Vector3D axis, double angle)