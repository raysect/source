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
from raysect.core.math.vector cimport Vector3D
from raysect.core.math.affinematrix cimport AffineMatrix3D

cdef class Normal3D(_Vec3):

    cpdef Vector3D cross(self, _Vec3 v)

    cpdef Normal3D normalise(self)

    cpdef Normal3D transform(self, AffineMatrix3D m)

    cpdef Normal3D transform_with_inverse(self, AffineMatrix3D m)

    cdef Normal3D neg(self)

    cdef Normal3D add(self, _Vec3 v)

    cdef Normal3D sub(self, _Vec3 v)

    cdef Normal3D mul(self, double m)

    cdef Normal3D div(self, double m)

    cpdef Normal3D copy(self)

    cpdef Vector3D as_vector(self)

    cpdef Vector3D orthogonal(self)


cdef inline Normal3D new_normal3d(double x, double y, double z):
    """
    Normal3D factory function.

    Creates a new Normal3D object with less overhead than the equivalent Python
    call. This function is callable from cython only.
    """

    cdef Normal3D v
    v = Normal3D.__new__(Normal3D)
    v.x = x
    v.y = y
    v.z = z
    return v
