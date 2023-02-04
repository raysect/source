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

from raysect.core.math._mat4 cimport _Mat4


cdef class AffineMatrix3D(_Mat4):

    cpdef AffineMatrix3D inverse(self)

    cdef AffineMatrix3D mul(self, AffineMatrix3D m)


cdef inline AffineMatrix3D new_affinematrix3d(double m00, double m01, double m02, double m03,
                                              double m10, double m11, double m12, double m13,
                                              double m20, double m21, double m22, double m23,
                                              double m30, double m31, double m32, double m33):
    """
    AffineMatrix3D factory function.

    Creates a new AffineMatrix3D object with less overhead than the equivalent
    Python call. This function is callable from cython only.
    """

    cdef AffineMatrix3D v
    v = AffineMatrix3D.__new__(AffineMatrix3D)
    v.m[0][0] = m00
    v.m[0][1] = m01
    v.m[0][2] = m02
    v.m[0][3] = m03
    v.m[1][0] = m10
    v.m[1][1] = m11
    v.m[1][2] = m12
    v.m[1][3] = m13
    v.m[2][0] = m20
    v.m[2][1] = m21
    v.m[2][2] = m22
    v.m[2][3] = m23
    v.m[3][0] = m30
    v.m[3][1] = m31
    v.m[3][2] = m32
    v.m[3][3] = m33
    return v
