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

from libc.math cimport M_PI, M_1_PI, sqrt, sin, cos
from raysect.core.math cimport Point2D, new_point2d, Point3D, new_point3d, Vector3D, new_vector3d
from raysect.core.math.random cimport uniform
from raysect.core.math.cython cimport barycentric_coords, barycentric_interpolation

DEF R_2_PI = 0.15915494309189535  # 1 / (2 * pi)
DEF R_4_PI = 0.07957747154594767  # 1 / (4 * pi)


cdef class SolidAngleSampler:

    cpdef double pdf(self, Vector3D sample)

    cdef Vector3D sample(self)

    cdef tuple sample_with_pdf(self)

    cdef list samples(self, int samples)

    cdef list samples_with_pdfs(self, int samples)


cdef class SphereSampler(SolidAngleSampler):
    pass


cdef class HemisphereUniformSampler(SolidAngleSampler):
    pass


cdef class HemisphereCosineSampler(SolidAngleSampler):
    pass


cdef class ConeUniformSampler(SolidAngleSampler):

    cdef:
        readonly double angle
        double _angle_radians, _angle_cosine, _solid_angle, _solid_angle_inv


# cdef class ConeCosineSampler(SolidAngleSampler):
#
#     cdef:
#         readonly double angle
#         double _angle_radians, _angle_cosine, _solid_angle, _solid_angle_inv