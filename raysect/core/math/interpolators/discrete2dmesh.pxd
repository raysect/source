# cython: language_level=3

# Copyright (c) 2014-2016, Dr Alex Meakins, Raysect Project
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

cimport numpy as np
from raysect.core.boundingbox cimport BoundingBox2D
from raysect.core.math.function.function2d cimport Function2D
from raysect.core.math.spatial.kdtree2d cimport KDTree2DCore


cdef class _MeshKDTree(KDTree2DCore):

    cdef:
        double[:, ::1] _vertices
        np.int32_t[:, ::1] _triangles
        np.int32_t triangle_id

    cdef inline BoundingBox2D _generate_bounding_box(self, np.int32_t triangle)

    cdef inline void _calc_barycentric_coords(self, np.int32_t i1, np.int32_t i2, np.int32_t i3, double px, double py, double *alpha, double *beta, double *gamma) nogil

    cdef inline bint _hit_triangle(self, double alpha, double beta, double gamma) nogil


cdef class Discrete2DMesh(Function2D):

    cdef:
        double[::1] _triangle_data
        _MeshKDTree _kdtree
        bint _limit
        double _default_value

    cdef double evaluate(self, double x, double y) except? -1e999
