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

from numpy cimport ndarray, int32_t, uint8_t
from raysect.core cimport BoundingBox3D, Point3D, AffineMatrix3D
from raysect.core.math.spatial.kdtree3d cimport KDTree3DCore


cdef class TetraMeshData(KDTree3DCore):

    cdef:
        ndarray _vertices
        ndarray _tetrahedra
        double[:, ::1] vertices_mv
        int32_t[:, ::1] tetrahedra_mv
        int32_t tetrahedra_id
        int32_t i1, i2, i3, i4
        double alpha, beta, gamma, delta
        bint _cache_available
        double _cached_x
        double _cached_y
        double _cached_z
        bint _cached_result

    cpdef Point3D vertex(self, int index)

    cpdef ndarray tetrahedron(self, int index)

    cpdef Point3D barycenter(self, int index)

    cpdef double volume(self, int index)

    cpdef double volume_total(self)

    cdef double _volume(self, int index)

    cdef object _filter_tetrahedra(self)

    cdef BoundingBox3D _generate_bounding_box(self, int32_t tetrahedra)

    cpdef BoundingBox3D bounding_box(self, AffineMatrix3D to_world)

    cdef uint8_t _read_uint8(self, object file)

    cdef bint _read_bool(self, object file)