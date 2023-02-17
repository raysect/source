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

import numpy as np
cimport numpy as np
from numpy cimport float32_t, int32_t

from raysect.core.math.sampler cimport HemisphereCosineSampler
from raysect.optical cimport Point3D, Vector3D, AffineMatrix3D
from raysect.optical.observer.base cimport Observer0D
from raysect.primitive.mesh.mesh cimport Mesh


cdef class MeshPixel(Observer0D):

    cdef:
        double _surface_offset, _solid_angle, _collection_area
        readonly Mesh mesh
        float32_t[:, ::1] _vertices_mv
        float32_t[:, ::1] _face_normals_mv
        int32_t[:, ::1] _triangles_mv
        np.ndarray _cdf
        double [::1] _cdf_mv
        HemisphereCosineSampler _vector_sampler

    cdef object _calculate_areas(self)
    cdef double _triangle_area(self, Point3D v1, Point3D v2, Point3D v3)
    cdef int32_t _pick_triangle(self)
    cdef AffineMatrix3D _surface_to_local(self, Vector3D normal)
