# cython: language_level=3

# Copyright (c) 2014-2017, Dr Alex Meakins, Raysect Project
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

# TODO: split instance into its own method (e.g. mymesh.instance(), this change is planned for Interpolate2DMesh. Both classes should be consistent.

from raysect.core cimport Primitive, Ray, Intersection, BoundingBox3D, AffineMatrix3D, Normal3D, Point3D
from raysect.core.math.spatial cimport KDTree3DCore
from numpy cimport float32_t, int32_t, uint8_t


cdef class MeshData(KDTree3DCore):

    cdef:
        float32_t[:, ::1] vertices
        float32_t[:, ::1] vertex_normals
        float32_t[:, ::1] face_normals
        int32_t[:, ::1] triangles
        public bint smoothing
        public bint closed
        int32_t _ix, _iy, _iz
        float _sx, _sy, _sz
        float _u, _v, _w, _t
        int32_t _i

    cdef object _filter_triangles(self)

    cdef object _generate_face_normals(self)

    cdef BoundingBox3D _generate_bounding_box(self, int32_t i)

    cdef void _calc_rayspace_transform(self, Ray ray)

    cdef bint _hit_triangle(self, int32_t i, Ray ray, float[4] hit_data)

    cpdef Intersection calc_intersection(self, Ray ray)

    cdef Normal3D _intersection_normal(self)

    cpdef bint contains(self, Point3D p)

    cpdef BoundingBox3D bounding_box(self, AffineMatrix3D to_world)

    cdef inline uint8_t _read_uint8(self, object file)

    cdef inline bint _read_bool(self, object file)

    cdef inline double _read_float(self, object file)


cdef class Mesh(Primitive):

    cdef:
        MeshData _data
        bint _seek_next_intersection
        Ray _next_world_ray
        Ray _next_local_ray
        double _ray_distance

    cdef Intersection _process_intersection(self, Ray world_ray, Ray local_ray)
