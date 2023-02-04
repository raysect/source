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

from raysect.core cimport Primitive, Ray, Intersection, BoundingBox3D, AffineMatrix3D, Normal3D, Point3D
from raysect.core.math.spatial cimport KDTree3DCore
from numpy cimport float32_t, int32_t, uint8_t, ndarray


cdef class MeshIntersection(Intersection):

    cdef:
        public int32_t triangle
        public float u, v, w


cdef class MeshData(KDTree3DCore):

    cdef:
        ndarray _vertices
        ndarray _vertex_normals
        ndarray _face_normals
        ndarray _triangles
        float32_t[:, ::1] vertices_mv
        float32_t[:, ::1] vertex_normals_mv
        float32_t[:, ::1] face_normals_mv
        int32_t[:, ::1] triangles_mv
        public bint smoothing
        public bint closed
        int32_t _ix, _iy, _iz
        float _sx, _sy, _sz
        float _u, _v, _w, _t
        int32_t _i

    cpdef Point3D vertex(self, int index)

    cpdef ndarray triangle(self, int index)

    cpdef Normal3D vertex_normal(self, int index)

    cpdef Normal3D face_normal(self, int index)

    cdef object _filter_triangles(self)

    cdef object _flip_normals(self)

    cdef object _generate_face_normals(self)

    cdef BoundingBox3D _generate_bounding_box(self, int32_t i)

    cdef void _calc_rayspace_transform(self, Ray ray)

    cdef bint _hit_triangle(self, int32_t i, Ray ray, float[4] hit_data)

    cpdef Intersection calc_intersection(self, Ray ray)

    cdef Normal3D _intersection_normal(self)

    cpdef bint contains(self, Point3D p)

    cpdef BoundingBox3D bounding_box(self, AffineMatrix3D to_world)

    cdef uint8_t _read_uint8(self, object file)

    cdef bint _read_bool(self, object file)

    cdef double _read_float(self, object file)


cdef class Mesh(Primitive):

    cdef:
        readonly MeshData data
        bint _seek_next_intersection
        Ray _next_world_ray
        Ray _next_local_ray
        double _ray_distance

    cdef Intersection _process_intersection(self, Ray world_ray, Ray local_ray)


cdef inline MeshIntersection new_mesh_intersection(
        Ray ray, double ray_distance, Primitive primitive,
        Point3D hit_point, Point3D inside_point, Point3D outside_point, Normal3D normal, bint exiting,
        AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
        int32_t triangle, float u, float v, float w):

    cdef Intersection intersection
    intersection = MeshIntersection.__new__(MeshIntersection)
    intersection._construct(ray, ray_distance, primitive, hit_point, inside_point, outside_point, normal, exiting, world_to_primitive, primitive_to_world)
    intersection.triangle = triangle
    intersection.u = u
    intersection.v = v
    intersection.w = w
    return intersection
