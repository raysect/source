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

from raysect.core.ray cimport Ray
from raysect.core.math cimport Point3D, Normal3D, AffineMatrix3D
from raysect.core.scenegraph.primitive cimport Primitive


cdef class Intersection:

    cdef public Ray ray
    cdef public double ray_distance
    cdef public bint exiting
    cdef public Primitive primitive
    cdef public Point3D hit_point
    cdef public Point3D inside_point
    cdef public Point3D outside_point
    cdef public Normal3D normal
    cdef public AffineMatrix3D world_to_primitive
    cdef public AffineMatrix3D primitive_to_world

    cdef void _construct(self, Ray ray, double ray_distance, Primitive primitive,
                                Point3D hit_point, Point3D inside_point, Point3D outside_point,
                                Normal3D normal, bint exiting, AffineMatrix3D world_to_primitive,
                                AffineMatrix3D primitive_to_world)



cdef inline Intersection new_intersection(Ray ray, double ray_distance, Primitive primitive,
                                          Point3D hit_point, Point3D inside_point, Point3D outside_point,
                                          Normal3D normal, bint exiting, AffineMatrix3D world_to_primitive,
                                          AffineMatrix3D primitive_to_world):

    cdef Intersection intersection
    intersection = Intersection.__new__(Intersection)
    intersection._construct(ray, ray_distance, primitive, hit_point, inside_point, outside_point, normal, exiting, world_to_primitive, primitive_to_world)
    return intersection
