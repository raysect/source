# cython: language_level=3

#Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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

from raysect.core.math.point cimport Point
from raysect.core.math.vector cimport Vector
from raysect.core.math.normal cimport Normal
from raysect.core.math.affinematrix cimport AffineMatrix
from raysect.core.scenegraph.world cimport World
from raysect.core.scenegraph.primitive cimport Primitive

cdef class Ray:
    
    cdef public Point origin
    cdef public Vector direction
    cdef public double min_distance
    cdef public double max_distance

    cpdef trace(self, World world)
    

cdef class Intersection:
    
    cdef public Ray ray
    cdef public Primitive primitive
    cdef public Point hit_point
    cdef public Normal surface_normal
    cdef public double ray_distance    
    cdef public AffineMatrix to_world
    cdef public AffineMatrix to_object


cdef class Material:

    cpdef evaluate_surface(self, Ray ray, Point point, Normal normal, AffineMatrix to_world, AffineMatrix to_object)
    
    cpdef evaluate_volume(self, Ray ray, Point entry, Point exit, AffineMatrix to_world, AffineMatrix to_object)