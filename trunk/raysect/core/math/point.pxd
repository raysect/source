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

from raysect.core.math._vec3 cimport _Vec3
from raysect.core.math.vector cimport Vector
from raysect.core.math.affinematrix cimport AffineMatrix

cdef class Point:
    
    cdef double d[3]
    
    cdef inline double get_x(self)
    
    cdef inline void set_x(self, double v)

    cdef inline double get_y(self)
    
    cdef inline void set_y(self, double v)
    
    cdef inline double get_z(self)
    
    cdef inline void set_z(self, double v)
    
    cpdef Vector vector_to(self, Point p)
    
    cpdef double distance_to(self, Point p)

    cpdef Point transform(self, AffineMatrix m)

    cdef inline Point add(self, _Vec3 v)  
    
    cdef inline Point sub(self, _Vec3 v)

cdef inline Point new_point(double x, double y, double z):
    
    cdef Point v
    v = Point.__new__(Point)
    v.d[0] = x
    v.d[1] = y
    v.d[2] = z
    return v