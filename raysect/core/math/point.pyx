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

cimport cython
from libc.math cimport sqrt
from raysect.core.math.vector cimport new_vector
from raysect.core.math._vec3 cimport _Vec3

cdef class Point:
    """
    Represents a point in 3D affine space.
    """

    def __init__(self, v = (0.0, 0.0, 0.0)):
        """
        Point constructor.
        
        If no initial values are passed, Point defaults to the origin:
        [0.0, 0.0, 0.0]
        
        Any three (or more) item indexable object can be used to initialise the
        point. The x, y and z coordinates will be assigned the values of 
        the items at indexes [0, 1, 2].
        
        e.g. Point([4.0, 5.0, 6.0]) sets the x, y and z coordinates as 4.0,
        5.0 and 6.0 respectively.
        """

        cdef Point p
        cdef _Vec3 t

        if isinstance(v, _Vec3):
            
            t = <_Vec3>v
            
            self.x = t.x
            self.y = t.y
            self.z = t.z 
        
        elif isinstance(v, Point):
            
            p = <Point>v

            self.x = p.x
            self.y = p.y
            self.z = p.z
            
        else:
        
            try:
                
                self.x = v[0]
                self.y = v[1]
                self.z = v[2]
                
            except:
                
                raise TypeError("Point can only be initialised with an indexable object, containing numerical values, of length >= 3 items.")
    
    def __repr__(self):
        """Returns a string representation of the Point object."""

        return "Point([" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + "])"
    
    def __getitem__(self, int i):
        """Returns the point coordinates by index ([0,1,2] -> [x,y,z])."""

        if i < 0 or i > 2:
            raise IndexError("Index out of range [0, 2].")
        elif i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            return self.z

    def __add__(object x, object y):
        """Addition operator."""
       
        cdef Point p
        cdef _Vec3 v
        
        if isinstance(x, Point) and isinstance(y, _Vec3):

            p = <Point>x
            v = <_Vec3>y
       
        else:

            return NotImplemented

        return new_point(p.x + v.x,
                         p.y + v.y,
                         p.z + v.z)

    def __sub__(object x, object y):
        """Subtraction operator."""
        
        cdef Point p
        cdef _Vec3 v
        
        if isinstance(x, Point) and isinstance(y, _Vec3):
            
            p = <Point>x
            v = <_Vec3>y

            return new_point(p.x - v.x,
                             p.y - v.y,
                             p.z - v.z)
        
        
        else:

            return NotImplemented
        
    def __mul__(object x, object y):
        """Multiply operator."""
        
        cdef AffineMatrix m
        cdef Point v
        cdef double w
        
        if isinstance(x, AffineMatrix) and isinstance(y, Point):
        
            m = <AffineMatrix>x
            v = <Point>y
        
            # 4th element of homogeneous coordinate 
            w = m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3]
            if w == 0.0:
                
                raise ZeroDivisionError("Bad matrix transform, 4th element of homogeneous coordinate is zero.")
            
            # pre divide for speed (dividing is much slower than multiplying)
            with cython.cdivision(True):
                
                w = 1.0 / w

            return new_point((m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3]) * w,
                             (m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3]) * w,
                             (m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3]) * w)        
            
        return NotImplemented

    cpdef Vector vector_to(self, Point p):
        """
        Returns a vector from this point to the passed point.
        """

        return new_vector(p.x - self.x,
                          p.y - self.y,
                          p.z - self.z)
    
    cpdef double distance_to(self, Point p):
        """
        Returns the distance between this point and the passed point.
        """
    
        cdef double x, y, z
        x = p.x - self.x
        y = p.y - self.y
        z = p.z - self.z
        return sqrt(x*x + y*y + z*z)

    cpdef Point transform(self, AffineMatrix m):
        """
        Transforms the point with the supplied Affine Matrix.
        
        The point is transformed by premultiplying the point by the affine
        matrix.
        
        For cython code this method is substantially faster than using the
        multiplication operator of the affine matrix.
        
        This method expects a valid affine transform. For speed reasons, minimal
        checks are performed on the matrix.
        """

        cdef double w
        
        # 4th element of homogeneous coordinate 
        w = m.m[3][0] * self.x + m.m[3][1] * self.y + m.m[3][2] * self.z + m.m[3][3]
        if w == 0.0:
            
            raise ZeroDivisionError("Bad matrix transform, 4th element of homogeneous coordinate is zero.")
        
        # pre divide for speed (dividing is much slower than multiplying)
        with cython.cdivision(True):
            
            w = 1.0 / w

        return new_point((m.m[0][0] * self.x + m.m[0][1] * self.y + m.m[0][2] * self.z + m.m[0][3]) * w,
                         (m.m[1][0] * self.x + m.m[1][1] * self.y + m.m[1][2] * self.z + m.m[1][3]) * w,
                         (m.m[2][0] * self.x + m.m[2][1] * self.y + m.m[2][2] * self.z + m.m[2][3]) * w)

    cdef inline Point add(self, _Vec3 v):

        return new_point(self.x + v.x,
                          self.y + v.y,
                          self.z + v.z)    
    
    cdef inline Point sub(self, _Vec3 v):
    
        return new_point(self.x - v.x,
                          self.y - v.y,
                          self.z - v.z)  