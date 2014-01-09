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

import numbers
cimport cython
from libc.math cimport sqrt
from raysect.core.math.vector cimport new_vector

cdef class Normal(_Vec3):
    """
    Represents a mathematical normal vector in 3D affine space.
    """
    
    def __init__(self, v = (1.0, 0.0, 0.0)):
        """
        Normal vector constructor.
        
        If no initial values are passed, Normal defaults to a unit vector
        aligned with the x-axis: [1.0, 0.0, 0.0]
        
        Any three (or more) item indexable object can be used to initialise the
        normal. The x, y and z coordinates will be assigned the values of 
        the items at indexes [0, 1, 2].
        
        e.g. Normal([4.0, 5.0, 6.0]) sets the x, y and z coordinates as 4.0,
        5.0 and 6.0 respectively.
        """
        
        try:
            
            self.d[0] = v[0]
            self.d[1] = v[1]
            self.d[2] = v[2]
            
        except:
            
            raise TypeError("Normal can only be initialised with an indexable object, containing numerical values, of length >= 3 items.")

    def __repr__(self):
        """Returns a string representation of the Normal object."""

        return "Normal([" + str(self.d[0]) + ", " + str(self.d[1]) + ", " + str(self.d[2]) + "])"

    def __neg__(self):
        """Returns a normal with the reverse orientation."""

        return new_normal(-self.d[0],
                          -self.d[1],
                          -self.d[2])


    def __add__(object x, object y):
        """Vector addition."""

        cdef _Vec3 vx, vy
        
        if isinstance(x, _Vec3) and isinstance(y, _Vec3):
            
            vx = <_Vec3>x
            vy = <_Vec3>y

            return new_normal(vx.d[0] + vy.d[0],
                              vx.d[1] + vy.d[1],
                              vx.d[2] + vy.d[2])
        
        else:

            raise TypeError("Unsupported operand type. Expects a Vector, Normal or Point.")

    def __sub__(object x, object y):
        """Vector subtraction."""
        
        cdef _Vec3 vx, vy
        
        if isinstance(x, _Vec3) and isinstance(y, _Vec3):
            
            vx = <_Vec3>x
            vy = <_Vec3>y
            
            return new_normal(vx.d[0] - vy.d[0],
                              vx.d[1] - vy.d[1],
                              vx.d[2] - vy.d[2])
        
        else:

            raise TypeError("Unsupported operand type. Expects a Vector, Normal or Point.")
            
    def __mul__(object x, object y):
        """Multiply normal vector by a scalar."""

        cdef double m
        cdef Normal v
        
        if isinstance(x, numbers.Real) and isinstance(y, Normal):
        
            m = <double>x
            v = <Normal>y
        
        elif isinstance(x, Normal) and isinstance(y, numbers.Real):
        
            m = <double>y
            v = <Normal>x
        
        else:
        
            raise TypeError("Unsupported operand type. Expects a real number.")

        return new_normal(m * v.d[0],
                          m * v.d[1],
                          m * v.d[2])


    def __truediv__(object x, object y):
        """Division of a vector by a scalar."""

        cdef double d
        cdef Normal v
        
        if isinstance(x, Normal) and isinstance(y, numbers.Real):
        
            d = <double>y

            # prevent divide my zero
            if d == 0.0:

                raise ZeroDivisionError("Cannot divide a vector by a zero scalar.")

            v = <Normal>x
            
            with cython.cdivision(True):

                d = 1.0 / d 

            return new_normal(d * v.d[0],
                              d * v.d[1],
                              d * v.d[2])
        
        else:
        
            raise TypeError("Unsupported operand type. Expects a real number.")

    cpdef Vector cross(self, _Vec3 v):
        """
        Calculates the cross product between this vector and the supplied
        vector: 
        
            C = A.cross(B) <=> C = A x B
            
        Returns a Vector.
        """
    
        return new_vector(self.d[1] * v.d[2] - v.d[1] * self.d[2],
                          self.d[2] * v.d[0] - v.d[2] * self.d[0],
                          self.d[0] * v.d[1] - v.d[0] * self.d[1])
 
    cpdef Normal normalise(self):
        """
        Returns a normalised copy of the normal vector.
        
        The returned normal is normalised to length 1.0 - a unit vector.
        """
   
        cdef double t
    
        # if current length is zero, problem is ill defined
        t = self.d[0] * self.d[0] + self.d[1] * self.d[1] + self.d[2] * self.d[2]
        if t == 0.0:
            
            raise ZeroDivisionError("A zero length vector can not be normalised as the direction of a zero length vector is undefined.")
        
        # normalise and rescale vector
        with cython.cdivision(True):
            
            t = 1.0 / sqrt(t)

        return new_normal(self.d[0] * t,
                          self.d[1] * t,
                          self.d[2] * t)

    #cpdef Normal transform(AffineMatrix t):
        ## warn users that this method is slow and that they should use transform_with_inverse if an inverse is already available
        
        #pass
    
    #cpdef Normal transform_with_inverse(AffineMatrix t):
        
        ## avoids the need to invert the matrix = faster!
        #pass

    cdef inline Normal neg(self):

        return new_normal(-self.d[0],
                          -self.d[1],
                          -self.d[2])
    
    cdef inline Normal add(self, _Vec3 v):

        return new_normal(self.d[0] + v.d[0],
                          self.d[1] + v.d[1],
                          self.d[2] + v.d[2])    
    
    cdef inline Normal sub(self, _Vec3 v):
    
        return new_normal(self.d[0] - v.d[0],
                          self.d[1] - v.d[1],
                          self.d[2] - v.d[2])        
    
    cdef inline Normal mul(self, double m):

        return new_normal(self.d[0] * m,
                          self.d[1] * m,
                          self.d[2] * m)
    
    cdef inline Normal div(self, double d):
            
        if d == 0.0:
            
            raise ZeroDivisionError("Cannot divide a vector by a zero scalar.")

        with cython.cdivision:
            
            d = 1.0 / d
            
        return new_normal(self.d[0] * d,
                          self.d[1] * d,
                          self.d[2] * d)