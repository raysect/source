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
from raysect.core.math.vector cimport Vector
from libc.math cimport fabs, sin, cos, M_PI as pi

cdef class AffineMatrix(_Mat4):
    """Represents a 4x4 affine matrix."""
    
    def __repr__(self):
        """String representation."""

        cdef int i, j
    
        s = "AffineMatrix(["
        for i in range(0, 4):
            
            s += "["
            
            for j in range(0, 4):
                
                s += str(self.m[i][j])
                if j < 3:
                    s += ", "
                
            s += "]"
            
            if i < 3:
                s += ", "                    
        
        return s + "])"
   
    def __mul__(object x, object y):
        """Multiplication operator."""
        
        cdef AffineMatrix mx, my
        
        if isinstance(x, AffineMatrix) and isinstance(y, AffineMatrix):

            mx = <AffineMatrix>x
            my = <AffineMatrix>y
        
            return new_affinematrix(mx.m[0][0] * my.m[0][0] + mx.m[0][1] * my.m[1][0] + mx.m[0][2] * my.m[2][0] + mx.m[0][3] * my.m[3][0],
                                    mx.m[0][0] * my.m[0][1] + mx.m[0][1] * my.m[1][1] + mx.m[0][2] * my.m[2][1] + mx.m[0][3] * my.m[3][1],
                                    mx.m[0][0] * my.m[0][2] + mx.m[0][1] * my.m[1][2] + mx.m[0][2] * my.m[2][2] + mx.m[0][3] * my.m[3][2],
                                    mx.m[0][0] * my.m[0][3] + mx.m[0][1] * my.m[1][3] + mx.m[0][2] * my.m[2][3] + mx.m[0][3] * my.m[3][3],
                                    mx.m[1][0] * my.m[0][0] + mx.m[1][1] * my.m[1][0] + mx.m[1][2] * my.m[2][0] + mx.m[1][3] * my.m[3][0],
                                    mx.m[1][0] * my.m[0][1] + mx.m[1][1] * my.m[1][1] + mx.m[1][2] * my.m[2][1] + mx.m[1][3] * my.m[3][1],
                                    mx.m[1][0] * my.m[0][2] + mx.m[1][1] * my.m[1][2] + mx.m[1][2] * my.m[2][2] + mx.m[1][3] * my.m[3][2],
                                    mx.m[1][0] * my.m[0][3] + mx.m[1][1] * my.m[1][3] + mx.m[1][2] * my.m[2][3] + mx.m[1][3] * my.m[3][3],
                                    mx.m[2][0] * my.m[0][0] + mx.m[2][1] * my.m[1][0] + mx.m[2][2] * my.m[2][0] + mx.m[2][3] * my.m[3][0],
                                    mx.m[2][0] * my.m[0][1] + mx.m[2][1] * my.m[1][1] + mx.m[2][2] * my.m[2][1] + mx.m[2][3] * my.m[3][1],
                                    mx.m[2][0] * my.m[0][2] + mx.m[2][1] * my.m[1][2] + mx.m[2][2] * my.m[2][2] + mx.m[2][3] * my.m[3][2],
                                    mx.m[2][0] * my.m[0][3] + mx.m[2][1] * my.m[1][3] + mx.m[2][2] * my.m[2][3] + mx.m[2][3] * my.m[3][3],
                                    mx.m[3][0] * my.m[0][0] + mx.m[3][1] * my.m[1][0] + mx.m[3][2] * my.m[2][0] + mx.m[3][3] * my.m[3][0],
                                    mx.m[3][0] * my.m[0][1] + mx.m[3][1] * my.m[1][1] + mx.m[3][2] * my.m[2][1] + mx.m[3][3] * my.m[3][1],
                                    mx.m[3][0] * my.m[0][2] + mx.m[3][1] * my.m[1][2] + mx.m[3][2] * my.m[2][2] + mx.m[3][3] * my.m[3][2],
                                    mx.m[3][0] * my.m[0][3] + mx.m[3][1] * my.m[1][3] + mx.m[3][2] * my.m[2][3] + mx.m[3][3] * my.m[3][3])

        return NotImplemented

    cpdef AffineMatrix inverse(self):
        """
        Calculates the inverse of the affine matrix.
        
        Returns an AffineMatrix containing the inverse.
        
        Raises a ValueError if the matrix is singular ans the inverse can not be
        calculated. All valid affine transforms should be invertable.
        """
       
        cdef double t[22], det, idet

        # calculate 4x4 determinant
        t[0] = self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]
        t[1] = self.m[0][0] * self.m[1][2] - self.m[0][2] * self.m[1][0]
        t[2] = self.m[0][0] * self.m[1][3] - self.m[0][3] * self.m[1][0]
        t[3] = self.m[0][1] * self.m[1][2] - self.m[0][2] * self.m[1][1]
        t[4] = self.m[0][1] * self.m[1][3] - self.m[0][3] * self.m[1][1]
        t[5] = self.m[0][2] * self.m[1][3] - self.m[0][3] * self.m[1][2]

        t[18] = self.m[2][0] * t[3] - self.m[2][1] * t[1] + self.m[2][2] * t[0]
        t[19] = self.m[2][0] * t[4] - self.m[2][1] * t[2] + self.m[2][3] * t[0]
        t[20] = self.m[2][0] * t[5] - self.m[2][2] * t[2] + self.m[2][3] * t[1]
        t[21] = self.m[2][1] * t[5] - self.m[2][2] * t[4] + self.m[2][3] * t[3]

        det = t[20] * self.m[3][1] + t[18] * self.m[3][3] - t[21] * self.m[3][0] - t[19] * self.m[3][2]

        # check matrix is invertible, small value must be greater than machine precision
        if fabs(det) < 1e-14:
        
            raise ValueError("Matrix is singular and not invertible.")
        
        with cython.cdivision(True):
            
            idet = 1.0 / det

        # apply Cramer's rule to invert matrix
        t[6] = self.m[0][0] * self.m[3][1] - self.m[0][1] * self.m[3][0]
        t[7] = self.m[0][0] * self.m[3][2] - self.m[0][2] * self.m[3][0]
        t[8] = self.m[0][0] * self.m[3][3] - self.m[0][3] * self.m[3][0]
        t[9] = self.m[0][1] * self.m[3][2] - self.m[0][2] * self.m[3][1]
        t[10] = self.m[0][1] * self.m[3][3] - self.m[0][3] * self.m[3][1]
        t[11] = self.m[0][2] * self.m[3][3] - self.m[0][3] * self.m[3][2]

        t[12] = self.m[1][0] * self.m[3][1] - self.m[1][1] * self.m[3][0]
        t[13] = self.m[1][0] * self.m[3][2] - self.m[1][2] * self.m[3][0]
        t[14] = self.m[1][0] * self.m[3][3] - self.m[1][3] * self.m[3][0]
        t[15] = self.m[1][1] * self.m[3][2] - self.m[1][2] * self.m[3][1]
        t[16] = self.m[1][1] * self.m[3][3] - self.m[1][3] * self.m[3][1]
        t[17] = self.m[1][2] * self.m[3][3] - self.m[1][3] * self.m[3][2]
        
        return new_affinematrix((self.m[2][2] * t[16] - self.m[2][1] * t[17] - self.m[2][3] * t[15]) * idet,
                                (self.m[2][1] * t[11] - self.m[2][2] * t[10] + self.m[2][3] * t[ 9]) * idet,
                                (self.m[3][1] * t[ 5] - self.m[3][2] * t[ 4] + self.m[3][3] * t[ 3]) * idet,
                                -t[21] * idet,
                                (self.m[2][0] * t[17] - self.m[2][2] * t[14] + self.m[2][3] * t[13]) * idet,
                                (self.m[2][2] * t[ 8] - self.m[2][0] * t[11] - self.m[2][3] * t[ 7]) * idet,
                                (self.m[3][2] * t[ 2] - self.m[3][0] * t[ 5] - self.m[3][3] * t[ 1]) * idet,
                                t[20] * idet,
                                (self.m[2][1] * t[14] - self.m[2][0] * t[16] - self.m[2][3] * t[12]) * idet,
                                (self.m[2][0] * t[10] - self.m[2][1] * t[ 8] + self.m[2][3] * t[ 6]) * idet,
                                (self.m[3][0] * t[ 4] - self.m[3][1] * t[ 2] + self.m[3][3] * t[ 0]) * idet,
                                -t[19] * idet,
                                (self.m[2][0] * t[15] - self.m[2][1] * t[13] + self.m[2][2] * t[12]) * idet,
                                (self.m[2][1] * t[ 7] - self.m[2][0] * t[ 9] - self.m[2][2] * t[ 6]) * idet,
                                (self.m[3][1] * t[ 1] - self.m[3][0] * t[ 3] - self.m[3][2] * t[ 0]) * idet,
                                t[18] * idet)

    cdef inline AffineMatrix mul(self, AffineMatrix m):

        return new_affinematrix(self.m[0][0] * m.m[0][0] + self.m[0][1] * m.m[1][0] + self.m[0][2] * m.m[2][0] + self.m[0][3] * m.m[3][0],
                                self.m[0][0] * m.m[0][1] + self.m[0][1] * m.m[1][1] + self.m[0][2] * m.m[2][1] + self.m[0][3] * m.m[3][1],
                                self.m[0][0] * m.m[0][2] + self.m[0][1] * m.m[1][2] + self.m[0][2] * m.m[2][2] + self.m[0][3] * m.m[3][2],
                                self.m[0][0] * m.m[0][3] + self.m[0][1] * m.m[1][3] + self.m[0][2] * m.m[2][3] + self.m[0][3] * m.m[3][3],
                                self.m[1][0] * m.m[0][0] + self.m[1][1] * m.m[1][0] + self.m[1][2] * m.m[2][0] + self.m[1][3] * m.m[3][0],
                                self.m[1][0] * m.m[0][1] + self.m[1][1] * m.m[1][1] + self.m[1][2] * m.m[2][1] + self.m[1][3] * m.m[3][1],
                                self.m[1][0] * m.m[0][2] + self.m[1][1] * m.m[1][2] + self.m[1][2] * m.m[2][2] + self.m[1][3] * m.m[3][2],
                                self.m[1][0] * m.m[0][3] + self.m[1][1] * m.m[1][3] + self.m[1][2] * m.m[2][3] + self.m[1][3] * m.m[3][3],
                                self.m[2][0] * m.m[0][0] + self.m[2][1] * m.m[1][0] + self.m[2][2] * m.m[2][0] + self.m[2][3] * m.m[3][0],
                                self.m[2][0] * m.m[0][1] + self.m[2][1] * m.m[1][1] + self.m[2][2] * m.m[2][1] + self.m[2][3] * m.m[3][1],
                                self.m[2][0] * m.m[0][2] + self.m[2][1] * m.m[1][2] + self.m[2][2] * m.m[2][2] + self.m[2][3] * m.m[3][2],
                                self.m[2][0] * m.m[0][3] + self.m[2][1] * m.m[1][3] + self.m[2][2] * m.m[2][3] + self.m[2][3] * m.m[3][3],
                                self.m[3][0] * m.m[0][0] + self.m[3][1] * m.m[1][0] + self.m[3][2] * m.m[2][0] + self.m[3][3] * m.m[3][0],
                                self.m[3][0] * m.m[0][1] + self.m[3][1] * m.m[1][1] + self.m[3][2] * m.m[2][1] + self.m[3][3] * m.m[3][1],
                                self.m[3][0] * m.m[0][2] + self.m[3][1] * m.m[1][2] + self.m[3][2] * m.m[2][2] + self.m[3][3] * m.m[3][2],
                                self.m[3][0] * m.m[0][3] + self.m[3][1] * m.m[1][3] + self.m[3][2] * m.m[2][3] + self.m[3][3] * m.m[3][3])
    
    
cpdef AffineMatrix translate(double x, double y, double z):
    """
    Returns an affine matrix representing a translation of the coordinate space.
    """ 
    
    return new_affinematrix(1, 0, 0, x,
                            0, 1, 0, y,
                            0, 0, 1, z,
                            0, 0, 0, 1)


cpdef AffineMatrix rotate_x(double angle):
    """
    Returns an affine matrix representing the rotation of the coordinate space
    about the X axis by the supplied angle.
    
    The angle is specified in degrees.
    """    
    
    cdef double r
    
    with cython.cdivision(True):
        
        r = pi * angle / 180.0
    
    return new_affinematrix(1, 0, 0, 0,
                            0, cos(r), -sin(r), 0,
                            0, sin(r), cos(r), 0,
                            0, 0, 0, 1)


cpdef AffineMatrix rotate_y(double angle):
    """
    Returns an affine matrix representing the rotation of the coordinate space
    about the Y axis by the supplied angle.
    
    The angle is specified in degrees.
    """
    
    cdef double r
    
    with cython.cdivision(True):
        
        r = pi * angle / 180.0
    
    return new_affinematrix(cos(r), 0, sin(r), 0,
                            0, 1, 0, 0,
                            -sin(r), 0, cos(r), 0,
                            0, 0, 0, 1)


cpdef AffineMatrix rotate_z(double angle):
    """
    Returns an affine matrix representing the rotation of the coordinate space
    about the Z axis by the supplied angle.
    
    The angle is specified in degrees.
    """    
    
    cdef double r
    
    with cython.cdivision(True):
        
        r = pi * angle / 180.0
    
    return new_affinematrix(cos(r), -sin(r), 0, 0,
                            sin(r), cos(r), 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1)


cpdef AffineMatrix rotate_vector(double angle, Vector v):
    """
    Returns an affine matrix representing the rotation of the coordinate space
    about the supplied vector by the specified angle.
    
    The angle is specified in degrees.
    """
    
    cdef Vector vn
    cdef double r, s, c, ci

    vn = v.normalise()

    with cython.cdivision(True):
        
        r = pi * angle / 180.0
    
    s = sin(r)
    c = cos(r)
    ci = 1.0 - c
    
    return new_affinematrix(vn.x * vn.x + (1.0 - vn.x * vn.x) * c,
                            vn.x * vn.y * ci - vn.z * s,
                            vn.x * vn.z * ci + vn.y * s,
                            0,
                            vn.x * vn.y * ci + vn.z * s,
                            vn.y * vn.y + (1.0 - vn.y * vn.y) * c,
                            vn.y * vn.z * ci - vn.x * s,
                            0,
                            vn.x * vn.z * ci - vn.y * s,
                            vn.y * vn.z * ci + vn.x * s,
                            vn.z * vn.z + (1.0 - vn.z * vn.z) * c,
                            0,
                            0,
                            0,
                            0,
                            1)


cpdef AffineMatrix rotate(double alpha, double beta, double gamma):
    """
    Returns an affine transform matrix representing an intrinsic rotation with
    an axis order (-Y)(-X)'Z''.
    
    If an object is aligned such that forward/back is the Z-axis, left/right is
    the X-axis and up/down is the Y-axis then alpha, beta and gamma correspond
    to the yaw, pitch and roll of the object.
    
    All angles are specified in degrees.
    """

    cdef double xr, yr, zr, sx, sy, sz, cx, cy, cz
    
    # optimised version of rotate_z(gamma) * rotate_x(-beta) * rotate_y(-alpha)
    xr = pi * -beta / 180.0
    yr = pi * -alpha / 180.0
    zr = pi * gamma / 180.0
    
    sx = sin(xr)
    sy = sin(yr)
    sz = sin(zr)
    
    cx = cos(xr)
    cy = cos(yr)
    cz = cos(zr)
    
    return new_affinematrix(cy*cz - sx*sy*sz,
                            -sz*cx,
                            sy*cz + sx*sz*cy,
                            0,
                            sz*cy + sx*sy*cz,
                            cx*cz,
                            sy*sz - sx*cy*cz,
                            0,
                            -sy*cx,
                            sx,
                            cx*cy,
                            0,
                            0,
                            0,
                            0,
                            1)


cpdef AffineMatrix scale(double x, double y, double z):
    """
    Returns an affine matrix representing a rescaling of the coordinate space.
    """ 
    
    if x <=0 or y <= 0 or z<= 0:
        
        raise ValueError("Scale must not be less than or equal to 0.")
    
    return new_affinematrix(x, 0, 0, 0,
                            0, y, 0, 0,
                            0, 0, z, 0,
                            0, 0, 0, 1)

