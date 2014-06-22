# cython: language_level=3

# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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
from raysect.core.math.point cimport Point

cdef class Normal(_Vec3):
    """
    Represents a normal vector in 3D affine space.
    """

    def __init__(self, v = (0.0, 0.0, 1.0)):
        """
        Normal vector constructor.

        If no initial values are passed, Normal defaults to a unit vector
        aligned with the z-axis: [1.0, 0.0, 0.0]

        Any three (or more) item indexable object can be used to initialise the
        normal. The x, y and z coordinates will be assigned the values of
        the items at indexes [0, 1, 2].

        e.g. Normal([4.0, 5.0, 6.0]) sets the x, y and z coordinates as 4.0,
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

                raise TypeError("Normal can only be initialised with an indexable object, containing numerical values, of length >= 3 items.")


    def __repr__(self):
        """Returns a string representation of the Normal object."""

        return "Normal([" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + "])"

    def __neg__(self):
        """Returns a normal with the reverse orientation."""

        return new_normal(-self.x,
                          -self.y,
                          -self.z)

    def __add__(object x, object y):
        """Addition operator."""

        cdef _Vec3 vx, vy

        if isinstance(x, _Vec3) and isinstance(y, _Vec3):

            vx = <_Vec3>x
            vy = <_Vec3>y

            return new_normal(vx.x + vy.x,
                              vx.y + vy.y,
                              vx.z + vy.z)

        else:

            return NotImplemented

    def __sub__(object x, object y):
        """Subtract operator."""

        cdef _Vec3 vx, vy

        if isinstance(x, _Vec3) and isinstance(y, _Vec3):

            vx = <_Vec3>x
            vy = <_Vec3>y

            return new_normal(vx.x - vy.x,
                              vx.y - vy.y,
                              vx.z - vy.z)

        else:

            return NotImplemented

    def __mul__(object x, object y):
        """Multiply operator."""

        cdef double s
        cdef Normal v
        cdef AffineMatrix m, minv

        if isinstance(x, numbers.Real) and isinstance(y, Normal):

            s = <double>x
            v = <Normal>y

            return new_normal(s * v.x,
                              s * v.y,
                              s * v.z)

        elif isinstance(x, Normal) and isinstance(y, numbers.Real):

            s = <double>y
            v = <Normal>x

            return new_normal(s * v.x,
                              s * v.y,
                              s * v.z)

        elif isinstance(x, AffineMatrix) and isinstance(y, Normal):

            m = <AffineMatrix>x
            v = <Normal>y

            minv = m.inverse()
            return new_normal(minv.m[0][0] * v.x + minv.m[1][0] * v.y + minv.m[2][0] * v.z,
                              minv.m[0][1] * v.x + minv.m[1][1] * v.y + minv.m[2][1] * v.z,
                              minv.m[0][2] * v.x + minv.m[1][2] * v.y + minv.m[2][2] * v.z)

        else:

            return NotImplemented

    def __truediv__(object x, object y):
        """Division operator."""

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

            return new_normal(d * v.x,
                              d * v.y,
                              d * v.z)

        else:

            return NotImplemented

    cpdef Vector cross(self, _Vec3 v):
        """
        Calculates the cross product between this vector and the supplied
        vector:

            C = A.cross(B) <=> C = A x B

        Returns a Vector.
        """

        return new_vector(self.y * v.z - v.y * self.z,
                          self.z * v.x - v.z * self.x,
                          self.x * v.y - v.x * self.y)

    cpdef Normal normalise(self):
        """
        Returns a normalised copy of the normal vector.

        The returned normal is normalised to length 1.0 - a unit vector.
        """

        cdef double t

        # if current length is zero, problem is ill defined
        t = self.x * self.x + self.y * self.y + self.z * self.z
        if t == 0.0:

            raise ZeroDivisionError("A zero length vector can not be normalised as the direction of a zero length vector is undefined.")

        # normalise and rescale vector
        with cython.cdivision(True):

            t = 1.0 / sqrt(t)

        return new_normal(self.x * t,
                          self.y * t,
                          self.z * t)

    cpdef Normal transform(self, AffineMatrix m):
        """
        Transforms the normal with the supplied Affine Matrix.

        The normal is multiplied by the inverse transpose of the transform
        matrix. This resulting normal remains perpendicular to its surface post
        transform.

        Warning: this method performs a costly inversion of the supplied matrix.
        If an inverse matrix is already available in scope, use the
        transform_with_inverse() method as it is considerably faster.

        For cython code this method is substantially faster than using the
        multiplication operator of the affine matrix.
        """

        cdef AffineMatrix minv
        minv = m.inverse()
        return new_normal(minv.m[0][0] * self.x + minv.m[1][0] * self.y + minv.m[2][0] * self.z,
                          minv.m[0][1] * self.x + minv.m[1][1] * self.y + minv.m[2][1] * self.z,
                          minv.m[0][2] * self.x + minv.m[1][2] * self.y + minv.m[2][2] * self.z)

    cpdef Normal transform_with_inverse(self, AffineMatrix m):
        """
        Transforms the normal with the supplied inverse Affine Matrix.

        If an inverse matrix is already available in scope, this method is
        considerably faster than the transform() method - it skips a matrix
        inversion required to calculate the transformed normal (see the
        transform() documentation).

        For cython code this method is substantially faster than using the
        multiplication operator of the affine matrix.
        """

        return new_normal(m.m[0][0] * self.x + m.m[1][0] * self.y + m.m[2][0] * self.z,
                          m.m[0][1] * self.x + m.m[1][1] * self.y + m.m[2][1] * self.z,
                          m.m[0][2] * self.x + m.m[1][2] * self.y + m.m[2][2] * self.z)

    cdef inline Normal neg(self):

        return new_normal(-self.x,
                          -self.y,
                          -self.z)

    cdef inline Normal add(self, _Vec3 v):

        return new_normal(self.x + v.x,
                          self.y + v.y,
                          self.z + v.z)

    cdef inline Normal sub(self, _Vec3 v):

        return new_normal(self.x - v.x,
                          self.y - v.y,
                          self.z - v.z)

    cdef inline Normal mul(self, double m):

        return new_normal(self.x * m,
                          self.y * m,
                          self.z * m)

    cdef inline Normal div(self, double d):

        if d == 0.0:

            raise ZeroDivisionError("Cannot divide a vector by a zero scalar.")

        with cython.cdivision:

            d = 1.0 / d

        return new_normal(self.x * d,
                          self.y * d,
                          self.z * d)

    cpdef Normal copy(self):
        """
        Returns a copy of the normal.
        """

        return new_normal(self.x,
                          self.y,
                          self.z)