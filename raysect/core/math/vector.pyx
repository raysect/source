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
from raysect.core.math.point cimport Point

cdef class Vector(_Vec3):
    """
    Represents a vector in 3D affine space.
    """

    def __init__(self, v = (0.0, 0.0, 1.0)):
        """
        Vector constructor.

        If no initial values are passed, Vector defaults to a unit vector
        aligned with the z-axis: [0.0, 0.0, 1.0]

        Any three (or more) item indexable object can be used to initialise the
        vector. The x, y and z coordinates will be assigned the values of
        the items at indexes [0, 1, 2].

        e.g. Vector([4.0, 5.0, 6.0]) sets the x, y and z coordinates as 4.0,
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

                raise TypeError("Vector can only be initialised with an indexable object, containing numerical values, of length >= 3 items.")

    def __repr__(self):
        """Returns a string representation of the Vector object."""

        return "Vector([" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + "])"

    def __neg__(self):
        """Returns a vector with the reverse orientation."""

        return new_vector(-self.x,
                          -self.y,
                          -self.z)

    def __add__(object x, object y):
        """Addition operator."""

        cdef _Vec3 vx, vy

        if isinstance(x, _Vec3) and isinstance(y, _Vec3):

            vx = <_Vec3>x
            vy = <_Vec3>y

            return new_vector(vx.x + vy.x,
                              vx.y + vy.y,
                              vx.z + vy.z)

        else:

            return NotImplemented

    def __sub__(object x, object y):
        """Subtraction operator."""

        cdef _Vec3 vx, vy

        if isinstance(x, _Vec3) and isinstance(y, _Vec3):

            vx = <_Vec3>x
            vy = <_Vec3>y

            return new_vector(vx.x - vy.x,
                              vx.y - vy.y,
                              vx.z - vy.z)

        else:

            return NotImplemented

    def __mul__(object x, object y):
        """Multiplication operator."""

        cdef double s
        cdef Vector v
        cdef AffineMatrix m

        if isinstance(x, numbers.Real) and isinstance(y, Vector):

            s = <double>x
            v = <Vector>y

            return new_vector(s * v.x,
                              s * v.y,
                              s * v.z)

        elif isinstance(x, Vector) and isinstance(y, numbers.Real):

            s = <double>y
            v = <Vector>x

            return new_vector(s * v.x,
                              s * v.y,
                              s * v.z)

        elif isinstance(x, AffineMatrix) and isinstance(y, Vector):

            m = <AffineMatrix>x
            v = <Vector>y

            return new_vector(m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z,
                              m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z,
                              m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z)

        else:

            return NotImplemented

    def __truediv__(object x, object y):
        """Division operator."""

        cdef double d
        cdef Vector v

        if isinstance(x, Vector) and isinstance(y, numbers.Real):

            d = <double>y

            # prevent divide my zero
            if d == 0.0:

                raise ZeroDivisionError("Cannot divide a vector by a zero scalar.")

            v = <Vector>x

            with cython.cdivision(True):

                d = 1.0 / d

            return new_vector(d * v.x,
                              d * v.y,
                              d * v.z)

        else:

            raise TypeError("Unsupported operand type. Expects a real number.")

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


    cpdef Vector normalise(self):
        """
        Returns a normalised copy of the vector.

        The returned vector is normalised to length 1.0 - a unit vector.
        """

        cdef double t

        # if current length is zero, problem is ill defined
        t = self.x * self.x + self.y * self.y + self.z * self.z
        if t == 0.0:

            raise ZeroDivisionError("A zero length vector can not be normalised as the direction of a zero length vector is undefined.")

        # normalise and rescale vector
        with cython.cdivision(True):

            t = 1.0 / sqrt(t)

        return new_vector(self.x * t,
                          self.y * t,
                          self.z * t)

    cpdef Vector transform(self, AffineMatrix m):
        """
        Transforms the vector with the supplied Affine Matrix.

        The vector is transformed by premultiplying the vector by the affine
        matrix.

        For cython code this method is substantially faster than using the
        multiplication operator of the affine matrix.
        """

        return new_vector(m.m[0][0] * self.x + m.m[0][1] * self.y + m.m[0][2] * self.z,
                          m.m[1][0] * self.x + m.m[1][1] * self.y + m.m[1][2] * self.z,
                          m.m[2][0] * self.x + m.m[2][1] * self.y + m.m[2][2] * self.z)


    cdef inline Vector neg(self):

        return new_vector(-self.x,
                          -self.y,
                          -self.z)

    cdef inline Vector add(self, _Vec3 v):

        return new_vector(self.x + v.x,
                          self.y + v.y,
                          self.z + v.z)

    cdef inline Vector sub(self, _Vec3 v):

        return new_vector(self.x - v.x,
                          self.y - v.y,
                          self.z - v.z)

    cdef inline Vector mul(self, double m):

        return new_vector(self.x * m,
                          self.y * m,
                          self.z * m)

    cdef inline Vector div(self, double d):

        if d == 0.0:

            raise ZeroDivisionError("Cannot divide a vector by a zero scalar.")

        with cython.cdivision:

            d = 1.0 / d

        return new_vector(self.x * d,
                          self.y * d,
                          self.z * d)

    cpdef Vector copy(self):
        """
        Returns a copy of the vector.
        """

        return new_vector(self.x,
                          self.y,
                          self.z)
