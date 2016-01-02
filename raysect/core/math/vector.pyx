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

cdef class Vector3D(_Vec3):
    """
    Represents a vector in 3D affine space.

    Vectors are described by their (x, y, z) coordinates in the chosen coordinate system. Standard Vector3D operations are
    supported such as addition, subtraction, scaling, dot product, cross product, normalisation and coordinate
    transformations.
    """

    def __init__(self, double x=0.0, double y=0.0, double z=1.0):
        """
        Vector3D Constructor.

        If no initial values are passed, Vector3D defaults to a unit vector
        aligned with the z-axis: Vector3D(0.0, 0.0, 1.0)
        """

        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        """Returns a string representation of the Vector3D object."""

        return "Vector3D(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

    def __richcmp__(self, object other, int op):
        """Provides basic vector comparison operations."""

        cdef Vector3D v

        if not isinstance(other, Vector3D):
            return NotImplemented

        v = <Vector3D> other
        if op == 2:     # __eq__()
            return self.x == v.x and self.y == v.y and self.z == v.z
        elif op == 3:   # __ne__()
            return self.x != v.x or self.y != v.y or self.z != v.z
        else:
            return NotImplemented

    def __neg__(self):
        """Returns a vector with the reverse orientation (negation operator)."""

        return new_vector3d(-self.x,
                            -self.y,
                            -self.z)

    def __add__(object x, object y):
        """Addition operator."""

        cdef _Vec3 vx, vy

        if isinstance(x, _Vec3) and isinstance(y, _Vec3):

            vx = <_Vec3>x
            vy = <_Vec3>y

            return new_vector3d(vx.x + vy.x,
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

            return new_vector3d(vx.x - vy.x,
                                vx.y - vy.y,
                                vx.z - vy.z)

        else:

            return NotImplemented

    def __mul__(object x, object y):
        """Multiplication operator."""

        cdef double s
        cdef Vector3D v
        cdef AffineMatrix3D m

        if isinstance(x, numbers.Real) and isinstance(y, Vector3D):

            s = <double>x
            v = <Vector3D>y

            return new_vector3d(s * v.x,
                                s * v.y,
                                s * v.z)

        elif isinstance(x, Vector3D) and isinstance(y, numbers.Real):

            s = <double>y
            v = <Vector3D>x

            return new_vector3d(s * v.x,
                                s * v.y,
                                s * v.z)

        elif isinstance(x, AffineMatrix3D) and isinstance(y, Vector3D):

            m = <AffineMatrix3D>x
            v = <Vector3D>y

            return new_vector3d(m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z,
                                m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z,
                                m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z)

        else:

            return NotImplemented

    @cython.cdivision(True)
    def __truediv__(object x, object y):
        """Division operator."""

        cdef double d
        cdef Vector3D v

        if isinstance(x, Vector3D) and isinstance(y, numbers.Real):

            d = <double>y

            # prevent divide my zero
            if d == 0.0:

                raise ZeroDivisionError("Cannot divide a vector by a zero scalar.")

            v = <Vector3D>x
            d = 1.0 / d

            return new_vector3d(d * v.x,
                                d * v.y,
                                d * v.z)

        else:

            raise TypeError("Unsupported operand type. Expects a real number.")

    cpdef Vector3D cross(self, _Vec3 v):
        """
        Calculates the cross product between this vector and the supplied
        vector:

            C = A.cross(B) <=> C = A x B

        Returns a Vector3D.
        """

        return new_vector3d(self.y * v.z - v.y * self.z,
                            self.z * v.x - v.z * self.x,
                            self.x * v.y - v.x * self.y)

    @cython.cdivision(True)
    cpdef Vector3D normalise(self):
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
        t = 1.0 / sqrt(t)

        return new_vector3d(self.x * t,
                            self.y * t,
                            self.z * t)

    cpdef Vector3D transform(self, AffineMatrix3D m):
        """
        Transforms the vector with the supplied AffineMatrix3D.

        The vector is transformed by pre-multiplying the vector by the affine
        matrix.

        This method is substantially faster than using the multiplication
        operator of AffineMatrix3D when called from cython code.
        """

        return new_vector3d(m.m[0][0] * self.x + m.m[0][1] * self.y + m.m[0][2] * self.z,
                            m.m[1][0] * self.x + m.m[1][1] * self.y + m.m[1][2] * self.z,
                            m.m[2][0] * self.x + m.m[2][1] * self.y + m.m[2][2] * self.z)

    cdef inline Vector3D neg(self):
        """
        Fast negation operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_vector3d(-self.x,
                            -self.y,
                            -self.z)

    cdef inline Vector3D add(self, _Vec3 v):
        """
        Fast addition operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_vector3d(self.x + v.x,
                            self.y + v.y,
                            self.z + v.z)

    cdef inline Vector3D sub(self, _Vec3 v):
        """
        Fast subtraction operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_vector3d(self.x - v.x,
                            self.y - v.y,
                            self.z - v.z)

    cdef inline Vector3D mul(self, double m):
        """
        Fast multiplication operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_vector3d(self.x * m,
                            self.y * m,
                            self.z * m)

    @cython.cdivision(True)
    cdef inline Vector3D div(self, double d):
        """
        Fast division operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        if d == 0.0:

            raise ZeroDivisionError("Cannot divide a vector by a zero scalar.")

        d = 1.0 / d

        return new_vector3d(self.x * d,
                            self.y * d,
                            self.z * d)

    cpdef Vector3D copy(self):
        """
        Returns a copy of the vector.
        """

        return new_vector3d(self.x,
                            self.y,
                            self.z)


