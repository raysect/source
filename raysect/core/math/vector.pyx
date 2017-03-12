# cython: language_level=3

# Copyright (c) 2014-2017, Dr Alex Meakins, Raysect Project
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
from libc.math cimport sqrt, fabs


cdef class Vector3D(_Vec3):
    """
    Represents a vector in 3D affine space.

    Vectors are described by their (x, y, z) coordinates in the chosen coordinate system. Standard Vector3D operations are
    supported such as addition, subtraction, scaling, dot product, cross product, normalisation and coordinate
    transformations.

    If no initial values are passed, Vector3D defaults to a unit vector
    aligned with the z-axis: Vector3D(0.0, 0.0, 1.0)

    :param float x: initial x coordinate, defaults to x = 0.0.
    :param float y: initial y coordinate, defaults to y = 0.0.
    :param float z: initial z coordinate, defaults to z = 0.0.

    :ivar float x: x-coordinate
    :ivar float y: y-coordinate
    :ivar float z: z-coordinate
    """

    def __init__(self, double x=0.0, double y=0.0, double z=1.0):

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

    def __getitem__(self, int i):
        """Returns the vector coordinates by index ([0,1,2] -> [x,y,z]).

            >>> a = Vector3D(1, 0, 0)
            >>> a[0]
            1
        """

        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        elif i == 2:
            return self.z
        else:
            raise IndexError("Index out of range [0, 2].")

    def __setitem__(self, int i, double value):
        """Sets the vector coordinates by index ([0,1,2] -> [x,y,z]).

            >>> a = Vector3D(1, 0, 0)
            >>> a[1] = 2
            >>> a
            Vector3D(1.0, 2.0, 0.0)
        """

        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        elif i == 2:
            self.z = value
        else:
            raise IndexError("Index out of range [0, 2].")

    def __iter__(self):
        """ Iterates over the vector coordinates (x, y, z) """
        yield self.x
        yield self.y
        yield self.z

    def __neg__(self):
        """Returns a vector with the reverse orientation (negation operator)."""

        return new_vector3d(-self.x,
                            -self.y,
                            -self.z)

    def __add__(object x, object y):
        """Addition operator.

            >>> Vector3D(1, 0, 0) + Vector3D(0, 1, 0)
            Vector3D(1.0, 1.0, 0.0)
        """

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
        """Subtraction operator.

            >>> Vector3D(1, 0, 0) - Vector3D(0, 1, 0)
            Vector3D(1.0, -1.0, 0.0)
        """

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
        """Multiplication operator.

        3D vectors can be multiplied with both scalars and transformation matrices.

            >>> 2 * Vector3D(1, 2, 3)
            Vector3D(2.0, 4.0, 6.0)
            >>> rotate_x(90) * Vector3D(0, 0, 1)
            Vector3D(0.0, -1.0, 0.0)
        """

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
        """Division operator.

            >>> Vector3D(1, 1, 1) / 2
            Vector3D(0.5, 0.5, 0.5)
        """

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
        Calculates the cross product between this vector and the supplied vector

        C = A.cross(B) <=> :math:`\\vec{C} = \\vec{A} \\times \\vec{B}`

        :param Vector3D v: An input vector with which to calculate the cross product.
        :rtype: Vector3D
        """

        return new_vector3d(self.y * v.z - v.y * self.z,
                            self.z * v.x - v.z * self.x,
                            self.x * v.y - v.x * self.y)

    @cython.cdivision(True)
    cpdef Vector3D normalise(self):
        """
        Returns a normalised copy of the vector.

        The returned vector is normalised to length 1.0 - a unit vector.

        :rtype: Vector3D
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

        .. math::

            \\vec{C} = \\textbf{A} \\times \\vec{B}

        This method is substantially faster than using the multiplication
        operator of AffineMatrix3D when called from cython code.

        :param AffineMatrix3D m: The affine matrix describing the required coordinate transformation.
        :return: A new instance of this vector that has been transformed with the supplied Affine Matrix.
        :rtype: Vector3D
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

        :rtype: Vector3D
        """

        return new_vector3d(self.x,
                            self.y,
                            self.z)

    # todo: this is common code with normal, move into math.cython and call
    cpdef Vector3D orthogonal(self):
        """
        Returns a unit vector that is guaranteed to be orthogonal to the vector.

        :rtype: vector3D
        """

        cdef:
            Vector3D n
            Vector3D v
            double m

        n = self.normalise()

        # try x-axis first, if too closely aligned use the y-axis
        v = new_vector3d(1, 0, 0)
        if fabs(n.dot(v)) > 0.5:
            v = new_vector3d(0, 1, 0)

        # make vector perpendicular to normal
        m = n.dot(v)
        v = new_vector3d(v.x - m * n.x, v.y - m * n.y, v.z - m * n.z)
        v = v.normalise()

        return v


cdef class Vector2D:
    """
    Represents a vector in 2D space.

    2D vectors are described by their (x, y) coordinates. Standard Vector2D operations are
    supported such as addition, subtraction, scaling, dot product, cross product and normalisation.

    If no initial values are passed, Vector2D defaults to a unit vector
    aligned with the x-axis: Vector2D(1.0, 0.0)

    :param float x: initial x coordinate, defaults to x = 0.0.
    :param float y: initial y coordinate, defaults to y = 0.0.

    :ivar float x: x-coordinate
    :ivar float y: y-coordinate
    """

    def __init__(self, double x=1.0, double y=0.0):

        self.x = x
        self.y = y

    def __repr__(self):
        """Returns a string representation of the Vector2D object."""

        return "Vector2D(" + str(self.x) + ", " + str(self.y) + ")"

    def __richcmp__(self, object other, int op):
        """Provides basic vector comparison operations."""

        cdef Vector2D v

        if not isinstance(other, Vector2D):
            return NotImplemented

        v = <Vector2D> other
        if op == 2:     # __eq__()
            return self.x == v.x and self.y == v.y
        elif op == 3:   # __ne__()
            return self.x != v.x or self.y != v.y
        else:
            return NotImplemented

    def __getitem__(self, int i):
        """Returns the vector coordinates by index ([0,1] -> [x,y]).

            >>> a = Vector2D(1, 0)
            >>> a[0]
            1
        """

        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            raise IndexError("Index out of range [0, 1].")

    def __setitem__(self, int i, double value):
        """Sets the vector coordinates by index ([0,1] -> [x,y]).

            >>> a = Vector2D(1, 0)
            >>> a[1] = 2
            >>> a
            Vector2D(1.0, 2.0)
        """

        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            raise IndexError("Index out of range [0, 1].")

    def __iter__(self):
        """ Iterates over the vector coordinates (x, y) """
        yield self.x
        yield self.y

    def __neg__(self):
        """Returns a vector with the reverse orientation (negation operator)."""

        return new_vector2d(-self.x, -self.y)

    def __add__(object x, object y):
        """Addition operator.

            >>> Vector2D(1, 0) + Vector2D(0, 1)
            Vector2D(1.0, 1.0)
        """

        cdef Vector2D vx, vy

        if isinstance(x, Vector2D) and isinstance(y, Vector2D):

            vx = <Vector2D>x
            vy = <Vector2D>y

            return new_vector2d(vx.x + vy.x, vx.y + vy.y)

        else:

            return NotImplemented

    def __sub__(object x, object y):
        """Subtraction operator.

            >>> Vector2D(1, 0) - Vector2D(0, 1)
            Vector2D(1.0, -1.0)
        """

        cdef Vector2D vx, vy

        if isinstance(x, Vector2D) and isinstance(y, Vector2D):

            vx = <Vector2D>x
            vy = <Vector2D>y

            return new_vector2d(vx.x - vy.x, vx.y - vy.y)

        else:

            return NotImplemented

    # TODO - add 2D affine transformations
    def __mul__(object x, object y):
        """Multiplication operator.

            >>> 2 * Vector3D(1, 2)
            Vector2D(2.0, 4.0)
        """

        cdef double s
        cdef Vector2D v
        # cdef AffineMatrix2D m

        if isinstance(x, numbers.Real) and isinstance(y, Vector2D):

            s = <double>x
            v = <Vector2D>y

            return new_vector2d(s * v.x, s * v.y,)

        elif isinstance(x, Vector2D) and isinstance(y, numbers.Real):

            s = <double>y
            v = <Vector2D>x

            return new_vector2d(s * v.x, s * v.y)

        # elif isinstance(x, AffineMatrix3D) and isinstance(y, Vector3D):
        #
        #     m = <AffineMatrix3D>x
        #     v = <Vector3D>y
        #
        #     return new_vector3d(m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z,
        #                         m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z,
        #                         m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z)

        else:

            return NotImplemented

    @cython.cdivision(True)
    def __truediv__(object x, object y):
        """Division operator.

            >>> Vector2D(1, 1) / 2
            Vector2D(0.5, 0.5)
        """

        cdef double d
        cdef Vector2D v

        if isinstance(x, Vector2D) and isinstance(y, numbers.Real):

            d = <double>y

            # prevent divide my zero
            if d == 0.0:

                raise ZeroDivisionError("Cannot divide a vector by a zero scalar.")

            v = <Vector2D>x
            d = 1.0 / d

            return new_vector2d(d * v.x, d * v.y)

        else:

            raise TypeError("Unsupported operand type. Expects a real number.")

    property length:
        """
        The vector's length.

        Raises a ZeroDivisionError if an attempt is made to change the length of
        a zero length vector. The direction of a zero length vector is
        undefined hence it can not be lengthened.
        """

        def __get__(self):

            return self.get_length()

        def __set__(self, double v):

            self.set_length(v)

    cpdef double dot(self, Vector2D v):
        """
        Calculates the dot product between this vector and the supplied vector.

        Returns a scalar.
        """

        return self.x * v.x + self.y * v.y

    cdef inline double get_length(self):
        """
        Fast function to obtain the vectors length.

        Cython only, equivalent to length.__get__() property.

        Use instead of Python attribute access in cython code.
        """

        return sqrt(self.x * self.x + self.y * self.y)

    @cython.cdivision(True)
    cdef inline void set_length(self, double v) except *:
        """
        Fast function to set the vectors length.

        Cython only, equivalent to length.__set__() property.

        Use instead of Python attribute access in cython code.
        """

        cdef double t

        # if current length is zero, problem is ill defined
        t = self.x * self.x + self.y * self.y
        if t == 0.0:
            raise ZeroDivisionError("A zero length vector can not be rescaled as the direction of a zero length vector is undefined.")

        # normalise and rescale vector
        t = v / sqrt(t)

        self.x = self.x * t
        self.y = self.y * t

    cdef inline double get_index(self, int index):
        """
        Fast getting of coordinates via indexing.

        Cython equivalent to __getitem__, without the checks and call overhead.

        If an invalid index is passed this function return NaN.
        """

        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            return float("NaN")

    cdef inline void set_index(self, int index, double value):
        """
        Fast setting of coordinates via indexing.

        Cython equivalent to __setitem__, without the checks and call overhead.

        If an invalid index is passed this function does nothing.
        """

        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value

    cpdef double cross(self, Vector2D v):
        """
        Calculates the 2D cross product analogue between this vector and the supplied vector

        C = A.cross(B) <=> C = A x B <=> det(A, B) = A.x B.y - A.y B.x

        Note that for 2D vectors, the cross product is the equivalent of the determinant of a
        2x2 matrix. The result is a scalar.

        :param Vector2D v: An input vector with which to calculate the cross product.
        :rtype: float
        """

        return self.x * v.y - self.y * v.x

    @cython.cdivision(True)
    cpdef Vector2D normalise(self):
        """
        Returns a normalised copy of the vector.

        The returned vector is normalised to length 1.0 - a unit vector.

        :rtype: Vector2D
        """

        cdef double t

        # if current length is zero, problem is ill defined
        t = self.x * self.x + self.y * self.y
        if t == 0.0:

            raise ZeroDivisionError("A zero length vector can not be normalised as the direction of a zero length vector is undefined.")

        # normalise and rescale vector
        t = 1.0 / sqrt(t)

        return new_vector2d(self.x * t, self.y * t)

    # cpdef Vector3D transform(self, AffineMatrix3D m):
    #     """
    #     Transforms the vector with the supplied AffineMatrix3D.
    #
    #     The vector is transformed by pre-multiplying the vector by the affine
    #     matrix.
    #
    #     This method is substantially faster than using the multiplication
    #     operator of AffineMatrix3D when called from cython code.
    #
    #     :param AffineMatrix3D m: The affine matrix describing the required coordinate transformation.
    #     :return: A new instance of this vector that has been transformed with the supplied Affine Matrix.
    #     :rtype: Vector3D
    #     """
    #
    #     return new_vector3d(m.m[0][0] * self.x + m.m[0][1] * self.y + m.m[0][2] * self.z,
    #                         m.m[1][0] * self.x + m.m[1][1] * self.y + m.m[1][2] * self.z,
    #                         m.m[2][0] * self.x + m.m[2][1] * self.y + m.m[2][2] * self.z)

    cdef inline Vector2D neg(self):
        """
        Fast negation operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_vector2d(-self.x, -self.y)

    cdef inline Vector2D add(self, Vector2D v):
        """
        Fast addition operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_vector2d(self.x + v.x, self.y + v.y)

    cdef inline Vector2D sub(self, Vector2D v):
        """
        Fast subtraction operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_vector2d(self.x - v.x, self.y - v.y)

    cdef inline Vector2D mul(self, double m):
        """
        Fast multiplication operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_vector2d(self.x * m, self.y * m)

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

        return new_vector2d(self.x * d, self.y * d)

    cpdef Vector2D copy(self):
        """
        Returns a copy of the vector.

        :rtype: Vector2D
        """

        return new_vector2d(self.x, self.y)

    cpdef Vector2D orthogonal(self):
        """
        Returns a unit vector that is guaranteed to be orthogonal to the vector.

        :rtype: vector2D
        """

        cdef:
            Vector2D n

        n = self.normalise()

        return new_vector2d(-n.y, n.x)


