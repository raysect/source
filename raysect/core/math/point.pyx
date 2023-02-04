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

cimport cython
from libc.math cimport sqrt, NAN
from raysect.core.math.vector cimport new_vector2d, new_vector3d
from raysect.core.math._vec3 cimport _Vec3


@cython.freelist(256)
cdef class Point3D:
    """
    Represents a point in 3D affine space.

    A point is a location in 3D space which is defined by its x, y and z coordinates in a given coordinate system.
    Vectors can be added/subtracted from Points yielding another Vector3D. You can also find the Vector3D and distance
    between two Points, and transform a Point3D from one coordinate system to another.

    If no initial values are passed, Point3D defaults to the origin:
    Point3D(0.0, 0.0, 0.0)

    :param float x: initial x coordinate, defaults to x = 0.0.
    :param float y: initial y coordinate, defaults to y = 0.0.
    :param float z: initial z coordinate, defaults to z = 0.0.

    :ivar float x: x-coordinate
    :ivar float y: y-coordinate
    :ivar float z: z-coordinate

    .. code-block:: pycon

        >>> from raysect.core import Point3D
        >>> a = Point3D(0, 1, 2)

    """

    def __init__(self, double x=0.0, double y=0.0, double z=0.0):

        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        """Returns a string representation of the Point3D object."""

        return "Point3D(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

    def __richcmp__(self, object other, int op):
        """Provides basic point comparison operations."""

        cdef Point3D p

        if not isinstance(other, Point3D):
            return NotImplemented

        p = <Point3D> other
        if op == 2:     # __eq__()
            return self.x == p.x and self.y == p.y and self.z == p.z
        elif op == 3:   # __ne__()
            return self.x != p.x or self.y != p.y or self.z != p.z
        else:
            return NotImplemented

    def __getitem__(self, int i):
        """Returns the point coordinates by index ([0,1,2] -> [x,y,z]).

            >>> a = Point3D(1, 0, 0)
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
        """Sets the point coordinates by index ([0,1,2] -> [x,y,z]).

            >>> a = Point3D(1, 0, 0)
            >>> a[1] = 2
            >>> a
            Point3D(1.0, 2.0, 0.0)
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
        """Iterates over the coordinates (x, y, z)

            >>> a = Point3D(0, 1, 2)
            >>> x, y, z = a
            >>> x, y, z
            (0.0, 1.0, 2.0)
        """
        yield self.x
        yield self.y
        yield self.z

    def __add__(object x, object y):
        """Addition operator.

            >>> Point3D(1, 0, 0) + Vector3D(0, 1, 0)
            Point3D(1.0, 1.0, 0.0)
        """

        cdef Point3D p
        cdef _Vec3 v

        if isinstance(x, Point3D) and isinstance(y, _Vec3):

            p = <Point3D>x
            v = <_Vec3>y

        else:

            return NotImplemented

        return new_point3d(p.x + v.x,
                           p.y + v.y,
                           p.z + v.z)

    def __sub__(object x, object y):
        """Subtraction operator.

            >>> Point3D(1, 0, 0) - Vector3D(0, 1, 0)
            Point3D(1.0, -1.0, 0.0)
        """

        cdef Point3D p
        cdef _Vec3 v

        if isinstance(x, Point3D) and isinstance(y, _Vec3):

            p = <Point3D>x
            v = <_Vec3>y

            return new_point3d(p.x - v.x,
                               p.y - v.y,
                               p.z - v.z)

        else:

            return NotImplemented

    @cython.cdivision(True)
    def __mul__(object x, object y):
        """Multiplication operator.

        :param AffineMatrix3D x: transformation matrix x
        :param Point3D y: point to transform
        :return: Matrix multiplication of a 3D transformation matrix with the input point.
        :rtype: Point3D
        """

        cdef AffineMatrix3D m
        cdef Point3D v
        cdef double w

        if isinstance(x, AffineMatrix3D) and isinstance(y, Point3D):

            m = <AffineMatrix3D>x
            v = <Point3D>y

            # 4th element of homogeneous coordinate
            w = m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3]
            if w == 0.0:

                raise ZeroDivisionError("Bad matrix transform, 4th element of homogeneous coordinate is zero.")

            # pre divide for speed (dividing is much slower than multiplying)
            w = 1.0 / w

            return new_point3d((m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3]) * w,
                               (m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3]) * w,
                               (m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3]) * w)

        return NotImplemented

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.x, self.y, self.z

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.x = state[0]
        self.y = state[1]
        self.z = state[2]

    cpdef Vector3D vector_to(self, Point3D p):
        """
        Returns a vector from this point to the passed point.

        :param Point3D p: the point to which a vector will be calculated.
        :rtype: Vector3D

        .. code-block:: pycon

            >>> a = Point3D(0, 1, 2)
            >>> b = Point3D(1, 1, 1)
            >>> a.vector_to(b)
            Vector3D(1.0, 0.0, -1.0)

        """

        return new_vector3d(p.x - self.x,
                            p.y - self.y,
                            p.z - self.z)

    cpdef double distance_to(self, Point3D p):
        """
        Returns the distance between this point and the passed point.

        :param Point3D p: the point to which the distance will be calculated
        :rtype: float

        .. code-block:: pycon

            >>> a = Point3D(0, 1, 2)
            >>> b = Point3D(1, 1, 1)
            >>> a.distance_to(b)
            1.4142135623730951
        """

        cdef double x, y, z
        x = p.x - self.x
        y = p.y - self.y
        z = p.z - self.z
        return sqrt(x*x + y*y + z*z)

    @cython.cdivision(True)
    cpdef Point3D transform(self, AffineMatrix3D m):
        """
        Transforms the point with the supplied Affine Matrix.

        The point is transformed by premultiplying the point by the affine
        matrix.

        For cython code this method is substantially faster than using the
        multiplication operator of the affine matrix.

        This method expects a valid affine transform. For speed reasons, minimal
        checks are performed on the matrix.

        :param AffineMatrix3D m: The affine matrix describing the required coordinate transformation.
        :return: A new instance of this point that has been transformed with the supplied Affine Matrix.
        :rtype: Point3D
        """

        cdef double w

        # 4th element of homogeneous coordinate
        w = m.m[3][0] * self.x + m.m[3][1] * self.y + m.m[3][2] * self.z + m.m[3][3]
        if w == 0.0:

            raise ZeroDivisionError("Bad matrix transform, 4th element of homogeneous coordinate is zero.")

        # pre divide for speed (dividing is much slower than multiplying)
        w = 1.0 / w

        return new_point3d((m.m[0][0] * self.x + m.m[0][1] * self.y + m.m[0][2] * self.z + m.m[0][3]) * w,
                           (m.m[1][0] * self.x + m.m[1][1] * self.y + m.m[1][2] * self.z + m.m[1][3]) * w,
                           (m.m[2][0] * self.x + m.m[2][1] * self.y + m.m[2][2] * self.z + m.m[2][3]) * w)

    cdef Point3D add(self, _Vec3 v):
        """
        Fast addition operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_point3d(self.x + v.x,
                           self.y + v.y,
                           self.z + v.z)

    cdef Point3D sub(self, _Vec3 v):
        """
        Fast subtraction operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_point3d(self.x - v.x,
                           self.y - v.y,
                           self.z - v.z)

    cpdef Point3D copy(self):
        """
        Returns a copy of the point.

        :rtype: Point3D

        .. code-block:: pycon

            >>> a = Point3D(0, 1, 2)
            >>> a.copy()
            Point3D(0.0, 1.0, 2.0)
        """

        return new_point3d(self.x,
                           self.y,
                           self.z)

    cdef double get_index(self, int index) nogil:
        """
        Fast getting of coordinates via indexing.

        Cython equivalent to __getitem__, without the checks and call overhead.

        If an invalid index is passed this function return NaN.
        """

        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            return NAN

    cdef void set_index(self, int index, double value) nogil:
        """
        Fast setting of coordinates via indexing.

        Cython equivalent to __setitem__, without the checks and call overhead.

        If an invalid index is passed this function does nothing.
        """

        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value


cdef class Point2D:
    """
    Represents a point in 2D affine space.

    A 2D point is a location in 2D space which is defined by its x and y coordinates in a given coordinate system.
    Vector2D objects can be added/subtracted from Point2D yielding another Vector2D. You can also find the Vector2D
    and distance between two Point2Ds, and transform a Point2D from one coordinate system to another.

    If no initial values are passed, Point2D defaults to the origin: Point2D(0.0, 0.0)

    :param float x: initial x coordinate, defaults to x = 0.0.
    :param float y: initial y coordinate, defaults to y = 0.0.

    :ivar float x: x-coordinate
    :ivar float y: y-coordinate

    .. code-block:: pycon

        >>> from raysect.core import Point2D
        >>>
        >>> a = Point2D(1, 1)

    """

    def __init__(self, double x=0.0, double y=0.0):

        self.x = x
        self.y = y

    def __repr__(self):
        """Returns a string representation of the Point2D object."""

        return "Point2D(" + str(self.x) + ", " + str(self.y) + ")"

    def __richcmp__(self, object other, int op):
        """Provides basic point comparison operations."""

        cdef Point2D p

        if not isinstance(other, Point2D):
            return NotImplemented

        p = <Point2D> other
        if op == 2:     # __eq__()
            return self.x == p.x and self.y == p.y
        elif op == 3:   # __ne__()
            return self.x != p.x or self.y != p.y
        else:
            return NotImplemented

    def __getitem__(self, int i):
        """Returns the point coordinates by index ([0,1] -> [x,y]).

            >>> a = Point2D(1, 0)
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
        """Sets the point coordinates by index ([0,1] -> [x,y]).

            >>> a = Point2D(1, 0)
            >>> a[1] = 2
            >>> a
            Point2D(1.0, 2.0)
        """

        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            raise IndexError("Index out of range [0, 1].")

    def __iter__(self):
        """Iterates over the coordinates (x, y)

        >>> a = Point2D(1, 1)
        >>> x, y = a
        >>> x, y
        (1.0, 1.0)

        """
        yield self.x
        yield self.y

    def __add__(object x, object y):
        """Addition operator.

            >>> Point2D(1, 0) + Vector2D(0, 1)
            Point2D(1.0, 1.0)
        """

        cdef Point2D p
        cdef Vector2D v

        if isinstance(x, Point2D) and isinstance(y, Vector2D):

            p = <Point2D>x
            v = <Vector2D>y

        else:

            return NotImplemented

        return new_point2d(p.x + v.x, p.y + v.y)

    def __sub__(object x, object y):
        """Subtraction operator.

            >>> Point2D(1, 0) - Vector2D(0, 1)
            Point2D(1.0, -1.0)
        """

        cdef Point2D p
        cdef Vector2D v

        if isinstance(x, Point2D) and isinstance(y, Vector2D):

            p = <Point2D>x
            v = <Vector2D>y

            return new_point2d(p.x - v.x, p.y - v.y)

        else:

            return NotImplemented

    @cython.cdivision(True)
    def __mul__(object x, object y):
        """Multiply operator."""
        raise NotImplemented

        # cdef AffineMatrix3D m
        # cdef Point3D v
        # cdef double w
        #
        # if isinstance(x, AffineMatrix3D) and isinstance(y, Point3D):
        #
        #     m = <AffineMatrix3D>x
        #     v = <Point3D>y
        #
        #     # 4th element of homogeneous coordinate
        #     w = m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3]
        #     if w == 0.0:
        #
        #         raise ZeroDivisionError("Bad matrix transform, 4th element of homogeneous coordinate is zero.")
        #
        #     # pre divide for speed (dividing is much slower than multiplying)
        #     w = 1.0 / w
        #
        #     return new_point3d((m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3]) * w,
        #                      (m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3]) * w,
        #                      (m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3]) * w)
        #
        # return NotImplemented

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.x, self.y

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.x = state[0]
        self.y = state[1]

    cpdef Vector2D vector_to(self, Point2D p):
        """
        Returns a vector from this point to the passed point.

        :param Point2D p: point to which a vector will be calculated
        :rtype: Vector2D

        .. code-block:: pycon

            >>> a = Point2D(1, 0)
            >>> b = Point2D(1, 1)
            >>> a.vector_to(b)
            Vector2D(0.0, 1.0)

        """

        return new_vector2d(p.x - self.x, p.y - self.y)

    cpdef double distance_to(self, Point2D p):
        """
        Returns the distance between this point and the passed point.

        :param Point2D p: the point to which the distance will be calculated
        :rtype: float

        .. code-block:: pycon

            >>> a = Point2D(1, 0)
            >>> b = Point2D(1, 1)
            >>> a.distance_to(b)
            1.0

        """

        cdef double x, y
        x = p.x - self.x
        y = p.y - self.y
        return sqrt(x*x + y*y)

    # @cython.cdivision(True)
    # cpdef Point3D transform(self, AffineMatrix3D m):
    #     """
    #     Transforms the point with the supplied Affine Matrix.
    #
    #     The point is transformed by premultiplying the point by the affine
    #     matrix.
    #
    #     For cython code this method is substantially faster than using the
    #     multiplication operator of the affine matrix.
    #
    #     This method expects a valid affine transform. For speed reasons, minimal
    #     checks are performed on the matrix.
    #     """
    #
    #     cdef double w
    #
    #     # 4th element of homogeneous coordinate
    #     w = m.m[3][0] * self.x + m.m[3][1] * self.y + m.m[3][2] * self.z + m.m[3][3]
    #     if w == 0.0:
    #
    #         raise ZeroDivisionError("Bad matrix transform, 4th element of homogeneous coordinate is zero.")
    #
    #     # pre divide for speed (dividing is much slower than multiplying)
    #     w = 1.0 / w
    #
    #     return new_point3d((m.m[0][0] * self.x + m.m[0][1] * self.y + m.m[0][2] * self.z + m.m[0][3]) * w,
    #                      (m.m[1][0] * self.x + m.m[1][1] * self.y + m.m[1][2] * self.z + m.m[1][3]) * w,
    #                      (m.m[2][0] * self.x + m.m[2][1] * self.y + m.m[2][2] * self.z + m.m[2][3]) * w)

    cdef Point2D add(self, Vector2D v):
        """
        Fast addition operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_point2d(self.x + v.x, self.y + v.y)

    cdef Point2D sub(self, Vector2D v):
        """
        Fast subtraction operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_point2d(self.x - v.x, self.y - v.y)

    cpdef Point2D copy(self):
        """
        Returns a copy of the point.

        :rtype: Point2D

        .. code-block:: pycon

            >>> a = Point2D(1, 1)
            >>> a.copy()
            Point2D(1.0, 1.0)

        """
        return new_point2d(self.x, self.y)

    cdef double get_index(self, int index) nogil:
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
            return NAN

    cdef void set_index(self, int index, double value) nogil:
        """
        Fast setting of coordinates via indexing.

        Cython equivalent to __setitem__, without the checks and call overhead.

        If an invalid index is passed this function does nothing.
        """
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
