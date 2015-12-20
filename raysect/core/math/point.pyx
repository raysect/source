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

cimport cython
from libc.math cimport sqrt
from raysect.core.math.vector cimport new_vector
from raysect.core.math._vec3 cimport _Vec3

cdef class Point:
    """
    Represents a point in 3D affine space.

    A point is a location in 3D space which is defined by its x, y and z coordinates in a given coordinate system.
    Vectors can be added/subtracted from Points yielding another Vector. You can also find the Vector and distance
    between two Points, and transform a Point from one coordinate system to another.
    """

    def __init__(self, double x=0.0, double y=0.0, double z=0.0):
        """
        Point constructor.

        If no initial values are passed, Point defaults to the origin:
        Point(0.0, 0.0, 0.0)
        """

        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        """Returns a string representation of the Point object."""

        return "Point([" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + "])"

    def __richcmp__(self, object other, int op):
        """Provides basic point comparison operations."""

        cdef Point p

        if not isinstance(other, Point):
            return NotImplemented

        p = <Vector> other
        if op == 2:     # __eq__()
            return self.x == p.x and self.y == p.y and self.z == p.z
        elif op == 3:   # __ne__()
            return self.x != p.x or self.y != p.y or self.z != p.z
        else:
            return NotImplemented

    def __getitem__(self, int i):
        """Returns the point coordinates by index ([0,1,2] -> [x,y,z])."""

        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        elif i == 2:
            return self.z
        else:
            raise IndexError("Index out of range [0, 2].")

    def __setitem__(self, int i, double value):
        """Sets the point coordinates by index ([0,1,2] -> [x,y,z])."""

        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        elif i == 2:
            self.z = value
        else:
            raise IndexError("Index out of range [0, 2].")

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

    @cython.cdivision(True)
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
            w = 1.0 / w

            return new_point((m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3]) * w,
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

    @cython.cdivision(True)
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
        w = 1.0 / w

        return new_point((m.m[0][0] * self.x + m.m[0][1] * self.y + m.m[0][2] * self.z + m.m[0][3]) * w,
                         (m.m[1][0] * self.x + m.m[1][1] * self.y + m.m[1][2] * self.z + m.m[1][3]) * w,
                         (m.m[2][0] * self.x + m.m[2][1] * self.y + m.m[2][2] * self.z + m.m[2][3]) * w)

    cdef inline Point add(self, _Vec3 v):
        """
        Fast addition operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_point(self.x + v.x,
                         self.y + v.y,
                         self.z + v.z)

    cdef inline Point sub(self, _Vec3 v):
        """
        Fast subtraction operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_point(self.x - v.x,
                         self.y - v.y,
                         self.z - v.z)

    cpdef Point copy(self):
        """
        Returns a copy of the point.
        """

        return new_point(self.x,
                         self.y,
                         self.z)

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
        elif index == 2:
            return self.z
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
        elif index == 2:
            self.z = value



cdef class Point2D:
    """
    Represents a point in 2D affine space.

    A 2D point is a location in 2D space which is defined by its u and v coordinates in a given coordinate system.
    Vectors can be added/subtracted from Points yielding another 2D Vector. You can also find the 2D Vector and distance
    between two Points, and transform a Point from one coordinate system to another.
    """

    def __init__(self, double u=0.0, double v=0.0):
        """
        Point2D constructor.

        If no initial values are passed, Point2D defaults to the origin:
        Point2D(0.0, 0.0)
        """

        self.u = u
        self.v = v

    def __repr__(self):
        """Returns a string representation of the Point2D object."""

        return "Point2D([" + str(self.u) + ", " + str(self.v) + "])"

    def __richcmp__(self, object other, int op):
        """Provides basic point comparison operations."""

        raise NotImplemented

        # cdef Point2D p
        #
        # if not isinstance(other, Point2D):
        #     return NotImplemented
        #
        # p = <Vector> other
        # if op == 2:     # __eq__()
        #     return self.x == p.x and self.y == p.y and self.z == p.z
        # elif op == 3:   # __ne__()
        #     return self.x != p.x or self.y != p.y or self.z != p.z
        # else:
        #     return NotImplemented

    def __getitem__(self, int i):
        """Returns the point coordinates by index ([0,1] -> [u,v])."""

        if i == 0:
            return self.u
        elif i == 1:
            return self.v
        else:
            raise IndexError("Index out of range [0, 1].")

    def __setitem__(self, int i, double value):
        """Sets the point coordinates by index ([0,1] -> [u,v])."""

        if i == 0:
            self.u = value
        elif i == 1:
            self.v = value
        else:
            raise IndexError("Index out of range [0, 1].")

    def __add__(object x, object y):
        """Addition operator."""
        raise NotImplemented

        # cdef Point p
        # cdef _Vec3 v
        #
        # if isinstance(x, Point) and isinstance(y, _Vec3):
        #
        #     p = <Point>x
        #     v = <_Vec3>y
        #
        # else:
        #
        #     return NotImplemented
        #
        # return new_point(p.x + v.x,
        #                  p.y + v.y,
        #                  p.z + v.z)

    def __sub__(object x, object y):
        """Subtraction operator."""
        raise NotImplemented

        # cdef Point p
        # cdef _Vec3 v
        #
        # if isinstance(x, Point) and isinstance(y, _Vec3):
        #
        #     p = <Point>x
        #     v = <_Vec3>y
        #
        #     return new_point(p.x - v.x,
        #                      p.y - v.y,
        #                      p.z - v.z)
        #
        # else:
        #
        #     return NotImplemented

    @cython.cdivision(True)
    def __mul__(object x, object y):
        """Multiply operator."""
        raise NotImplemented

        # cdef AffineMatrix m
        # cdef Point v
        # cdef double w
        #
        # if isinstance(x, AffineMatrix) and isinstance(y, Point):
        #
        #     m = <AffineMatrix>x
        #     v = <Point>y
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
        #     return new_point((m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3]) * w,
        #                      (m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3]) * w,
        #                      (m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3]) * w)
        #
        # return NotImplemented

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.u, self.v

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.u = state[0]
        self.v = state[1]

    # cpdef Vector vector_to(self, Point p):
    #     """
    #     Returns a vector from this point to the passed point.
    #     """
    #
    #     return new_vector(p.x - self.x,
    #                       p.y - self.y,
    #                       p.z - self.z)

    cpdef double distance_to(self, Point2D p):
        """
        Returns the distance between this point and the passed point.
        """

        cdef double u, v
        u = p.u - self.u
        v = p.v - self.v
        return sqrt(u*u + v*v)

    # @cython.cdivision(True)
    # cpdef Point transform(self, AffineMatrix m):
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
    #     return new_point((m.m[0][0] * self.x + m.m[0][1] * self.y + m.m[0][2] * self.z + m.m[0][3]) * w,
    #                      (m.m[1][0] * self.x + m.m[1][1] * self.y + m.m[1][2] * self.z + m.m[1][3]) * w,
    #                      (m.m[2][0] * self.x + m.m[2][1] * self.y + m.m[2][2] * self.z + m.m[2][3]) * w)

    # cdef inline Point add(self, _Vec3 v):
    #     """
    #     Fast addition operator.
    #
    #     This is a cython only function and is substantially faster than a call
    #     to the equivalent python operator.
    #     """
    #
    #     return new_point(self.x + v.x,
    #                      self.y + v.y,
    #                      self.z + v.z)

    # cdef inline Point sub(self, _Vec3 v):
    #     """
    #     Fast subtraction operator.
    #
    #     This is a cython only function and is substantially faster than a call
    #     to the equivalent python operator.
    #     """
    #
    #     return new_point(self.x - v.x,
    #                      self.y - v.y,
    #                      self.z - v.z)

    cpdef Point2D copy(self):
        """
        Returns a copy of the point.
        """
        return new_point2d(self.u, self.v)

    cdef inline double get_index(self, int index):
        """
        Fast getting of coordinates via indexing.

        Cython equivalent to __getitem__, without the checks and call overhead.

        If an invalid index is passed this function return NaN.
        """
        if index == 0:
            return self.u
        elif index == 1:
            return self.v
        else:
            return float("NaN")

    cdef inline void set_index(self, int index, double value):
        """
        Fast setting of coordinates via indexing.

        Cython equivalent to __setitem__, without the checks and call overhead.

        If an invalid index is passed this function does nothing.
        """
        if index == 0:
            self.u = value
        elif index == 1:
            self.v = value
