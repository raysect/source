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
from raysect.core.math.vector cimport Vector3D
from libc.math cimport fabs, sin, cos, M_PI as pi

cdef class AffineMatrix3D(_Mat4):
    """Represents a 4x4 affine matrix."""

    def __repr__(self):
        """String representation."""

        cdef int i, j

        s = "AffineMatrix3D(["
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

        cdef AffineMatrix3D mx, my

        if isinstance(x, AffineMatrix3D) and isinstance(y, AffineMatrix3D):

            mx = <AffineMatrix3D>x
            my = <AffineMatrix3D>y
            return new_affinematrix3d(mx.m[0][0] * my.m[0][0] + mx.m[0][1] * my.m[1][0] + mx.m[0][2] * my.m[2][0] + mx.m[0][3] * my.m[3][0],
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

    @cython.cdivision(True)
    cpdef AffineMatrix3D inverse(self):
        """
        Calculates the inverse of the affine matrix.

        Returns an AffineMatrix3D containing the inverse.

        Raises a ValueError if the matrix is singular and the inverse can not be
        calculated. All valid affine transforms should be invertable.
        """

        cdef:
            double t[22]
            double det, idet

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

        return new_affinematrix3d((self.m[2][2] * t[16] - self.m[2][1] * t[17] - self.m[2][3] * t[15]) * idet,
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

    cdef inline AffineMatrix3D mul(self, AffineMatrix3D m):

        return new_affinematrix3d(self.m[0][0] * m.m[0][0] + self.m[0][1] * m.m[1][0] + self.m[0][2] * m.m[2][0] + self.m[0][3] * m.m[3][0],
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


cpdef AffineMatrix3D translate(double x, double y, double z):
    """
    Returns an affine matrix representing a translation of the coordinate space.
    """

    return new_affinematrix3d(1, 0, 0, x,
                              0, 1, 0, y,
                              0, 0, 1, z,
                              0, 0, 0, 1)


@cython.cdivision(True)
cpdef AffineMatrix3D rotate_x(double angle):
    """
    Returns an affine matrix representing the rotation of the coordinate space
    about the X axis by the supplied angle.

    The angle is specified in degrees.
    """

    cdef double r

    r = pi * angle / 180.0
    return new_affinematrix3d(1, 0, 0, 0,
                              0, cos(r), -sin(r), 0,
                              0, sin(r), cos(r), 0,
                              0, 0, 0, 1)


@cython.cdivision(True)
cpdef AffineMatrix3D rotate_y(double angle):
    """
    Returns an affine matrix representing the rotation of the coordinate space
    about the Y axis by the supplied angle.

    The angle is specified in degrees.
    """

    cdef double r

    r = pi * angle / 180.0
    return new_affinematrix3d(cos(r), 0, sin(r), 0,
                              0, 1, 0, 0,
                              -sin(r), 0, cos(r), 0,
                              0, 0, 0, 1)


@cython.cdivision(True)
cpdef AffineMatrix3D rotate_z(double angle):
    """
    Returns an affine matrix representing the rotation of the coordinate space
    about the Z axis by the supplied angle.

    The angle is specified in degrees.
    """

    cdef double r

    r = pi * angle / 180.0
    return new_affinematrix3d(cos(r), -sin(r), 0, 0,
                              sin(r), cos(r), 0, 0,
                              0, 0, 1, 0,
                              0, 0, 0, 1)


@cython.cdivision(True)
cpdef AffineMatrix3D rotate_vector(double angle, Vector3D v):
    """
    Returns an affine matrix representing the rotation of the coordinate space
    about the supplied vector by the specified angle.

    The angle is specified in degrees.
    """

    cdef Vector3D vn
    cdef double r, s, c, ci

    vn = v.normalise()
    r = pi * angle / 180.0
    s = sin(r)
    c = cos(r)
    ci = 1.0 - c
    return new_affinematrix3d(vn.x * vn.x + (1.0 - vn.x * vn.x) * c,
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


cpdef AffineMatrix3D rotate(double yaw, double pitch, double roll):
    """
    Returns an affine transform matrix representing an intrinsic rotation with
    an axis order (-Y)(-X)'Z''.

    For an object aligned such that forward is the +ve Z-axis, left is the +ve
    X-axis and up is the +ve Y-axis then this rotation operation corresponds to
    the yaw, pitch and roll of the object.

    :param yaw: Yaw angle in degrees.
    :param pitch: Pitch angle in degrees.
    :param roll: Roll angle in degrees.
    :return: An AffineMatrix3D object.
    """

    return rotate_y(-yaw) * rotate_x(-pitch) * rotate_z(roll)


cpdef AffineMatrix3D rotate_basis(Vector3D forward, Vector3D up):
    """
    Returns a rotation matrix defined by forward and up vectors.

    The +ve Z-axis of the resulting coordinate space will be aligned with the
    forward vector. The +ve Y-axis will be aligned to lie in the plane defined
    the forward and up vectors, along the projection of the up vector that
    lies orthogonal to the forward vector. The X-axis will lie perpendicular to
    the plane.

    The forward and upwards vectors need not be orthogonal. The up vector will
    be rotated in the plane defined by the two vectors until it is orthogonal.

    :param forward: A Vector3D object defining the forward direction.
    :param up: A Vector3D object defining the up direction.
    :return: An AffineMatrix3D object.
    """

    cdef Vector3D x, y, z

    if forward is None:
        raise ValueError("Forward vector must not be None.")

    if up is None:
        raise ValueError("Up vector must not be None.")

    z = forward.normalise()
    y = up.normalise()

    # forward and up vectors must not be the same!
    if y == z:
        raise ValueError("Forward and up vectors must not be coincident.")

    # ensure y vector is perpendicular to z
    y = y - y.dot(z) * z
    y = y.normalise()

    # generate remaining basis vector
    x = y.cross(z)

    return new_affinematrix3d(x.x, y.x, z.x, 0.0,
                              x.y, y.y, z.y, 0.0,
                              x.z, y.z, z.z, 0.0,
                              0.0, 0.0, 0.0, 1.0)


