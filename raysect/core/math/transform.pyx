# cython: language_level=3

# Copyright (c) 2014-2016, Dr Alex Meakins, Raysect Project
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

from libc.math cimport sin, cos, M_PI as pi
from raysect.core.math.affinematrix cimport new_affinematrix3d
cimport cython


cpdef AffineMatrix3D translate(double x, double y, double z):
    """
    Returns an affine matrix representing a translation of the coordinate space.

    Equivalent to the transform matrix, :math:`\\mathbf{T_{AB}}`, where :math:`\\vec{t}`
    is the vector from the origin of space A to space B.

    .. math::

        \\mathbf{T_{AB}} = \\left( \\begin{array}{cccc} 1 & 0 & 0 & \\vec{t}.x \\\\
        0 & 1 & 0 & \\vec{t}.y \\\\
        0 & 0 & 1 & \\vec{t}.z \\\\
        0 & 0 & 0 & 1 \\end{array} \\right)

    :param float x: x-coordinate
    :param float y: y-coordinate
    :param float z: z-coordinate
    :rtype: AffineMatrix3D
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

    The rotation direction is clockwise when looking along the x-axis.

    .. math::

        \\mathbf{T_{AB}} = \\left( \\begin{array}{cccc} 1 & 0 & 0 & 0 \\\\
        0 & \\cos{\\theta} & -\\sin{\\theta} & 0 \\\\
        0 & \\sin{\\theta} & \\cos{\\theta} & 0 \\\\
        0 & 0 & 0 & 1 \\end{array} \\right)

    :param float angle: The angle :math:`\\theta` specified in degrees.
    :rtype: AffineMatrix3D
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

    The rotation direction is clockwise when looking along the y-axis.

    .. math::

        \\mathbf{T_{AB}} = \\left( \\begin{array}{cccc} \\cos{\\theta} & 0 & \\sin{\\theta} & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        -\\sin{\\theta} & 0 & \\cos{\\theta} & 0 \\\\
        0 & 0 & 0 & 1 \\end{array} \\right)

    :param float angle: The angle :math:`\\theta` specified in degrees.
    :rtype: AffineMatrix3D
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

    The rotation direction is clockwise when looking along the z-axis.

    .. math::

        \\mathbf{T_{AB}} = \\left( \\begin{array}{cccc} \\cos{\\theta} & -\\sin{\\theta} & 0 & 0 \\\\
        \\sin{\\theta} & \\cos{\\theta} & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & 1 \\end{array} \\right)

    :param float angle: The angle :math:`\\theta` specified in degrees.
    :rtype: AffineMatrix3D
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

    :param float angle: The angle specified in degrees.
    :param Vector3D v: The vector about which to rotate.
    :rtype: AffineMatrix3D
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

    :param float yaw: Yaw angle in degrees.
    :param float pitch: Pitch angle in degrees.
    :param float roll: Roll angle in degrees.
    :rtype: AffineMatrix3D
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

    :param Vector3D forward: A Vector3D object defining the forward direction.
    :param Vector3D up: A Vector3D object defining the up direction.
    :rtype: AffineMatrix3D
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
