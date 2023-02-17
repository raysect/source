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

from libc.math cimport sin, cos, sqrt, asin, atan2, M_PI as pi
from raysect.core.math.affinematrix cimport new_affinematrix3d
from raysect.core.math.point cimport new_point3d
cimport cython


DEF RAD2DEG = 57.29577951308232000  # 180 / pi
DEF DEG2RAD = 0.017453292519943295  # pi / 180


FORWARD = 'forward'
UP = 'up'


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

    .. code-block:: pycon

        >>> from raysect.core import translate
        >>> translate(0, 1, 2)
        AffineMatrix3D([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 2.0],
                        [0.0, 0.0, 0.0, 1.0]])
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

    .. code-block:: pycon

        >>> from raysect.core import rotate_x
        >>> rotate_x(45)
        AffineMatrix3D([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.7071067811865476, -0.7071067811865475, 0.0],
                        [0.0, 0.7071067811865475, 0.7071067811865476, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
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

    .. code-block:: pycon

        >>> from raysect.core import rotate_vector
        >>> rotate_vector(90, Vector3D(1, 0, 0))
        AffineMatrix3D([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
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

    .. code-block:: pycon

        >>> from raysect.core import rotate_basis, Vector3D
        >>> rotate_basis(Vector3D(1, 0, 0), Vector3D(0, 0, 1))
        AffineMatrix3D([[0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
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


cpdef tuple to_cylindrical(Point3D point):
    """
    Convert the given 3D point in cartesian space to cylindrical coordinates. 
    
    :param Point3D point: The 3D point to be transformed into cylindrical coordinates.
    :rtype: tuple
    :return: A tuple of r, z, phi coordinates.
    
    .. code-block:: pycon
    
        >>> from raysect.core.math import to_cylindrical, Point3D
        
        >>> point = Point3D(1, 1, 1)
        >>> to_cylindrical(point)
        (1.4142135623730951, 1.0, 45.0)
    """

    cdef double r, phi

    r = sqrt(point.x*point.x + point.y*point.y)
    phi = atan2(point.y, point.x) * RAD2DEG

    return r, point.z, phi


cpdef Point3D from_cylindrical(double r, double z, double phi):
    """
    Convert a 3D point in cylindrical coordinates to a point in cartesian coordinates.
    
    :param float r: The radial coordinate.
    :param float z: The z-axis height coordinate.
    :param float phi: The azimuthal coordinate in degrees.
    :rtype: Point3D
    :return: A Point3D in cartesian space.
    
    .. code-block:: pycon
    
        >>> from raysect.core.math import from_cylindrical

        >>> from_cylindrical(1, 0, 45)
        Point3D(0.7071067811865476, 0.7071067811865475, 0.0)
    """


    cdef double x, y

    if r < 0:
        raise ValueError("R coordinate cannot be less than 0.")

    x = r * cos(phi * DEG2RAD)
    y = r * sin(phi * DEG2RAD)

    return new_point3d(x, y, z)


cpdef (double, double, double) extract_rotation(AffineMatrix3D m, bint z_up=False):
    """
    Extracts the rotation component of the affine matrix.
    
    The yaw, pitch and roll can be extracted for two common coordinate
    conventions by specifying the z_axis orientation:
    
        forward: +ve z is forward, +ve y is up, +ve x is left  
        up:      +ve z is up, +ve y is left, +ve x is forward

    The Raysect default is z axis forward. This function can be switched
    to z axis up by setting the z_up parameter to True. 
    
    The matrix must consist of only rotation and translation operations.
    
    :param AffineMatrix3D m: An affine matrix.
    :param bint z_up: Is the z-axis pointed upwards (default=False). 
    :return: A tuple containing (yaw, pitch, roll). 
    """

    cdef double yaw, pitch, roll

    if z_up:

        # operation order XYZ
        yaw = -atan2(m.get_element(1, 0), m.get_element(0, 0)) * RAD2DEG
        pitch = asin(m.get_element(2, 0)) * RAD2DEG
        roll = atan2(m.get_element(2, 1), m.get_element(2, 2)) * RAD2DEG
        return yaw, pitch, roll

    # operation order ZYX
    yaw = -atan2(m.get_element(0, 2), m.get_element(2, 2)) * RAD2DEG
    pitch = asin(m.get_element(1, 2)) * RAD2DEG
    roll = atan2(m.get_element(1, 0), m.get_element(1, 1)) * RAD2DEG
    return yaw, pitch, roll


cpdef (double, double, double) extract_translation(AffineMatrix3D m):
    """
    Extracts the translation component of the affine matrix.
    
    The matrix must consist of only rotation and translation operations.

    :param AffineMatrix3D m: An affine matrix.    
    :return: tuple containing the x, y and z components of the translation. 
    """

    return m.get_element(0, 3), m.get_element(1, 3), m.get_element(2, 3)
