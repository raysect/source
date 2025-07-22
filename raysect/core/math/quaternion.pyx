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

import numbers
cimport cython
from libc.math cimport sqrt, sin, cos, asin, acos, atan2, fabs, M_PI, copysign

from raysect.core.math.vector cimport new_vector3d
from raysect.core.math.affinematrix cimport new_affinematrix3d, AffineMatrix3D

DEF RAD2DEG = 57.29577951308232000  # 180 / pi
DEF DEG2RAD = 0.017453292519943295  # pi / 180


cdef class Quaternion:

    def __init__(self, double x=0.0, double y=0.0, double z=0.0, double s=1.0):

        self.x = x
        self.y = y
        self.z = z
        self.s = s

    def __repr__(self):
        """Returns a string representation of the Quaternion object."""

        return f'Quaternion({self.x}, {self.y}, {self.z}, {self.s})'

    def __getitem__(self, int i):
        """Returns the quaternion coordinates by index ([0,1,2,3] -> [x,y,z,s]).

        .. code-block:: pycon

            >>> a = Quaternion(0, 0, 0, 1)
            >>> a[3]
            1
        """

        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        elif i == 2:
            return self.z
        elif i == 3:
            return self.s
        else:
            raise IndexError("Index out of range [0, 3].")

    def __setitem__(self, int i, double value):
        """Sets the quaternion coordinates by index ([0,1,2,3] -> [x,y,z,s]).

        .. code-block:: pycon

            >>> a = Quaternion(0, 0, 0, 1)
            >>> a[1] = 2
            >>> a
            Quaternion(0.0, 2.0, 0.0, 1.0)
        """

        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        elif i == 2:
            self.z = value
        elif i == 3:
            self.s = value
        else:
            raise IndexError('Index out of range [0, 3].')

    def __iter__(self):
        """Iterates over the quaternion coordinates (x, y, z, s)

            >>> a = Quaternion(0, 1, 2, 3)
            >>> x, y, z, s = a
            >>> x, y, z, s
            (0.0, 1.0, 2.0, 3.0)
        """
        yield self.x
        yield self.y
        yield self.z
        yield self.s

    def __neg__(self):
        """
        Returns a Quaternion with the reverse orientation (negation operator).

        Note however that (s + x i + y j + z k) and (- s - x i - y j - z k)
        represent the same rotations. Even though negation generates a different
        quaternion it represents the same overall rotation.

        .. code-block:: pycon

            >>> a = Quaternion(0, 0, 0, 1)
            >>> -a
            Quaternion(-0.0, -0.0, -0.0, -1.0)
        """

        return new_quaternion(-self.x, -self.y, -self.z, -self.s)

    def __eq__(object x, object y):
        """
        Equality operator.

        .. code-block:: pycon

            >>> Quaternion(0, 0, 0, 1) == Quaternion(0, 0, 0, 1)
            True
        """

        cdef Quaternion q1, q2

        if isinstance(x, Quaternion) and isinstance(y, Quaternion):

            q1 = <Quaternion> x
            q2 = <Quaternion> y
            return q1.x == q2.x and q1.y == q2.y and q1.z == q2.z and q1.s == q2.s

        else:
            raise TypeError('A quaternion can only be equality tested against another quaternion.')

    def __add__(object x, object y):
        """
        Addition operator.

        .. code-block:: pycon

            >>> Quaternion(0, 0, 0, 1) + Quaternion(0, 1, 0, 0)
            Quaternion(0.0, 1.0, 0.0, 1.0)
        """

        cdef Quaternion q1, q2

        if isinstance(x, Quaternion) and isinstance(y, Quaternion):

            q1 = <Quaternion> x
            q2 = <Quaternion> y
            return new_quaternion(q1.x + q2.x, q1.y + q2.y, q1.z + q2.z, q1.s + q2.s)

        else:
            return NotImplemented

    def __sub__(object x, object y):
        """Subtraction operator.

        .. code-block:: pycon

            >>> Quaternion(0, 0, 0, 1) - Quaternion(0, 1, 0, 0)
            Quaternion(0.0, -1.0, 0, 1.0)
        """

        cdef Quaternion q1, q2

        if isinstance(x, Quaternion) and isinstance(y, Quaternion):

            q1 = <Quaternion> x
            q2 = <Quaternion> y
            return new_quaternion(q1.x - q2.x, q1.y - q2.y, q1.z - q2.z, q1.s - q2.s)

        else:
            return NotImplemented

    def __mul__(object x, object y):
        """Multiplication operator.

        .. code-block:: pycon

            >>> Quaternion(0, 0, 1, 1) * 2
            Quaternion(0.0, 0.0, 2.0, 2.0)
            >>> Quaternion(0, 1, 0, 1) * Quaternion(1, 2, 3, 0)
            Quaternion(4.0, 2.0, 2.0, -2.0)
        """

        cdef double s
        cdef Quaternion q1, q2

        if isinstance(x, numbers.Real) and isinstance(y, Quaternion):

            s = <double> x
            q1 = <Quaternion> y
            return q1.mul_scalar(s)

        elif isinstance(x, Quaternion) and isinstance(y, numbers.Real):

            q1 = <Quaternion> x
            s = <double> y
            return q1.mul_scalar(s)

        elif isinstance(x, Quaternion) and isinstance(y, Quaternion):

            q1 = <Quaternion> x
            q2 = <Quaternion> y
            return q1.mul_quaternion(q2)

        else:
            return NotImplemented()

    @cython.cdivision(True)
    def __truediv__(object x, object y):
        """Division operator.

        .. code-block:: pycon

            >>> Quaternion(0, 0, 1, 1) / 2
            Quaternion(0.0, 0.0, 0.5, 0.0.5)
            >>> Quaternion(0, 0, 1, 1) / Quaternion(1, 2, 3, 0)
            Quaternion(-0.28571, -0.14286, -0.14286, 0.14286)
        """

        cdef double d
        cdef Quaternion q1, q2, q2_inv

        if isinstance(x, Quaternion) and isinstance(y, numbers.Real):

            d = <double> y
            q1 = <Quaternion> x
            return q1.div_scalar(d)

        elif isinstance(x, Quaternion) and isinstance(y, Quaternion):

            q1 = <Quaternion> x
            q2 = <Quaternion> y
            return q1.div_quaternion(q2)

        else:
            raise TypeError('Unsupported operand type. Expects a real number.')

    @property
    def length(self):
        """
        Calculates the length (norm) of the quaternion.

        .. code-block:: pycon

            >>> Quaternion(1, 2, 3, 0).length
            3.7416573867739413
        """
        return self.get_length()

    @length.setter
    def length(self, value):
        self.set_length(value)


    @property
    def axis(self):
        """
        The axis around which this quaternion rotates.
        """
        return self.get_axis()

    @property
    def angle(self):
        """The magnitude of rotation around this quaternion's rotation axis in degrees."""
        return self.get_angle()

    cpdef Quaternion copy(self):
        """Returns a copy of this quaternion."""

        return new_quaternion(self.x, self.y, self.z, self.s)

    cpdef Quaternion conjugate(self):
        """
        Complex conjugate operator. 
        
        .. code-block:: pycon

            >>> Quaternion(1, 2, 3, 0).conjugate()
            Quaternion(-1, -2, -3, 0)
        """

        return new_quaternion(-self.x, -self.y, -self.z, self.s)

    @cython.cdivision(True)
    cpdef Quaternion inverse(self):
        """
        Inverse operator.

        .. code-block:: pycon

            >>> Quaternion(1, 2, 3, 0).inverse()
            Quaternion(-0.07143, -0.14286, -0.21429, 0.0)
        """

        cdef double n = self.get_length()**2
        return new_quaternion(-self.x/n, -self.y/n, -self.z/n, self.s/n)

    @cython.cdivision(True)
    cpdef Quaternion normalise(self):
        """
        Returns a normalised copy of the quaternion.

        The returned quaternion is normalised to have norm length 1.0 - a unit quaternion.

        .. code-block:: pycon
        
            >>> a = Quaternion(1, 2, 3, 0)
            >>> a.normalise()
            Quaternion(0.26726, 0.53452, 0.80178, 0.0)
        """

        cdef double n

        # if current length is zero, problem is ill defined
        n = self.get_length()
        if n == 0.0:
            raise ZeroDivisionError('A zero length quaternion cannot be normalised.')

        # normalise and rescale quaternion
        n = 1.0 / n
        return self.mul_scalar(n)

    cpdef bint is_unit(self, double tolerance=1e-10):
        """
        Returns True if this is a unit quaternion (versor) to within specified tolerance.

        :param float tolerance: The numerical tolerance by which the quaternion norm can differ by 1.0.
        """
        return fabs(1.0 - self.get_length()) <= tolerance

    cpdef Quaternion transform(self, AffineMatrix3D m):
        """
        Transforms the quaternion with the supplied AffineMatrix3D.

        :param AffineMatrix3D m: The affine matrix describing the required coordinate transformation.
        :return: A new instance of this quaternion that has been transformed with the supplied Affine Matrix.
        :rtype: Quaternion
        """

        cdef Quaternion q = Quaternion.from_axis_angle(self.axis.transform(m), self.angle)
        q.set_length(self.get_length())
        return q

    cpdef AffineMatrix3D as_matrix(self):
        """
        Generates an AffineMatrix3D representation of this Quaternion.

        .. code-block:: pycon

           >>> from raysect.core.math import Quaternion
           >>>
           >>> q = Quaternion(0.5, 0, 0, 0.5)
           >>> q.as_matrix()
           AffineMatrix3D([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        """

        cdef:
            double qs, qx, qy, qz
            double qs2, qx2, qy2, qz2
            double m00, m01, m02
            double m10, m11, m12
            double m20, m21, m22
            double temp1, temp2
            Quaternion unit_q

        unit_q = self.normalise()

        qx = unit_q.x
        qy = unit_q.y
        qz = unit_q.z
        qs = unit_q.s

        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz
        qs2 = qs * qs

        m00 = 1 - 2*qy2 - 2*qz2
        m01 = 2*qx*qy - 2*qz*qs
        m02 = 2*qx*qz + 2*qy*qs

        m10 = 2*qx*qy + 2*qz*qs
        m11 = 1 - 2*qx2 - 2*qz2
        m12 = 2*qy*qz - 2*qx*qs

        m20 = 2*qx*qz - 2*qy*qs
        m21 = 2*qy*qz + 2*qx*qs
        m22 = 1 - 2*qx2 - 2*qy2

        return new_affinematrix3d(m00, m01, m02, 0,
                                  m10, m11, m12, 0,
                                  m20, m21, m22, 0,
                                  0, 0, 0, 1)

    cpdef Quaternion quaternion_to(self, Quaternion q):
        """
        Calculates the quaternion between quaternions.

        This method calculates the quaternion required to map this quaternion
        onto the supplied quaternion. Both quaternions will be normalised and
        a normalised quaternion will be returned.
        
        .. code-block:: pycon
        
          >>> from raysect.core.math import Quaternion
          >>>
          >>> q1 = Quaternion.from_axis_angle(Vector3D(1,0,0), 10) 
          >>> q2 = Quaternion.from_axis_angle(Vector3D(1,0,0), 25)
          >>> d = q1.quaternion_to(q2)
          >>> d
          Quaternion(0.13052619222005157, 0.0, 0.0, 0.9914448613738104)
          >>> d.angle
          15.000000000000027
          >>> d.axis
          Vector3D(1.0, 0.0, 0.0)
        
        :param Quaternion q: The target quaternion.
        :return: A new Quaternion object representing the specified rotation.        
        """

        return q.normalise().mul_quaternion(self.normalise().conjugate()).normalise()

    @classmethod
    def from_matrix(cls, AffineMatrix3D matrix):
        """
        Extract the rotation part of an AffineMatrix3D as a Quaternion.

        Note, the translation component of this matrix will be ignored.

        :param AffineMatrix3D matrix: The AffineMatrix3D instance from which to extract the rotation component.
        :return: A quaternion representation of the rotation specified in this transform matrix.

        .. code-block:: pycon

           >>> from raysect.core.math import rotate_x, Quaternion
           >>>
           >>> Quaternion.from_matrix(rotate_x(90))
           Quaternion(0.7071067811865475, 0.0, 0.0, 0.7071067811865476)
        """

        return new_quaternion_from_matrix(matrix)

    @classmethod
    def from_axis_angle(cls, Vector3D axis, double angle):
        """
        Generates a new Quaternion from the axis-angle specification.

        :param Vector3D axis: The axis about which rotation will be performed.
        :param float angle: An angle in degrees specifying the magnitude of the
          rotation about the axis vector.
        :return: A new Quaternion object representing the specified rotation.

        .. code-block:: pycon

           >>> from raysect.core.math import Quaternion, Vector3D
           >>>
           >>> Quaternion.from_axis_angle(Vector3D(1, 0, 0), 45)
           Quaternion(0.3826834323650898, 0.0, 0.0, 0.9238795325112867)
        """

        return new_quaternion_from_axis_angle(axis, angle)

    cdef Quaternion neg(self):
        """
        Fast negation operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_quaternion(-self.x, -self.y, -self.z, -self.s)

    cdef Quaternion add(self, Quaternion q):
        """
        Fast addition operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_quaternion(self.x + q.x, self.y + q.y, self.z + q.z, self.s + q.s)

    cdef Quaternion sub(self, Quaternion q):
        """
        Fast subtraction operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_quaternion(self.x - q.x, self.y - q.y, self.z - q.z, self.s - q.s)

    cdef Quaternion mul_quaternion(self, Quaternion q):
        """
        Fast multiplication operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        cdef double ns, nx, ny, nz

        nx = self.s*q.x + self.x*q.s + self.y*q.z - self.z*q.y
        ny = self.s*q.y - self.x*q.z + self.y*q.s + self.z*q.x
        nz = self.s*q.z + self.x*q.y - self.y*q.x + self.z*q.s
        ns = self.s*q.s - self.x*q.x - self.y*q.y - self.z*q.z

        return new_quaternion(nx, ny, nz, ns)

    cdef Quaternion mul_scalar(self, double d):
        return new_quaternion(d * self.x, d * self.y, d * self.z, d * self.s)

    cdef Quaternion div_quaternion(self, Quaternion q):
        """
        Fast division operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return self.mul_quaternion(q.inverse())

    @cython.cdivision(True)
    cdef Quaternion div_scalar(self, double d):

        # prevent divide by zero
        if d == 0.0:
            raise ZeroDivisionError('Cannot divide a quaternion by a zero scalar.')

        d = 1.0 / d
        return new_quaternion(d * self.x, d * self.y, d * self.z, d * self.s)

    cdef Vector3D get_axis(self):

        # quaternion vector component is scaled by a constant value, so simply re-normalise to obtain a unit vector
        cdef Vector3D v = new_vector3d(self.x, self.y, self.z)

        # a null (zero rotation) quaternion returns a null vector
        if v.get_length() == 0:
            return v

        return v.normalise()

    @cython.cdivision(True)
    cdef double get_angle(self):
        """The magnitude of rotation around this quaternion's rotation axis in degrees."""

        cdef Quaternion q = self.normalise()
        return 2 * acos(q.s) * RAD2DEG

    cdef double get_length(self) nogil:
        """
        Fast function to obtain the quaternion length (norm).

        Cython only, equivalent to length.__get__() property.

        Use instead of Python attribute access in cython code.
        """
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.s * self.s)

    @cython.cdivision(True)
    cdef object set_length(self, double v):
        """
        Fast function to set the quaternions length.

        Cython only, equivalent to length.__set__() property.

        Use instead of Python attribute access in cython code.
        """

        cdef double t

        # if current length is zero, problem is ill defined
        t = self.x * self.x + self.y * self.y + self.z * self.z + self.s * self.s
        if t == 0.0:
            raise ZeroDivisionError('A zero length quaternion cannot be rescaled.')

        # normalise and rescale quaternion
        t = v / sqrt(t)
        self.x = self.x * t
        self.y = self.y * t
        self.z = self.z * t
        self.s = self.s * t


cdef Quaternion new_quaternion_from_matrix(AffineMatrix3D matrix):
    """
    Quaternion factory function to instance a Quaternion from an AffineMatrix3D.

    Note, the translation component of this matrix will be ignored.

    Creates a new Quaternion object with less overhead than the equivalent Python
    call. This function is callable from cython only.

    :param AffineMatrix3D matrix: The AffineMatrix3D instance from which to extract the rotation component.
    :return: A quaternion representation of the rotation specified in this transform matrix.
    """

    cdef:
        AffineMatrix3D m = matrix
        double qs, qx, qy, qz
        double trace, s

    trace = m.m[0][0] + m.m[1][1] + m.m[2][2]

    if trace > 0:

        s = sqrt(trace+1.0) * 2  # s = 4*qs
        qx = (m.m[2][1] - m.m[1][2]) / s
        qy = (m.m[0][2] - m.m[2][0]) / s
        qz = (m.m[1][0] - m.m[0][1]) / s
        qs = 0.25 * s

    elif m.m[0][0] > m.m[1][1] and m.m[0][0] > m.m[2][2]:

        s = sqrt(1.0 + m.m[0][0] - m.m[1][1] - m.m[2][2]) * 2  # s = 4*qx
        qx = 0.25 * s
        qy = (m.m[0][1] + m.m[1][0]) / s
        qz = (m.m[0][2] + m.m[2][0]) / s
        qs = (m.m[2][1] - m.m[1][2]) / s

    elif m.m[1][1] > m.m[2][2]:

        s = sqrt(1.0 + m.m[1][1] - m.m[0][0] - m.m[2][2]) * 2  # s = 4*qy
        qx = (m.m[0][1] + m.m[1][0]) / s
        qy = 0.25 * s
        qz = (m.m[1][2] + m.m[2][1]) / s
        qs = (m.m[0][2] - m.m[2][0]) / s

    else:

        s = sqrt(1.0 + m.m[2][2] - m.m[0][0] - m.m[1][1]) * 2  # s = 4*qz
        qx = (m.m[0][2] + m.m[2][0]) / s
        qy = (m.m[1][2] + m.m[2][1]) / s
        qz = 0.25 * s
        qs = (m.m[1][0] - m.m[0][1]) / s

    return new_quaternion(qx, qy, qz, qs)


cdef Quaternion new_quaternion_from_axis_angle(Vector3D axis, double angle):
    """
    Quaternion factory function to instance a Quaternion from an axis and angle.

    Creates a new Quaternion object with less overhead than the equivalent Python
    call. This function is callable from cython only.

    :param Vector3D axis: The axis about which rotation will be performed.
    :param float angle: An angle in degrees specifying the magnitude of the
      rotation about the axis vector.
    :return: A new Quaternion object representing the specified rotation.
    """

    # keep the angle inside [-180, 180) to avoid negative quaternions
    angle = (angle + 180) % 360 - 180
    if angle == 0:
        return new_quaternion(0, 0, 0, 1)

    axis = axis.normalise()
    theta_2 = angle * DEG2RAD / 2

    qx = axis.x * sin(theta_2)
    qy = axis.y * sin(theta_2)
    qz = axis.z * sin(theta_2)
    qs = cos(theta_2)

    return new_quaternion(qx, qy, qz, qs)