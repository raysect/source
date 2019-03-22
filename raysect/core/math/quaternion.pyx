
# Copyright (c) 2014-2018, Dr Alex Meakins, Raysect Project
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
from libc.math cimport sqrt, sin, cos

from raysect.core.math.affinematrix cimport new_affinematrix3d
from raysect.core.math.vector cimport new_vector3d


DEF RAD2DEG = 57.29577951308232000  # 180 / pi
DEF DEG2RAD = 0.017453292519943295  # pi / 180


cdef class Quaternion:

    def __init__(self, double s=1.0, double x=0.0, double y=0.0, double z=0.0):

        self.s = s
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        """Returns a string representation of the Quaternion object."""

        return "Quaternion(" + str(self.s) + ", " + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

    def __getitem__(self, int i):
        """Returns the quaternion coordinates by index ([0,1,2,3] -> [s,x,y,z]).

        .. code-block:: pycon

            >>> a = Quaternion(1, 0, 0, 0)
            >>> a[0]
            1
        """

        if i == 0:
            return self.s
        elif i == 1:
            return self.x
        elif i == 2:
            return self.y
        elif i == 3:
            return self.z
        else:
            raise IndexError("Index out of range [0, 3].")

    def __setitem__(self, int i, double value):
        """Sets the quaternion coordinates by index ([0,1,2,3] -> [s,x,y,z]).

        .. code-block:: pycon

            >>> a = Quaternion(1, 0, 0, 0)
            >>> a[1] = 2
            >>> a
            Quaternion(1.0, 2.0, 0.0, 0.0)
        """

        if i == 0:
            self.s = value
        elif i == 1:
            self.x = value
        elif i == 2:
            self.y = value
        elif i == 3:
            self.z = value
        else:
            raise IndexError("Index out of range [0, 3].")

    def __iter__(self):
        """Iterates over the quaternion coordinates (s, x, y, z)

            >>> a = Quaternion(0, 1, 2, 3)
            >>> s, x, y, z = a
            >>> s, x, y, z
            (0.0, 1.0, 2.0, 3.0)
        """
        yield self.s
        yield self.x
        yield self.y
        yield self.z

    def __neg__(self):
        """
        Returns a Quaternion with the reverse orientation (negation operator).

        Note however that (s + x i + y j + z k) and (- s - x i - y j - z k)
        represent the same rotations. Even though negation generates a different
        quaternion it represents the same overall rotation.

        .. code-block:: pycon

            >>> a = Quaternion(1, 0, 0, 0)
            >>> -a
            Quaternion(-1.0, -0.0, -0.0, -0.0)
        """

        return new_quaternion(-self.s, -self.x, -self.y, -self.z)

    def __eq__(object x, object y):
        """
        Equality operator.

        .. code-block:: pycon

            >>> Quaternion(1, 0, 0, 0) == Quaternion(1, 0, 0, 0)
            True
        """

        cdef Quaternion q1, q2

        if isinstance(x, Quaternion) and isinstance(y, Quaternion):

            q1 = <Quaternion> x
            q2 = <Quaternion> y

            if q1.s == q2.s and q1.x == q2.x and q1.y == q2.y and q1.z == q2.z:
                return True

            else:
                return False

        else:

            raise TypeError("A quaternion can only be equality tested against another quaternion.")

    def __add__(object x, object y):
        """
        Addition operator.

        .. code-block:: pycon

            >>> Quaternion(1, 0, 0, 0) + Quaternion(0, 1, 0, 0)
            Quaternion(1.0, 1.0, 0.0, 0.0)
        """

        cdef Quaternion q1, q2

        if isinstance(x, Quaternion) and isinstance(y, Quaternion):

            q1 = <Quaternion> x
            q2 = <Quaternion> y

            return new_quaternion(q1.s + q2.s, q1.x + q2.x, q1.y + q2.y, q1.z + q2.z)

        else:

            return NotImplemented

    def __sub__(object x, object y):
        """Subtraction operator.

        .. code-block:: pycon

            >>> Quaternion(1, 0, 0, 0) - Quaternion(0, 1, 0, 0)
            Quaternion(1.0, -1.0, 0, 0)
        """

        cdef Quaternion q1, q2

        if isinstance(x, Quaternion) and isinstance(y, Quaternion):

            q1 = <Quaternion> x
            q2 = <Quaternion> y

            return new_quaternion(q1.s - q2.s, q1.x - q2.x, q1.y - q2.y, q1.z - q2.z)

        else:

            return NotImplemented

    def __mul__(object x, object y):
        """Multiplication operator.

        .. code-block:: pycon

            >>> Quaternion(1, 0, 1, 0) * 2
            Quaternion(2.0, 0.0, 2.0, 0.0)
            >>> Quaternion(1, 0, 1, 0) * Quaternion(0, 1, 2, 3)
            Quaternion(-2.0, 4.0, 2.0, 2.0)
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

            return q1.mul(q2)

        else:

            return NotImplemented()

    @cython.cdivision(True)
    def __truediv__(object x, object y):
        """Division operator.

        .. code-block:: pycon

            >>> Quaternion(1, 0, 1, 0) / 2
            Quaternion(0.5, 0.0, 0.5, 0.0)
            >>> Quaternion(1, 0, 1, 0) / Quaternion(0, 1, 2, 3)
            Quaternion(0.14286, -0.28571, -0.14286, -0.14286)
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

            return q1.div(q2)

        else:

            raise TypeError("Unsupported operand type. Expects a real number.")

    cdef Quaternion neg(self):
        """
        Fast negation operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        return new_quaternion(-self.s, -self.x, -self.y, -self.z)

    cdef Quaternion add(self, Quaternion q2):
        """
        Fast addition operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        cdef Quaternion q1 = self

        return new_quaternion(q1.s + q2.s, q1.x + q2.x, q1.y + q2.y, q1.z + q2.z)

    cdef Quaternion sub(self, Quaternion q2):
        """
        Fast subtraction operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        cdef Quaternion q1 = self

        return new_quaternion(q1.s - q2.s, q1.x - q2.x, q1.y - q2.y, q1.z - q2.z)

    cdef Quaternion mul(self, Quaternion q2):
        """
        Fast multiplication operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        cdef Quaternion q1 = self
        cdef double ns, nx, ny, nz

        ns = q1.s*q2.s - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
        nx = q1.s*q2.x + q1.x*q2.s + q1.y*q2.z - q1.z*q2.y
        ny = q1.s*q2.y - q1.x*q2.z + q1.y*q2.s + q1.z*q2.x
        nz = q1.s*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.s

        return new_quaternion(ns, nx, ny, nz)

    cdef Quaternion mul_scalar(self, double d):

        cdef Quaternion q = self

        return new_quaternion(d * q.s, d * q.x, d * q.y, d * q.z)

    cpdef Quaternion conjugate(self):
        """
        Complex conjugate operator. 
        
        .. code-block:: pycon

            >>> Quaternion(0, 1, 2, 3).conjugate()
            Quaternion(0, -1, -2, -3)
        """

        return new_quaternion(self.s, -self.x, -self.y, -self.z)

    @cython.cdivision(True)
    cpdef Quaternion inverse(self):
        """
        Inverse operator.

        .. code-block:: pycon

            >>> Quaternion(0, 1, 2, 3).inverse()
            Quaternion(0.0, -0.07143, -0.14286, -0.21429)
        """

        cdef Quaternion q = self
        cdef double n

        n = self.norm()**2
        return new_quaternion(q.s/n, -q.x/n, -q.y/n, -q.z/n)

    cpdef double norm(self):
        """
        Calculates the norm of the quaternion.

        .. code-block:: pycon

            >>> Quaternion(0, 1, 2, 3).norm()
            3.7416573867739413
        """

        return sqrt(self.s * self.s + self.x * self.x + self.y * self.y + self.z * self.z)

    cdef Quaternion div(self, Quaternion q2):
        """
        Fast division operator.

        This is a cython only function and is substantially faster than a call
        to the equivalent python operator.
        """

        cdef Quaternion q1 = self, q2_inv

        q2_inv = q2.inverse()

        return q1.mul(q2_inv)

    @cython.cdivision(True)
    cdef Quaternion div_scalar(self, double d):

        cdef Quaternion q = self

        # prevent divide my zero
        if d == 0.0:

            raise ZeroDivisionError("Cannot divide a quaternion by a zero scalar.")

        d = 1.0 / d

        return new_quaternion(d * q.s, d * q.x, d * q.y, d * q.z)

    @cython.cdivision(True)
    cpdef Quaternion normalise(self):
        """
        Returns a normalised copy of the quaternion.

        The returned quaternion is normalised to have norm length 1.0 - a unit quaternion.

        .. code-block:: pycon
        
            >>> a = Quaternion(0, 1, 2, 3)
            >>> a.normalise()
            Quaternion(0.0, 0.26726, 0.53452, 0.80178)
        """

        cdef double n

        # if current length is zero, problem is ill defined
        n = self.norm()
        if n == 0.0:

            raise ZeroDivisionError("A zero length quaternion cannot be normalised as the direction "
                                    "of a zero length quaternion is undefined.")

        # normalise and rescale quaternion
        n = 1.0 / n

        return self.mul_scalar(n)

    cpdef Quaternion copy(self):
        """Returns a copy of this quaternion."""

        return new_quaternion(self.s, self.x, self.y, self.z)

    cpdef AffineMatrix3D to_matrix(self):
        """
        Generates an AffineMatrix3D representation of this Quaternion.

        .. code-block:: pycon

           >>> from raysect.core.math import Quaternion
           >>>
           >>> q = Quaternion(0.5, 0.5, 0, 0)
           >>> q.to_matrix()
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

        qs = unit_q.s
        qx = unit_q.x
        qy = unit_q.y
        qz = unit_q.z

        qs2 = qs * qs
        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz

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
           Quaternion(0.7071067811865476, 0.7071067811865475, 0.0, 0.0)
        """

        cdef:
            AffineMatrix3D m = matrix
            double qs, qx, qy, qz
            double trace, s

        trace = m.m[0][0] + m.m[1][1] + m.m[2][2]

        if trace > 0:

            s = sqrt(trace+1.0) * 2  # s = 4*qs
            qs = 0.25 * s
            qx = (m.m[2][1] - m.m[1][2]) / s
            qy = (m.m[0][2] - m.m[2][0]) / s
            qz = (m.m[1][0] - m.m[0][1]) / s

        elif m.m[0][0] > m.m[1][1] and m.m[0][0] > m.m[2][2]:

            s = sqrt(1.0 + m.m[0][0] - m.m[1][1] - m.m[2][2]) * 2  # s = 4*qx
            qs = (m.m[2][1] - m.m[1][2]) / s
            qx = 0.25 * s
            qy = (m.m[0][1] + m.m[1][0]) / s
            qz = (m.m[0][2] + m.m[2][0]) / s

        elif m.m[1][1] > m.m[2][2]:

            s = sqrt(1.0 + m.m[1][1] - m.m[0][0] - m.m[2][2]) * 2  # s = 4*qy
            qs = (m.m[0][2] - m.m[2][0]) / s
            qx = (m.m[0][1] + m.m[1][0]) / s
            qy = 0.25 * s
            qz = (m.m[1][2] + m.m[2][1]) / s

        else:

            s = sqrt(1.0 + m.m[2][2] - m.m[0][0] - m.m[1][1]) * 2  # s = 4*qz
            qs = (m.m[1][0] - m.m[0][1]) / s
            qx = (m.m[0][2] + m.m[2][0]) / s
            qy = (m.m[1][2] + m.m[2][1]) / s
            qz = 0.25 * s

        return new_quaternion(qs, qx, qy, qz)

    @classmethod
    def from_axis_angle(cls, Vector3D axis, double angle):
        """
        Generates a new Quaternion from the axis-angle specification.

        :param Vector3D axis: The axis about which rotation will be performed.
        :param float angle: An angle in degrees specifiying the magnitude of the
          rotation about the axis vector.
        :return: A new Quaternion object representing the specified rotation.

        .. code-block:: pycon

           >>> from raysect.core.math import Quaternion, Vector3D
           >>>
           >>> Quaternion.from_axis_angle(Vector3D(1, 0, 0), 45)
           Quaternion(0.9238795325112867, 0.3826834323650898, 0.0, 0.0)
        """

        if not -180 <= angle <= 180:
            raise ValueError("The angle of rotation must be in the range (-180, 180).")

        axis = axis.normalise()

        theta_2 = angle * DEG2RAD / 2

        qs = cos(theta_2)
        qx = axis.x * sin(theta_2)
        qy = axis.y * sin(theta_2)
        qz = axis.z * sin(theta_2)

        return new_quaternion(qs, qx, qy, qz)
