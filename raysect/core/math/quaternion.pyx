
import numbers
cimport cython
from libc.math cimport sqrt

from raysect.core.math.affinematrix cimport new_affinematrix3d


cdef class Quaternion:

    def __init__(self, double s=1.0, double x=0.0, double y=0.0, double z=0.0):

        self.s = s
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        """Returns a string representation of the Quaternion object."""

        return "Quaternion(" + str(self.s) + ", <" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ">)"

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
            Quaternion(1.0, <2.0, 0.0, 0.0>)
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
            Quaternion(-1.0, <-0.0, -0.0, -0.0>)
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
            Quaternion(1.0, <1.0, 0.0, 0.0>)
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
            Quaternion(1.0, <-1.0, 0, 0>)
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
            Quaternion(2.0, <0.0, 2.0, 0.0>)
            >>> Quaternion(1, 0, 1, 0) * Quaternion(0, 1, 2, 3)
            Quaternion(-2.0, <4.0, 2.0, 2.0>)
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
            Quaternion(0.5, <0.0, 0.5, 0.0>)
            >>> Quaternion(1, 0, 1, 0) / Quaternion(0, 1, 2, 3)
            Quaternion(0.14286, <-0.28571, -0.14286, -0.14286>)
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

    @cython.cdivision(True)
    cpdef Quaternion inv(self):
        """
        Inverse operator.

        .. code-block:: pycon

            >>> Quaternion(0, 1, 2, 3).inv()
            Quaternion(0.0, <-0.07143, -0.14286, -0.21429>)
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

        q2_inv = q2.inv()

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
            Quaternion(0.0, <0.26726, 0.53452, 0.80178>)
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

    cpdef AffineMatrix3D to_transform(self, Point3D origin=Point3D(0, 0, 0)):
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

        .. code-block:: pycon

            >>> z = Vector3D(0, 0, 1)
            >>> y = z.transform(rotate_x(90))
            >>> y
            Vector3D(0.0, -1.0, 6.123233995736766e-17)
        """

        cdef:
            double sqs, sqx, sqy, sqz
            double m00, m01, m02, m03
            double m10, m11, m12, m13
            double m20, m21, m22, m23
            double m30, m31, m32, m33
            double temp1, temp2
            Point3D o = origin

        # 1 - 2*qy2 - 2*qz2 	2*qx*qy - 2*qz*qw 	2*qx*qz + 2*qy*qw
        # 2*qx*qy + 2*qz*qw 	1 - 2*qx2 - 2*qz2 	2*qy*qz - 2*qx*qw
        # 2*qx*qz - 2*qy*qw 	2*qy*qz + 2*qx*qw 	1 - 2*qx2 - 2*qy2

        sqs = self.s * self.s
        sqx = self.x * self.x
        sqy = self.y * self.y
        sqz = self.z * self.z

        m00 = sqs + sqx - sqy - sqz
        m11 = sqs -sqx + sqy - sqz
        m22 = sqs -sqx - sqy + sqz

        temp1 = self.x * self.y
        temp2 = self.z * self.s
        m01 = 2.0 * (temp1 + temp2)
        m10 = 2.0 * (temp1 - temp2)

        temp1 = self.x * self.z
        temp2 = self.y * self.s
        m02 = 2.0 * (temp1 - temp2)
        m20 = 2.0 * (temp1 + temp2)

        temp1 = self.y * self.z
        temp2 = self.x * self.s
        m12 = 2.0 * (temp1 + temp2)
        m21 = 2.0 * (temp1 - temp2)

        m03 = o.x - o.x * m00 - o.y * m01 - o.z * m02
        m13 = o.y - o.x * m10 - o.y * m11 - o.z * m12
        m23 = o.z - o.x * m20 - o.y * m21 - o.z * m22

        m30 = m31 = m32 = 0.0
        m33 = 1.0

        return new_affinematrix3d(m00, m01, m02, m03,
                                  m10, m11, m12, m13,
                                  m20, m21, m22, m23,
                                  m30, m31, m32, m33)


cpdef Quaternion mat_to_quat(AffineMatrix3D matrix):

    cdef:
        AffineMatrix3D m = matrix
        double qs, qx, qy, qz
        double trace, s

    trace = m.m[0][0] + m.m[1][1] + m.m[2][2] + m.m[3][3]

    if trace > 0:

        s = sqrt(trace+1.0) * 2  # s = 4*qs
        qs = 0.25 * s
        qs = (m.m[2][1] - m.m[1][2]) / s
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
