
from raysect.core.math.affinematrix cimport AffineMatrix3D
from raysect.core.math.point cimport Point3D


cdef class Quaternion:

    cdef public double s, x, y, z

    cdef Quaternion neg(self)

    cdef Quaternion add(self, Quaternion q2)

    cdef Quaternion sub(self, Quaternion q2)

    cdef Quaternion mul(self, Quaternion q2)

    cdef Quaternion mul_scalar(self, double d)

    cpdef Quaternion inv(self)

    cpdef double norm(self)

    cdef Quaternion div(self, Quaternion q2)

    cdef Quaternion div_scalar(self, double d)

    cpdef Quaternion normalise(self)

    cpdef Quaternion copy(self)

    cpdef AffineMatrix3D to_transform(self, Point3D origin=*)


cpdef Quaternion mat_to_quat(AffineMatrix3D matrix)


cdef inline Quaternion new_quaternion(double s, double x, double y, double z):
    """
    Quaternion factory function.

    Creates a new Quaternion object with less overhead than the equivalent Python
    call. This function is callable from cython only.
    """

    cdef Quaternion q
    q = Quaternion.__new__(Quaternion)
    q.s = s
    q.x = x
    q.y = y
    q.z = z
    return q

