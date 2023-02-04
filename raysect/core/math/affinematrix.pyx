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
from libc.math cimport fabs



cdef class AffineMatrix3D(_Mat4):
    """A 4x4 affine matrix.

    These matrices are used for transforming between coordinate systems. Every
    primitive in Raysect works in its own local coordinate system, so it is common
    to need to transform 3D points from local to world spave and vice versa. Even
    though the vectors themselves are 3D, a 4x4 matrix is needed to completely
    specify a transformation from one 3D space to another.

    The coordinate transformation is applied by multiplying the column vector for
    the desired Point3D/Vector3D against the transformation matrix. For example,
    if the original vector :math:`\\vec{V_a}` is in space A and the transformation matrix
    :math:`\\mathbf{T_{AB}}` describes the position and orientation of Space A relative
    to Space B, then the multiplication

    .. math::

        \\vec{V_{b}} = \\mathbf{T_{AB}} \\times \\vec{V_a}

    yields the same vector transformed into coordinate Space B, :math:`\\vec{V_b}`.

    The individual terms of the transformation matrix can be visualised in terms of
    the way they change the underlying basis vectors.

    .. math::

        \\mathbf{T_{AB}} = \\left( \\begin{array}{cccc} \\vec{x_b}.x & \\vec{y_b}.x & \\vec{z_b}.x & \\vec{t}.x \\\\
        \\vec{x_b}.y & \\vec{y_b}.y & \\vec{z_b}.y & \\vec{t}.y \\\\
        \\vec{x_b}.z & \\vec{y_b}.z & \\vec{z_b}.z & \\vec{t}.z \\\\
        0 & 0 & 0 & 1 \\end{array} \\right)

    Here the unit x-axis vector in space A, :math:`\\vec{x}_a = (1, 0, 0)`, has been transformed
    into space B, :math:`\\vec{x_b}`. The same applies to :math:`\\vec{y_b}` and :math:`\\vec{z_b}` for
    the  :math:`\\vec{y}_a` and :math:`\\vec{z}_a` unit vectors respectively. Together the new basis
    vectors describe a rotation of the original coordinate system.

    The vector :math:`\\vec{t}` in the last column corresponds to a translation vector between the
    origin's of space A and space B.

    Strictly speaking, the new rotation vectors don't have to be normalised which
    corresponds to a scaling in addition to the rotation. For example, a scaling matrix
    would look like the following.

    .. math::

        \\mathbf{T_{scale}} = \\left( \\begin{array}{cccc} \\vec{s}.x & 0 & 0 & 0 \\\\
        0 & \\vec{s}.y & & 0 \\\\
        0 & 0 & \\vec{s}.z & 0 \\\\
        0 & 0 & 0 & 1 \\end{array} \\right)

    Multiple transformations can be chained together by multiplying the
    matrices together, the resulting matrix will encode the full transformation.
    The order in which transformations are applied is very important. The operation
    :math:`\\mathbf{M_{translate}} \\times \\mathbf{M_{rotate}}` is different to
    :math:`\\mathbf{M_{rotate}} \\times \\mathbf{M_{translate}}` because matrices don't
    commute, and physically these are different operations.

    .. warning::
        Because we are using column vectors, transformations should be
        applied **right to left**.

    An an example operation, let us consider the case of moving and rotating a camera in
    our scene. Suppose we want to rotate our camera at an angle of :math:`\\theta_x=45`
    around the x-axis and translate the camera to position :math:`p=(0, 0, 3.5)`. This set
    of operations would be equivalent to:

    .. math::

        \\mathbf{T} = \\mathbf{T_{translate}} \\times \\mathbf{T_{rotate}}

    In code this would be equivalent to: ::

        >>> transform = translate(0, 0, -3.5) * rotate_x(45)

    If no initial values are passed to the matrix, it defaults to an identity matrix.

    :param object m: Any 4 x 4 indexable or 16 element object can be used to
      initialise the matrix. 16 element objects must be specified in
      row-major format.

    """

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
        """Multiplication operator.

            >>> from raysect.core import translate, rotate_x
            >>> translate(0, 0, -3.5) * rotate_x(45)
            AffineMatrix3D([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.7071067811865476, -0.7071067811865475, 0.0],
                            [0.0, 0.7071067811865475, 0.7071067811865476, -3.5],
                            [0.0, 0.0, 0.0, 1.0]])
        """

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

            >>> from raysect.core import AffineMatrix3D
            >>> m = AffineMatrix3D([[0.0, 0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])
            >>> m.inverse()
            AffineMatrix3D([[0.0, 1.0, 0.0, -0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, -0.0],
                            [0.0, 0.0, -0.0, 1.0]])
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

    cdef AffineMatrix3D mul(self, AffineMatrix3D m):

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
