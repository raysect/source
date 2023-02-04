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


@cython.freelist(256)
cdef class _Mat4:
    """4x4 matrix base class."""

    def __init__(self, v = ((1.0, 0.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0, 0.0),
                            (0.0, 0.0, 1.0, 0.0),
                            (0.0, 0.0, 0.0, 1.0))):
        """
        4x4 Matrix constructor.

        If no initial values are passed, _Mat4 defaults to an identity matrix.

        Any 4 x 4 indexable or 16 element object can be used to initialise the
        matrix. 16 element objects must be specified in row-major format.
        """

        cdef int i, j
        cdef _Mat4 m

        # special handling for _Mat4
        if isinstance(v, _Mat4):

            m = <_Mat4>v
            for i in range(0, 4):
                for j in range(0, 4):
                    self.m[i][j] = m.m[i][j]
            return

        # try reading object as 4x4 array of elements with separate indexing
        try:
            for i in range(0, 4):
                for j in range(0, 4):
                    self.m[i][j] = v[i][j]
            return
        except(IndexError, TypeError):
            pass

        # try reading object as 4x4 array of elements with tuple indexing
        try:
            for i in range(0, 4):
                for j in range(0, 4):
                    self.m[i][j] = v[i,j]
            return
        except(IndexError, TypeError):
            pass

        # try reading object as row-major 16 element array
        try:
            for i in range(0, 4):
                for j in range(0, 4):
                    self.m[i][j] = v[4*i + j]
            return
        except(IndexError, TypeError):
            raise TypeError("Matrix can only be initialised with an indexable object of 4 by 4 or 16 numerical values in row-major order.")

    def __getitem__(self, object i):
        """
        Indexing get operator.

        Expects a tuple (row, column) as the index.

        e.g. v = matrix[1, 2]

        """

        cdef int row, column

        try:
            row = i[0]
            column = i[1]
        except:
            raise IndexError("Index must be a tuple containing (at least) the row and column indicies e.g. matrix[1,3].")

        if row < 0 or row > 3:
            raise IndexError("Row index out of range [0, 3].")

        if column < 0 or column > 3:
            raise IndexError("Column index out of range [0, 3].")

        return self.m[row][column]

    def __setitem__(self, object i, double v):
        """
        Indexing set operator.

        Expects a tuple (row, column) as the index.

        e.g. matrix[1, 2] = 7.0
        """

        cdef int row, column

        try:
            row = i[0]
            column = i[1]
        except:
            raise IndexError("Index must be a tuple containing (at least) the row and column indicies e.g. matrix[1,3].")

        if row < 0 or row > 3:
            raise IndexError("Row index out of range [0, 3].")

        if column < 0 or column > 3:
            raise IndexError("Column index out of range [0, 3].")

        self.m[row][column] = v

    def __getstate__(self):
        """Encodes state for pickling."""

        return (self.m[0][0], self.m[0][1], self.m[0][2], self.m[0][3],
                self.m[1][0], self.m[1][1], self.m[1][2], self.m[1][3],
                self.m[2][0], self.m[2][1], self.m[2][2], self.m[2][3],
                self.m[3][0], self.m[3][1], self.m[3][2], self.m[3][3])

    def __setstate__(self, state):
        """Encodes state for pickling."""

        (self.m[0][0], self.m[0][1], self.m[0][2], self.m[0][3],
         self.m[1][0], self.m[1][1], self.m[1][2], self.m[1][3],
         self.m[2][0], self.m[2][1], self.m[2][2], self.m[2][3],
         self.m[3][0], self.m[3][1], self.m[3][2], self.m[3][3]) = state

    cdef double get_element(self, int row, int column):
        """
        Fast getting of matrix elements via indexing.

        Cython equivalent to __getitem__, without the checks and call overhead.

        For speed, this function performs no bounds checking. Invalid indexing
        may cause a segfault.
        """

        return self.m[row][column]

    cdef void set_element(self, int row, int column, double v):
        """
        Fast setting of matrix elements via indexing.

        Cython equivalent to __setitem__, without the checks and call overhead.

        For speed, this function performs no bounds checking. Invalid indexing
        may cause a segfault.
        """

        self.m[row][column] = v

    cpdef bint is_identity(self, double tolerance=1e-8):
        """
        Identifies if the matrix is an identity matrix.
        
        Returns True if the matrix is an identify matrix, False otherwise.
        
        The method has a default tolerance of 1e-8 to account for errors due to
        numerical accuracy limits. The tolerance may be altered by setting the
        tolerance argument. 
         
        :param tolerance: Numerical tolerance (default: 1e-8)
        :return: True/False
        """

        return self.is_close(_Mat4(), tolerance)

    cpdef bint is_close(self, _Mat4 other, double tolerance=1e-8):
        """
        Is this matrix equal to another matrix within a numerical tolerance.
        
        The method has a default tolerance of 1e-8 to account for errors that
        may have accumulated due to numerical accuracy limits. The tolerance
        may be altered by setting the tolerance argument. 
         
        :param other: The other matrix.
        :param tolerance: Numerical tolerance (default: 1e-8)
        :return: True/False
        """

        cdef:
            int i, j
            double v

        if tolerance < 0:
            raise ValueError('Tolerance cannot be less than zero.')

        for i in range(0, 4):
            for j in range(0, 4):
                if abs(self.m[i][j] - other.m[i][j]) >= tolerance:
                    return False

        return True