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
from libc.math cimport sqrt, acos, NAN, M_PI


@cython.freelist(512)
cdef class _Vec3:
    """3D Vector base class."""

    def __init__(self, double x=0.0, double y=0.0, double z=1.0):
        """
        Constructor.

        If no initial values are passed, _Vec3 defaults to a unit vector
        aligned with the z-axis: _Vec(0.0, 0.0, 1.0)
        """

        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, int i):
        """Returns the vector coordinates by index ([0,1,2] -> [x,y,z])."""

        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        elif i == 2:
            return self.z
        else:
            raise IndexError("Index out of range [0, 2].")

    def __setitem__(self, int i, double value):
        """Sets the vector coordinates by index ([0,1,2] -> [x,y,z])."""

        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        elif i == 2:
            self.z = value
        else:
            raise IndexError("Index out of range [0, 2].")

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.x, self.y, self.z

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.x = state[0]
        self.y = state[1]
        self.z = state[2]

    @property
    def length(self):
        """
        The vector's length.

        Raises a ZeroDivisionError if an attempt is made to change the length of
        a zero length vector. The direction of a zero length vector is
        undefined hence it can not be lengthened.

            >>> a = Vector3D(1, 1, 1)
            >>> a.length
            1.7320508075688772
        """
        return self.get_length()

    @length.setter
    def length(self, double v):
        self.set_length(v)

    cpdef double dot(self, _Vec3 v):
        """
        Calculates the dot product between this vector and the supplied vector.

        :rtype: float

        .. code-block:: pycon

            >>> a = Vector3D(1, 1, 1)
            >>> b = Vector3D(1, 0, 0)
            >>> a.dot(b)
            1.0
        """

        return self.x * v.x + self.y * v.y + self.z * v.z

    cpdef double angle(self, _Vec3 v):
        """
        Calculates the angle between this vector and the supplied vector.

        Returns the angle in degrees.

            >>> a = Vector3D(1, 1, 1)
            >>> b = Vector3D(1, 0, 0)
            >>> a.angle(b)
            54.735610317245346
        """

        return acos(min(1, self.dot(v) / (self.get_length() * v.get_length()))) * 180 / M_PI

    cdef double get_length(self) nogil:
        """
        Fast function to obtain the vectors length.

        Cython only, equivalent to length.__get__() property.

        Use instead of Python attribute access in cython code.
        """

        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @cython.cdivision(True)
    cdef object set_length(self, double v):
        """
        Fast function to set the vectors length.

        Cython only, equivalent to length.__set__() property.

        Use instead of Python attribute access in cython code.
        """

        cdef double t

        # if current length is zero, problem is ill defined
        t = self.x * self.x + self.y * self.y + self.z * self.z
        if t == 0.0:
            raise ZeroDivisionError("A zero length vector cannot be rescaled as the direction of a zero length vector is undefined.")

        # normalise and rescale vector
        t = v / sqrt(t)

        self.x = self.x * t
        self.y = self.y * t
        self.z = self.z * t

    cdef double get_index(self, int index) nogil:
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
            return NAN

    cdef void set_index(self, int index, double value) nogil:
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