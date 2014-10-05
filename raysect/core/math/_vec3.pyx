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
from libc.math cimport sqrt
from raysect.core.math.point cimport Point

cdef class _Vec3:
    """3D Vector base class."""

    # def __cinit__(self, double x=0.0, double y=0.0, double z=1.0):
    #     """
    #     Point constructor.
    #
    #     If no initial values are passed, _Vec3 defaults to a unit vector
    #     aligned with the z-axis: _Vec(0.0, 0.0, 1.0)
    #     """
    #
    #     self.x = x
    #     self.y = y
    #     self.z = z

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

        if i < 0 or i > 2:
            raise IndexError("Index out of range [0, 2].")
        elif i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            return self.z

    def __setitem__(self, int i, double value):
        """Sets the vector coordinates by index ([0,1,2] -> [x,y,z])."""

        if i < 0 or i > 2:
            raise IndexError("Index out of range [0, 2].")
        elif i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            self.z = value

    property length:
        """
        The vector's length.

        Raises a ZeroDivisionError if an attempt is made to change the length of
        a zero length vector. The direction of a zero length vector is
        undefined hence it can not be lengthened.
        """

        def __get__(self):

            return self.get_length()

        def __set__(self, double v):

            self.set_length(v)

    cpdef double dot(self, _Vec3 v):
        """
        Calculates the dot product between this vector and the supplied vector.

        Returns a scalar.
        """

        return self.x * v.x + self.y * v.y + self.z * v.z

    # length getters/setters
    cdef inline double get_length(self):

        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    cdef inline void set_length(self, double v) except *:

        cdef double t

        # if current length is zero, problem is ill defined
        t = self.x * self.x + self.y * self.y + self.z * self.z
        if t == 0.0:

            raise ZeroDivisionError("A zero length vector can not be rescaled as the direction of a zero length vector is undefined.")

        # normalise and rescale vector
        with cython.cdivision(True):

            t = v / sqrt(t)

        self.x = self.x * t
        self.y = self.y * t
        self.z = self.z * t

    cdef inline double get_index(self, int index):
        """
        Fast getting of coordinates via indexing.

        Cython equivalent to __getitem__, without the checks and call overhead.
        """

        if index == 0: return self.x
        elif index == 1: return self.y
        elif index == 2: return self.z
        else: return float("NaN")

    cdef inline void set_index(self, int index, double value):
        """
        Fast setting of coordinates via indexing.

        Cython equivalent to __setitem__, without the checks and call overhead.
        """

        if index == 0: self.x = value
        elif index == 1: self.y = value
        elif index == 2: self.z = value