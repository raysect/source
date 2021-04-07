# cython: language_level=3

# Copyright (c) 2014-2020, Dr Alex Meakins, Raysect Project
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

from raysect.optical.mueller cimport MuellerMatrix
from libc.math cimport sqrt
cimport cython


@cython.freelist(128)
cdef class StokesVector:

    # todo: complete docstring
    def __init__(self, double i=0, double q=0, double u=0, double v=0):
        """
        All orientations are defined looking down the beam in the direction of propagation.

            i is the total light intensity
            q and u describe the angle of linear polarisation
            v is the handedness of the circular polarisation
                v > 0: Right-handed polarisation
                v < 0: Left-handed polarisation

        The Stoke's vector component magnitudes are in units of spectral radiant intensity (W/m2/str/nm).

        The Stoke's vector in Raysect is defined such that:

            * horizontal linear polarisation aligns with the x axis
            * vertical linear polarisation aligns with the y axis
            * beam propagation is along the z axis
            * right hand circular light rotates counter-clockwise looking along the beam in the direction of propagation

        """

        if i < 0:
            raise ValueError('The total radiance of the Stoke\'s vector cannot be less than 0.')

        polarised_radiance = sqrt(self.q*self.q + self.u*self.u + self.v*self.v)
        if polarised_radiance > i:
            raise ValueError('The polarised radiance fraction cannot exceed the total radiance.')

        self.i = i
        self.q = q
        self.u = u
        self.v = v

    @cython.cdivision(True)
    cpdef double polarised_fraction(self):
        """
        Returns the degree of polarisation.
        
        A value in the range [0, 1] is returned that indicates the degree of
        polarisation. A value of zero means the light is unpolarised. A value
        of one means the light is entirely polarised.
        
        :returns: Fraction of polarisation.   
        """

        if self.i == 0.0:
            return 0.0

        return sqrt(self.q*self.q + self.u*self.u + self.v*self.v) / self.i

    @cython.cdivision(True)
    cpdef double linear_fraction(self):
        """
        Returns the degree of linear polarisation.
        
        A value in the range [0, 1] is returned that indicates the degree of
        linear polarisation. A value of zero means there is no linear
        polarised component. A value of one means the light is entirely
        linear polarised.
        
        :returns: Fraction of linear polarisation.   
        """

        if self.i == 0.0:
            return 0.0

        return sqrt(self.q*self.q + self.u*self.u) / self.i

    @cython.cdivision(True)
    cpdef double circular_fraction(self):
        """
        Returns the degree of circular polarisation.
        
        A value in the range [0, 1] is returned that indicates the degree of
        circular polarisation. A value of zero means there is no circular
        polarised component. A value of one means the light is entirely
        circular polarised.
        
        :returns: Fraction of circular polarisation.   
        """

        if self.i == 0.0:
            return 0.0

        return self.v / self.i

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.i, self.q, self.u, self.v

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.i = state[0]
        self.q = state[1]
        self.u = state[2]
        self.v = state[3]

    def __repr__(self):
        """Returns a string representation of the StokesVector object."""

        return f'StokesVector({self.i}, {self.q}, {self.u}, {self.v})'

    def __richcmp__(self, object other, int op):
        """Provides basic vector comparison operations."""

        cdef StokesVector v

        if not isinstance(other, StokesVector):
            return NotImplemented

        v = <StokesVector> other
        if op == 2:     # __eq__()
            return self.i == v.i and self.q == v.q and self.u == v.u and self.v == v.v
        elif op == 3:   # __ne__()
            return self.i != v.i or self.q != v.q or self.u != v.u or self.v != v.v
        else:
            return NotImplemented

    def __getitem__(self, int i):
        """Returns the values by index ([0, 1, 2, 3] -> [i, q, u, v]).

            >>> a = StokesVector(1, 0, 0, 0)
            >>> a[0]
            1
        """

        if i == 0:
            return self.i
        elif i == 1:
            return self.q
        elif i == 2:
            return self.u
        elif i == 3:
            return self.v
        else:
            raise IndexError("Index out of range [0, 3].")

    def __iter__(self):
        """Iterates over the vector values (i, q, u, v)

            >>> a = StokesVector(5.0, -1.0, 0.2, 0.4)
            >>> i, q, u, v = a
            >>> i, q, u, v
            (5.0, -1.0, 0.2, 0.4)
        """

        yield self.i
        yield self.q
        yield self.u
        yield self.v

    def __add__(object x, object y):
        """Addition operator.

            >>> StokesVector(1, 0, 0, 0) + StokesVector(0.5, 0.3, 0.4, 0.0)
            StokesVector(1.5, 0.3, 0.4, 0.0)
        """

        cdef StokesVector vx, vy

        if isinstance(x, StokesVector) and isinstance(y, StokesVector):

            vx = <StokesVector>x
            vy = <StokesVector>y
            return new_stokesvector(
                vx.i + vy.i,
                vx.q + vy.q,
                vx.u + vy.u,
                vx.v + vy.v
            )

        else:
            return NotImplemented

    def __mul__(object x, object y):
        """Multiplication operator.

        StokesVectors can be multiplied with both scalars and Mueller matrices.
        """

        cdef double s
        cdef StokesVector v
        cdef MuellerMatrix m

        if isinstance(x, numbers.Real) and isinstance(y, StokesVector):

            s = <double>x
            v = <StokesVector>y
            return new_stokesvector(s * v.i, s * v.q, s * v.u, s * v.v)

        elif isinstance(x, StokesVector) and isinstance(y, numbers.Real):

            s = <double>y
            v = <StokesVector>x
            return new_stokesvector(s * v.i, s * v.q, s * v.u, s * v.v)

        elif isinstance(x, MuellerMatrix) and isinstance(y, StokesVector):

            m = <MuellerMatrix>x
            v = <StokesVector>y
            return new_stokesvector(
                m.m[0][0] * v.i + m.m[0][1] * v.q + m.m[0][2] * v.u + m.m[0][3] * v.v,
                m.m[1][0] * v.i + m.m[1][1] * v.q + m.m[1][2] * v.u + m.m[1][3] * v.v,
                m.m[2][0] * v.i + m.m[2][1] * v.q + m.m[2][2] * v.u + m.m[2][3] * v.v,
                m.m[3][0] * v.i + m.m[3][1] * v.q + m.m[3][2] * v.u + m.m[3][3] * v.v
            )

        else:
            return NotImplemented

