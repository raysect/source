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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double linear2d(double x0, double x1, double y0, double y1, double [:,::1] f, double x, double y) nogil:
    """
    :param x0: 
    :param x1: 
    :param y0: 
    :param y1: 
    :param f: 
    :param x: 
    :param y: 
    :return: 
    """

    # interpolate along x
    cdef double k0 = linear1d(x0, x1, f[0][0], f[1][0], x)
    cdef double k1 = linear1d(x0, x1, f[0][1], f[1][1], x)

    # interpolate along y
    return linear1d(y0, y1, k0, k1, y)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double linear3d(double x0, double x1, double y0, double y1, double z0, double z1, double[:,:,::1] f, double x, double y, double z) nogil:
    """
    :param x0: 
    :param x1: 
    :param y0: 
    :param y1: 
    :param z0: 
    :param z1: 
    :param f:
    :param x: 
    :param y: 
    :param z: 
    :return: 
    """

    # interpolate along x
    cdef double k00 = linear1d(x0, x1, f[0][0][0], f[1][0][0], x)
    cdef double k01 = linear1d(x0, x1, f[0][0][1], f[1][0][1], x)
    cdef double k10 = linear1d(x0, x1, f[0][1][0], f[1][1][0], x)
    cdef double k11 = linear1d(x0, x1, f[0][1][1], f[1][1][1], x)

    # interpolate along y
    cdef double m0 = linear1d(y0, y1, k00, k10, y)
    cdef double m1 = linear1d(y0, y1, k01, k11, y)

    # interpolate along z
    return linear1d(z0, z1, m0, m1, z)
