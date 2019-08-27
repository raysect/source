# cython: language_level=3

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

cimport cython
from libc.stdint cimport uint8_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double cubic2d(double x0, double x1, double y0, double y1, double[:,:,::1] f,
                    double[:,:,::1] dfdx, double[:,:,::1] dfdy, double[:,:,::1] dfdxdy,
                    double x, double y) nogil:

    cdef double a[4][4]
    _calculate_coeff_2d(a)
    return _evaluate_cubic_2d(a, x, y)


cdef void _calculate_coeff_2d(double a[4][4]) nogil:
    pass


cdef double _evaluate_cubic_2d(double a[4][4], double x, double y) nogil:

    cdef double x2 = x*x
    cdef double x3 = x2*x

    cdef double y2 = y*y
    cdef double y3 = y2*y

    # calculate cubic polynomial
    return a[0][0] + a[0][1]*y + a[0][2]*y2 + a[0][3]*y3 + \
        a[1][0]*x + a[1][1]*x*y + a[1][2]*x*y2 + a[1][3]*x*y3 + \
        a[2][0]*x2 + a[2][1]*x2*y + a[2][2]*x2*y2 + a[2][3]*x2*y3 + \
        a[3][0]*x3 + a[3][1]*x3*y + a[3][2]*x3*y2 + a[3][3]*x3*y3


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double cubic3d(double x0, double x1, double y0, double y1, double z0, double z1, double[:,:,::1] f,
                    double[:,:,::1] dfdx, double[:,:,::1] dfdy, double[:,:,::1] dfdz,
                    double[:,:,::1] dfdxdy, double[:,:,::1] dfdxdz, double[:,:,::1] dfdydz,
                    double[:,:,::1] dfdxdydz, double x, double y, double z) nogil:

    cdef double a[4][4][4]
    _calculate_coeff_3d(a)
    return _evaluate_cubic_3d(a, x, y, z)


cdef void _calculate_coeff_3d(double a[4][4][4]) nogil:



    pass


cdef double _evaluate_cubic_3d(double a[4][4][4], double x, double y, double z) nogil:

    cdef double x2 = x*x
    cdef double x3 = x2*x

    cdef double y2 = y*y
    cdef double y3 = y2*y

    cdef double z2 = z*z
    cdef double z3 = z2*z

    return a[0][0][0] + a[0][0][1]*z + a[0][0][2]*z2 + a[0][0][3]*z3 + \
        a[0][1][0]*y + a[0][1][1]*y*z + a[0][1][2]*y*z2 + a[0][1][3]*y*z3 + \
        a[0][2][0]*y2 + a[0][2][1]*y2*z + a[0][2][2]*y2*z2 + a[0][2][3]*y2*z3 + \
        a[0][3][0]*y3 + a[0][3][1]*y3*z + a[0][3][2]*y3*z2 + a[0][3][3]*y3*z3 + \
        a[1][0][0]*x + a[1][0][1]*x*z + a[1][0][2]*x*z2 + a[1][0][3]*x*z3 + \
        a[1][1][0]*x*y + a[1][1][1]*x*y*z + a[1][1][2]*x*y*z2 + a[1][1][3]*x*y*z3 + \
        a[1][2][0]*x*y2 + a[1][2][1]*x*y2*z + a[1][2][2]*x*y2*z2 + a[1][2][3]*x*y2*z3 + \
        a[1][3][0]*x*y3 + a[1][3][1]*x*y3*z + a[1][3][2]*x*y3*z2 + a[1][3][3]*x*y3*z3 + \
        a[2][0][0]*x2 + a[2][0][1]*x2*z + a[2][0][2]*x2*z2 + a[2][0][3]*x2*z3 + \
        a[2][1][0]*x2*y + a[2][1][1]*x2*y*z + a[2][1][2]*x2*y*z2 + a[2][1][3]*x2*y*z3 + \
        a[2][2][0]*x2*y2 + a[2][2][1]*x2*y2*z + a[2][2][2]*x2*y2*z2 + a[2][2][3]*x2*y2*z3 + \
        a[2][3][0]*x2*y3 + a[2][3][1]*x2*y3*z + a[2][3][2]*x2*y3*z2 + a[2][3][3]*x2*y3*z3 + \
        a[3][0][0]*x3 + a[3][0][1]*x3*z + a[3][0][2]*x3*z2 + a[3][0][3]*x3*z3 + \
        a[3][1][0]*x3*y + a[3][1][1]*x3*y*z + a[3][1][2]*x3*y*z2 + a[3][1][3]*x3*y*z3 + \
        a[3][2][0]*x3*y2 + a[3][2][1]*x3*y2*z + a[3][2][2]*x3*y2*z2 + a[3][2][3]*x3*y2*z3 + \
        a[3][3][0]*x3*y3 + a[3][3][1]*x3*y3*z + a[3][3][2]*x3*y3*z2 + a[3][3][3]*x3*y3*z3