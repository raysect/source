# cython: language_level=3

# Copyright (c) 2014-2025, Dr Alex Meakins, Raysect Project
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

DEF EQN_EPS = 1.0e-9

cdef int find_index(double[::1] x, double v) noexcept nogil

cdef double interpolate(double[::1] x, double[::1] y, double p) noexcept nogil

cdef double integrate(double[::1] x, double[::1] y, double x0, double x1) noexcept nogil

cdef double average(double[::1] x, double[::1] y, double x0, double x1) noexcept nogil

cdef double maximum(double[:] data) noexcept nogil

cdef double minimum(double[:] data) noexcept nogil

cdef double peak_to_peak(double[:] data) noexcept nogil

cdef inline double clamp(double v, double minimum, double maximum) noexcept nogil:
    if v < minimum:
        return minimum
    if v > maximum:
        return maximum
    return v

cdef inline void swap_double(double *a, double *b) noexcept nogil:
    cdef double temp
    temp = a[0]
    a[0] = b[0]
    b[0] = temp

cdef inline void swap_int(int *a, int *b) noexcept nogil:
    cdef int temp
    temp = a[0]
    a[0] = b[0]
    b[0] = temp

cdef inline void sort_three_doubles(double *a, double *b, double *c) noexcept nogil:
    if a[0] > b[0]:
        swap_double(a, b)
    if b[0] > c[0]:
        swap_double(b, c)
        if a[0] > c[0]:
            swap_double(a, c)

cdef inline void sort_four_doubles(double *a, double *b, double *c, double *d) noexcept nogil:
    if a[0] > b[0]:
        swap_double(a, b)
    if b[0] > c[0]:
        swap_double(b, c)
    if c[0] > d[0]:
        swap_double(c, d)
    if a[0] > b[0]:
        swap_double(a, b)
    if b[0] > c[0]:
        swap_double(b, c)
    if a[0] > b[0]:
        swap_double(a, b)

cdef inline bint is_zero(double v) noexcept nogil:
    return v < EQN_EPS and v > -EQN_EPS

@cython.cdivision(True)
cdef inline double lerp(double x0, double x1, double y0, double y1, double x) noexcept nogil:
    return ((y1 - y0) / (x1 - x0)) * (x - x0) + y0

cdef bint solve_quadratic(double a, double b, double c, double *t0, double *t1) noexcept nogil

cdef int solve_cubic(double a, double b, double c, double d, double *t0, double *t1, double *t2) noexcept nogil

cdef int solve_quartic(double a, double b, double c, double d, double e,
                       double *t0, double *t1, double *t2, double *t3) noexcept nogil

cdef bint winding2d(double[:,::1] vertices) noexcept nogil

cdef bint point_inside_polygon(double[:,::1] vertices, double ptx, double pty) noexcept

cdef int factorial(int n) noexcept