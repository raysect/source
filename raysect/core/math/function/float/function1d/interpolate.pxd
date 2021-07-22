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

from raysect.core.math.function.float.function1d.base cimport Function1D
from numpy cimport ndarray


cdef class Interpolate1DArray(Function1D):
    cdef:
        ndarray x, f
        double[::1] _x_mv, _f_mv
        _Interpolator1D _interpolator
        _Extrapolator1D _extrapolator
        int _last_index
        double _extrapolation_range


cdef class _Interpolator1D:
    cdef:
        double[::1] _x, _f
        int _last_index

    cdef double evaluate(self, double px, int index) except? -1e999
    cdef double _analytic_gradient(self, double px, int index, int order)


cdef class _Interpolator1DLinear(_Interpolator1D):
    pass


cdef class _Interpolator1DCubic(_Interpolator1D):
    cdef:
        ndarray _a, _mask_a
        double[:, ::1] _a_mv
        int _n
        double evaluate(self, double px, int index) except? -1e999


cdef class _Extrapolator1D:
    cdef:
        double [::1] _x, _f
        int _last_index
    cdef double _analytic_gradient(self, double px, int index, int order)
    cdef double evaluate(self, double px, int index) except? -1e999


cdef class _Extrapolator1DNone(_Extrapolator1D):
    pass


cdef class _Extrapolator1DNearest(_Extrapolator1D):
    pass


cdef class _Extrapolator1DLinear(_Extrapolator1D):
    pass


cdef class _Extrapolator1DQuadratic(_Extrapolator1D):
    cdef double[3] _a_first, _a_last
    cdef int[2] _mask_a
    cdef void _calculate_quadratic_coefficients_start(self, double f1, double df1_dx, double df2_dx, double[3] a)
    cdef void _calculate_quadratic_coefficients_end(self, double f2, double df1_dx, double df2_dx, double[3] a)


cdef class _ArrayDerivative1D:
    cdef:
        double [::1] _x, _f
        int _last_index_x

    cdef double evaluate(self, int index_x, int derivative_order_x, bint rescale_norm_x) except? -1e999
    cdef double _evaluate_edge_x(self, int index_x, int derivative_order_x)
    cdef double _evaluate_x(self, int index_x, int derivative_order_x)

