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

cimport cython
from raysect.core.math.function.float.function1d.base cimport Function1D

from numpy cimport ndarray

DEF INT_LINEAR = 0
DEF INT_CUBIC = 1
DEF INT_CUBIC_CONSTRAINED = 2

_INTERPOLATION_TYPES = {
    'linear': INT_LINEAR,
    'cubic': INT_CUBIC,
    'cubic_constrained': INT_CUBIC_CONSTRAINED
}

DEF EXT_NONE = 0
DEF EXT_NEAREST = 1
DEF EXT_LINEAR = 2
DEF EXT_QUADRATIC = 3

_EXTRAPOLATION_TYPES = {
    'none': EXT_NONE,
    'nearest': EXT_NEAREST,
    'linear': EXT_LINEAR,
    'quadratic': EXT_QUADRATIC
}


cpdef enum InterpType:
    LinearInt = 1
    CubicInt = 2
    CubicConstrainedInt = 3

cpdef enum ExtrapType:
    NoExt = 1
    NearestExt = 2
    LinearExt = 3
    QuadraticExt = 4


cdef class Interpolate1D(Function1D):
    cdef:
        ndarray x, f
        double[::1] _x, _f
        _Interpolator1D _interpolator
        _Extrapolator1D _extrapolator
        int _last_index
        double _extrapolation_range


cdef class _Interpolator1D:
    cdef:
        double[::1] _x, _f

    cdef double evaluate(self, double px, int index) except? -1e999


cdef class _Interpolator1DLinear(_Interpolator1D):
    pass


cdef class _Interpolator1DCubic(_Interpolator1D):
    cdef:
        ndarray _a, _mask_a
        double[:, ::1] _a_mv
        int _n
        double evaluate(self, double px, int index) except? -1e999
        double get_gradient(self, double[::1] x_spline, double[::1] y_spline, int index)


cdef class _Interpolator1DCubicConstrained(_Interpolator1DCubic):
    cdef double get_gradient(self, double[::1] x_spline, double[::1] y_spline, int index)


cdef class _Extrapolator1D:
    cdef:
        double _range
        double [::1] _x, _f
        int _last_index

    cdef double extrapolate(self, double px, int order, int index, double rx) except? -1e999
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
    cdef calculate_quadratic_coefficients(self, double f1, double f2, double f3, double x_scal_3, double[3] a)



