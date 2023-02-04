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

from numpy cimport ndarray

cdef class SpectralFunction:

    cdef:
        double _average_cache
        double _average_cache_min_wvl
        double _average_cache_max_wvl

        ndarray _sample_cache
        double[::1] _sample_cache_mv
        double _sample_cache_min_wvl
        double _sample_cache_max_wvl
        int _sample_cache_num_samp

    cpdef double evaluate(self, double wavelength)
    cpdef double integrate(self, double min_wavelength, double max_wavelength)
    cpdef double average(self, double min_wavelength, double max_wavelength)
    cpdef ndarray sample(self, double min_wavelength, double max_wavelength, int bins)
    cdef double[::1] sample_mv(self, double min_wavelength, double max_wavelength, int bins)

    cdef void _average_cache_init(self)
    cdef bint _average_cache_valid(self, double min_wavelength, double max_wavelength)
    cdef double _average_cache_get(self)
    cdef void _average_cache_set(self, double min_wavelength, double max_wavelength, double average)

    cdef void _sample_cache_init(self)
    cdef bint _sample_cache_valid(self, double min_wavelength, double max_wavelength, int bins)
    cdef ndarray _sample_cache_get_array(self)
    cdef double[::1] _sample_cache_get_mv(self)
    cdef void _sample_cache_set(self, double min_wavelength, double max_wavelength, int bins, ndarray samples, double[::1] samples_mv)


cdef class NumericallyIntegratedSF(SpectralFunction):

    cdef readonly double sample_resolution

    cpdef double integrate(self, double min_wavelength, double max_wavelength)
    cpdef double function(self, double wavelength)


cdef class InterpolatedSF(SpectralFunction):

    cdef:
        ndarray wavelengths
        ndarray samples
        double[::1] wavelengths_mv
        double[::1] samples_mv


cdef class ConstantSF(SpectralFunction):

    cdef readonly double value
