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

from raysect.optical.spectralfunction cimport SpectralFunction
from numpy cimport ndarray

cdef class Spectrum(SpectralFunction):

    cdef:
        readonly double min_wavelength
        readonly double max_wavelength
        readonly int bins
        readonly double delta_wavelength
        readonly ndarray samples
        ndarray _wavelengths
        double[::1] samples_mv

    cdef void _wavelength_check(self, double min_wavelength, double max_wavelength)
    cdef void _attribute_check(self)
    cdef void _construct(self, double min_wavelength, double max_wavelength, int bins)
    cdef void _populate_wavelengths(self)

    cpdef bint is_compatible(self, double min_wavelength, double max_wavelength, int bins)
    cpdef bint is_zero(self)
    cpdef double total(self)
    cpdef ndarray to_photons(self)
    cpdef void clear(self)
    cpdef Spectrum new_spectrum(self)
    cpdef Spectrum copy(self)

    cdef void add_scalar(self, double value) nogil
    cdef void sub_scalar(self, double value) nogil
    cdef void mul_scalar(self, double value) nogil
    cdef void div_scalar(self, double value) nogil

    cdef void add_array(self, double[::1] array) nogil
    cdef void sub_array(self, double[::1] array) nogil
    cdef void mul_array(self, double[::1] array) nogil
    cdef void div_array(self, double[::1] array) nogil

    cdef void add_spectrum(self, Spectrum spectrum) nogil
    cdef void sub_spectrum(self, Spectrum spectrum) nogil
    cdef void mul_spectrum(self, Spectrum spectrum) nogil
    cdef void div_spectrum(self, Spectrum spectrum) nogil

    cdef void mad_scalar(self, double scalar, double[::1] array) nogil
    cdef void mad_array(self, double[::1] a, double[::1] b) nogil


cdef Spectrum new_spectrum(double min_wavelength, double max_wavelength, int bins)


cpdef double photon_energy(double wavelength) except -1
