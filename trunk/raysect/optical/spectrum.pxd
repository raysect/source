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

from numpy cimport ndarray, import_array, PyArray_SimpleNew, PyArray_FILLWBYTE, NPY_FLOAT64, npy_intp
from raysect.optical.ray cimport Ray

cdef class Spectrum:

    cdef readonly double min_wavelength
    cdef readonly double max_wavelength
    cdef readonly double delta_wavelength
    cdef readonly int samples
    cdef readonly ndarray bins
    cdef ndarray _wavelengths

    cdef inline void _construct(self, double min_wavelength, double max_wavelength, int samples)

    cdef inline void add_scalar(self, double value)
    cdef inline void sub_scalar(self, double value)
    cdef inline void mul_scalar(self, double value)
    cdef inline void div_scalar(self, double value)

    cdef inline void add_array(self, double[::1] array)
    cdef inline void sub_array(self, double[::1] array)
    cdef inline void mul_array(self, double[::1] array)
    cdef inline void div_array(self, double[::1] array)


cdef Spectrum new_spectrum(double min_wavelength, double max_wavelength, int samples)


cpdef double photon_energy(double wavelength)
