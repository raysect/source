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

from numpy cimport ndarray

cdef class SpectralFunction:

    cpdef double sample_single(self, double min_wavelength, double max_wavelength)

    cpdef SampledSF sample_multiple(self, double min_wavelength, double max_wavelength, int num_samples)


cdef class SampledSF(SpectralFunction):

    cdef:
        readonly double min_wavelength
        readonly double max_wavelength
        readonly int num_samples
        readonly double delta_wavelength
        public ndarray samples
        ndarray _wavelengths
        public bint fast_sample

    cdef inline void _construct(self, double min_wavelength, double max_wavelength, int num_samples, bint fast_sample)

    cdef inline void _populate_wavelengths(self)

    cpdef bint is_shaped(self, double min_wavelength, double max_wavelength, int num_samples)

    cdef inline void add_scalar(self, double value)
    cdef inline void sub_scalar(self, double value)
    cdef inline void mul_scalar(self, double value)
    cdef inline void div_scalar(self, double value)

    cdef inline void add_array(self, double[::1] array)
    cdef inline void sub_array(self, double[::1] array)
    cdef inline void mul_array(self, double[::1] array)
    cdef inline void div_array(self, double[::1] array)


cdef SampledSF new_sampledsf(double min_wavelength, double max_wavelength, int num_samples)


cdef class InterpolatedSF(SpectralFunction):

    cdef:
        public ndarray wavelengths
        public ndarray samples
        public bint fast_sample


cdef class ConstantSF(SpectralFunction):

    cdef:
        readonly double value
        SampledSF cached_samples