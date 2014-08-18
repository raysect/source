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

cimport cython

# Plank's constant * speed of light in a vacuum
DEF CONSTANT_HC = 1.9864456832693028e-25

# required by numpy c-api
import_array()

cdef class Spectrum:

    def __init__(self, double min_wavelength, double max_wavelength, int samples):

        if samples < 1:

                raise("Number of samples can not be less than 1.")

        if min_wavelength <= 0.0 or max_wavelength <= 0.0:

            raise ValueError("Wavelength can not be less than or equal to zero.")

        if min_wavelength >= max_wavelength:

            raise ValueError("Minimum wavelength can not be greater or equal to the maximum wavelength.")

        self._construct(min_wavelength, max_wavelength, samples)

    property wavelengths:

        @cython.boundscheck(False)
        @cython.wraparound(False)
        def __get__(self):

            cdef:
                npy_intp size
                int index
                double[::1] w_view

            if self._wavelengths is None:

                # create and populate central wavelength array
                size = self.samples
                self._wavelengths = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
                w_view = self._wavelengths

                for index in range(self.samples):

                    w_view[index] = self.min_wavelength + index * self.delta_wavelength

            return self._wavelengths

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void _construct(self, double min_wavelength, double max_wavelength, int samples):

        cdef:
            npy_intp size, index
            double[::1] wavelengths_view

        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.delta_wavelength = (max_wavelength - min_wavelength) / samples
        self.samples = samples

        # create spectral sample bins, initialise with zero
        size = self.samples
        self.bins = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
        PyArray_FILLWBYTE(self.bins, 0)

        # wavelengths is populated on demand
        self._wavelengths = None

    # low level scalar maths functions
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void add_scalar(self, double value):

        cdef:
            double[::1] bins_view
            npy_intp index

        bins_view = self.bins
        for index in range(bins_view.shape[0]):

            bins_view[index] += value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void sub_scalar(self, double value):

        cdef:
            double[::1] bins_view
            npy_intp index

        bins_view = self.bins
        for index in range(bins_view.shape[0]):

            bins_view[index] -= value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void mul_scalar(self, double value):

        cdef:
            double[::1] bins_view
            npy_intp index

        bins_view = self.bins
        for index in range(bins_view.shape[0]):

            bins_view[index] *= value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void div_scalar(self, double value):

        cdef:
            double[::1] bins_view
            double reciprocal
            npy_intp index

        bins_view = self.bins
        reciprocal = 1.0 / value
        for index in range(bins_view.shape[0]):

            bins_view[index] *= reciprocal

    # low level array maths functions
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void add_array(self, double[::1] array):

        cdef:
            double[::1] bins_view
            npy_intp index

        bins_view = self.bins
        for index in range(bins_view.shape[0]):

            bins_view[index] += array[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void sub_array(self, double[::1] array):

        cdef:
            double[::1] bins_view
            npy_intp index

        bins_view = self.bins
        for index in range(bins_view.shape[0]):

            bins_view[index] -= array[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void mul_array(self, double[::1] array):

        cdef:
            double[::1] bins_view
            npy_intp index

        bins_view = self.bins
        for index in range(bins_view.shape[0]):

            bins_view[index] *= array[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void div_array(self, double[::1] array):

        cdef:
            double[::1] bins_view
            npy_intp index

        bins_view = self.bins
        for index in range(bins_view.shape[0]):

            bins_view[index] /= array[index]


cdef Spectrum new_spectrum(double min_wavelength, double max_wavelength, int samples):

    cdef Spectrum v

    v = Spectrum.__new__(Spectrum)
    v._construct(min_wavelength, max_wavelength, samples)

    return v


cpdef double photon_energy(double wavelength):
    """
    Returns the energy of a photon with the specified wavelength.

    Arguements:
        wavelength: photon wavelength in nanometers

    Returns:
        photon energy in Joules
    """

    with cython.cdivision:

        # h * c / lambda
        return CONSTANT_HC / (wavelength * 1e-9)
