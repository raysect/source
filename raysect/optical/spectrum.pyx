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

from numpy cimport import_array, PyArray_SimpleNew, PyArray_FILLWBYTE, NPY_FLOAT64, npy_intp
cimport cython

# required by numpy c-api
import_array()

cdef class Waveband:
    """
    waveband: [min_wavelength, max_wavelength)
    """

    def __init__(self, double min_wavelength, double max_wavelength):

        if min_wavelength <= 0.0 or max_wavelength <= 0.0:

            raise ValueError("Wavelength can not be less than or equal to zero.")

        if min_wavelength >= max_wavelength:

            raise ValueError("Minimum wavelength can not be greater or eaual to the maximum wavelength.")

        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength

    property min_wavelength:

        def __get__(self):

            return self._min_wavelength

        def __set__(self, double wavelength):

            if wavelength <= 0.0:

                raise ValueError("Wavelength can not be less than or equal to zero.")

            if wavelength >= self._max_wavelength:

                raise ValueError("Minimum wavelength can not be greater than or equal to the maximum wavelength.")

            self._min_wavelength = wavelength

    property max_wavelength:

        def __get__(self):

            return self._max_wavelength

        def __set__(self, double wavelength):

            if wavelength <= 0.0:

                raise ValueError("Wavelength can not be less than or equal to zero.")

            if self._min_wavelength >= wavelength:

                raise ValueError("Maximum wavelength can not be less than or equal to the minimum wavelength.")

            self._max_wavelength = wavelength

    cpdef Waveband copy(self):

        return new_waveband(self._min_wavelength, self._max_wavelength)

    cdef inline double get_min_wavelength(self):

        return self._min_wavelength

    cdef inline double get_max_wavelength(self):

        return self._max_wavelength


cpdef ndarray new_spectrum_array(Ray ray):

    cdef ndarray array
    cdef npy_intp size

    size = ray.get_waveband_count()
    array = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
    PyArray_FILLWBYTE(array, 0)

    return array
