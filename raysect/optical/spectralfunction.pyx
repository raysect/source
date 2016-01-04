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
from raysect.core.math.utility cimport interpolate, integrate
from numpy cimport PyArray_SimpleNew, PyArray_FILLWBYTE, NPY_FLOAT64, npy_intp, import_array
from numpy import array, float64, argsort

# required by numpy c-api
import_array()


cdef class SpectralFunction:
    """
    Spectral function base class.
    """

    cpdef double sample_single(self, double min_wavelength, double max_wavelength):

        return NotImplemented

    cpdef ndarray sample_multiple(self, double min_wavelength, double max_wavelength, int num_samples):

        return NotImplemented


cdef class InterpolatedSF(SpectralFunction):
    """
    Linearly interpolated spectral function.

    spectral function defined by samples of regular or irregular spacing

    ends are extrapolated. must set ends to zero if you want function to end!
    """

    def __init__(self, object wavelengths, object samples, fast_sample=False):
        """
        wavelengths and samples will be sorted during initialisation.

        :param wavelengths: 1D array of wavelengths in nanometers.
        :param samples: 1D array of spectral samples.
        """

        self.wavelengths = array(wavelengths, dtype=float64)
        self.samples = array(samples, dtype=float64)
        self.fast_sample = fast_sample

        if self.wavelengths.ndim != 1:

            raise ValueError("Wavelength array must be 1D.")

        if self.samples.shape[0] != self.wavelengths.shape[0]:

            raise ValueError("Wavelength and sample arrays must be the same length.")

        # sort arrays by increasing wavelength
        indicies = argsort(self.wavelengths)
        self.wavelengths = self.wavelengths[indicies]
        self.samples = self.samples[indicies]

        # initialise cache with invalid values
        self.cache_samples = None
        self.cache_min_wavelength = -1
        self.cache_max_wavelength = -1
        self.cache_num_samples = -1

    def __call__(self, double wavelength):

        return interpolate(self.wavelengths, self.samples, wavelength)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double sample_single(self, double min_wavelength, double max_wavelength):

        if self.fast_sample:

            # sample data at bin centre by linearly interpolating
            return interpolate(self.wavelengths, self.samples, 0.5 * (min_wavelength + max_wavelength))

        else:

            # average value obtained by integrating linearly interpolated data and normalising
            return integrate(self.wavelengths, self.samples, min_wavelength, max_wavelength) / (max_wavelength - min_wavelength)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef ndarray sample_multiple(self, double min_wavelength, double max_wavelength, int num_samples):

        cdef:
            ndarray samples
            double[::1] s_view
            npy_intp size, index
            double lower_wavelength, upper_wavelength, delta_wavelength, reciprocal

        # is cached data available to return?
        if self.cache_samples is not None and \
            self.cache_min_wavelength == min_wavelength and \
            self.cache_max_wavelength == max_wavelength and \
            self.cache_num_samples == num_samples:

            return self.cache_samples

        # create new sample ndarray and obtain a memoryview for fast access
        size = num_samples
        samples = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
        PyArray_FILLWBYTE(samples, 0)
        s_view = samples

        delta_wavelength = (max_wavelength - min_wavelength) / num_samples
        if self.fast_sample:

            # sample data at bin centre by linearly interpolating
            for index in range(num_samples):

                centre_wavelength = min_wavelength + (0.5 + index) * delta_wavelength
                s_view[index] = interpolate(self.wavelengths, self.samples, centre_wavelength)

        else:

            # re-sample by averaging data across each bin
            lower_wavelength = min_wavelength
            reciprocal = 1.0 / delta_wavelength
            for index in range(num_samples):

                upper_wavelength = min_wavelength + (index + 1) * delta_wavelength

                # average value obtained by integrating linearly interpolated data and normalising
                s_view[index] = reciprocal * integrate(self.wavelengths, self.samples, lower_wavelength, upper_wavelength)

                lower_wavelength = upper_wavelength

        # update cache
        self.cache_samples = samples
        self.cache_min_wavelength = min_wavelength
        self.cache_max_wavelength = max_wavelength
        self.cache_num_samples = num_samples

        return samples


cdef class ConstantSF(SpectralFunction):
    """
    Constant value spectral function
    """

    def __init__(self, double value):

        self.value = value

        # initialise cache with invalid values
        self.cache_samples = None
        self.cache_min_wavelength = -1
        self.cache_max_wavelength = -1
        self.cache_num_samples = -1

    cpdef double sample_single(self, double min_wavelength, double max_wavelength):

        return self.value

    cpdef ndarray sample_multiple(self, double min_wavelength, double max_wavelength, int num_samples):

        cdef:
            ndarray samples
            npy_intp size
            double[::1] s_view

        # is cached data available to return?
        if self.cache_samples is not None and \
            self.cache_min_wavelength == min_wavelength and \
            self.cache_max_wavelength == max_wavelength and \
            self.cache_num_samples == num_samples:

            return self.cache_samples

        # create new sample ndarray and obtain a memoryview for fast access
        size = num_samples
        samples = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
        PyArray_FILLWBYTE(samples, 0)
        s_view = samples

        # generate samples
        s_view[:] = self.value

        # update cache
        self.cache_samples = samples
        self.cache_min_wavelength = min_wavelength
        self.cache_max_wavelength = max_wavelength
        self.cache_num_samples = num_samples

        return samples
