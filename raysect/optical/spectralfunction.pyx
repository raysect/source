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
from raysect.core.math.cython cimport interpolate, integrate
from numpy cimport PyArray_SimpleNew, PyArray_FILLWBYTE, NPY_FLOAT64, npy_intp, import_array
from numpy import array, float64, argsort
from libc.math cimport ceil

# required by numpy c-api
import_array()


cdef class SpectralFunction:
    """
    Spectral function base class.
    """

    def __init__(self):
        self._average_cache_init()
        self._sample_cache_init()

    cpdef double integrate(self, double min_wavelength, double max_wavelength):
        raise NotImplementedError("Virtual method integrate() not implemented.")

    @cython.cdivision(True)
    cpdef double average(self, double min_wavelength, double max_wavelength):

        # is a cached average already available?
        if self._average_cache_valid(min_wavelength, max_wavelength):
            return self._average_cache_get()

        average = self.integrate(min_wavelength, max_wavelength) / (max_wavelength - min_wavelength)

        # update cache
        self._average_cache_set(min_wavelength, max_wavelength, average)

        return average

    cdef inline void _average_cache_init(self):

        # initialise cache with invalid values
        self._average_cache = 0
        self._average_cache_min_wvl = -1
        self._average_cache_max_wvl = -1

    cdef inline bint _average_cache_valid(self, double min_wavelength, double max_wavelength):
        return (
            self._average_cache_min_wvl == min_wavelength and
            self._average_cache_max_wvl == max_wavelength
        )

    cdef inline double _average_cache_get(self):
        return self._average_cache

    cdef inline void _average_cache_set(self, double min_wavelength, double max_wavelength, double average):

        self._average_cache = average
        self._average_cache_min_wvl = min_wavelength
        self._average_cache_max_wvl = max_wavelength

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef ndarray sample(self, double min_wavelength, double max_wavelength, int num_samples):

        cdef:
            ndarray samples
            double[::1] s_view
            npy_intp size, index
            double lower, upper, delta, reciprocal

        # are cached samples already available?
        if self._sample_cache_valid(min_wavelength, max_wavelength, num_samples):
            return self._sample_cache_get()

        # create new sample ndarray and obtain a memoryview for fast access
        size = num_samples
        samples = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
        PyArray_FILLWBYTE(samples, 0)
        s_view = samples

        # re-sample by averaging data across each bin
        delta = (max_wavelength - min_wavelength) / num_samples
        lower = min_wavelength
        reciprocal = 1.0 / delta
        for index in range(num_samples):
            upper = min_wavelength + (index + 1) * delta
            s_view[index] = reciprocal * self.integrate(lower, upper)
            lower = upper

        # update cache
        self._sample_cache_set(min_wavelength, max_wavelength, num_samples, samples)

        return samples

    cdef inline void _sample_cache_init(self):

        # initialise cache with invalid values
        self._sample_cache = None
        self._sample_cache_min_wvl = -1
        self._sample_cache_max_wvl = -1
        self._sample_cache_num_samp = -1

    cdef inline bint _sample_cache_valid(self, double min_wavelength, double max_wavelength, int num_samples):

        return (
            self._sample_cache_min_wvl == min_wavelength and
            self._sample_cache_max_wvl == max_wavelength and
            self._sample_cache_num_samp == num_samples
        )

    cdef inline ndarray _sample_cache_get(self):
        return self._sample_cache

    cdef inline void _sample_cache_set(self, double min_wavelength, double max_wavelength, int num_samples, ndarray samples):

        self._sample_cache = samples
        self._sample_cache_min_wvl = min_wavelength
        self._sample_cache_max_wvl = max_wavelength
        self._sample_cache_num_samp = num_samples


cdef class NumericallyIntegratedSF(SpectralFunction):

    def __init__(self, double sample_resolution=1.0):
        super().__init__()

        if sample_resolution <= 0:
            raise ValueError("Sampling resolution must be greater than zero.")

        self.sample_resolution = sample_resolution

    cpdef double integrate(self, double min_wavelength, double max_wavelength):

        cdef:
            double delta, centre, sum
            int samples, i

        # calculate number of samples over range
        samples = <int> ceil((max_wavelength - min_wavelength) / self.sample_resolution)
        samples = max(samples, 1)

        # sample the function and integrate
        # TODO: improve this algorithm - e.g. simpsons rule
        sum = 0.0
        delta = (max_wavelength - min_wavelength) / samples
        for i in range(samples):
            centre = min_wavelength + (0.5 + i) * delta
            sum += self.function(centre) * delta

        return sum

    cpdef double function(self, double wavelength):
        raise NotImplementedError("Virtual method function() not implemented.")


cdef class InterpolatedSF(SpectralFunction):
    """
    Linearly interpolated spectral function.

    spectral function defined by samples of regular or irregular spacing

    ends are extrapolated. must set ends to zero if you want function to end!
    """

    def __init__(self, object wavelengths, object samples):
        """
        wavelengths and samples will be sorted during initialisation.

        :param wavelengths: 1D array of wavelengths in nanometers.
        :param samples: 1D array of spectral samples.
        """

        super().__init__()

        self.wavelengths = array(wavelengths, dtype=float64)
        self.samples = array(samples, dtype=float64)

        if self.wavelengths.ndim != 1:
            raise ValueError("Wavelength array must be 1D.")

        if self.samples.shape[0] != self.wavelengths.shape[0]:
            raise ValueError("Wavelength and sample arrays must be the same length.")

        # sort arrays by increasing wavelength
        indices = argsort(self.wavelengths)
        self.wavelengths = self.wavelengths[indices]
        self.samples = self.samples[indices]

    cpdef double integrate(self, double min_wavelength, double max_wavelength):
        return integrate(self.wavelengths, self.samples, min_wavelength, max_wavelength)

    def __call__(self, double wavelength):
        return interpolate(self.wavelengths, self.samples, wavelength)



cdef class ConstantSF(SpectralFunction):
    """
    Constant value spectral function
    """

    def __init__(self, double value):

        super().__init__()
        self.value = value

    cpdef double integrate(self, double min_wavelength, double max_wavelength):
        return self.value * (max_wavelength - min_wavelength)

    cpdef double average(self, double min_wavelength, double max_wavelength):
        return self.value

    cpdef ndarray sample(self, double min_wavelength, double max_wavelength, int num_samples):

        cdef:
            ndarray samples
            npy_intp size
            double[::1] s_view

        # are cached samples already available?
        if self._sample_cache_valid(min_wavelength, max_wavelength, num_samples):
            return self._sample_cache_get()

        # create new sample ndarray and obtain a memoryview for fast access
        size = num_samples
        samples = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
        PyArray_FILLWBYTE(samples, 0)
        s_view = samples

        # generate samples
        s_view[:] = self.value

        # update cache
        self._sample_cache_set(min_wavelength, max_wavelength, num_samples, samples)

        return samples
