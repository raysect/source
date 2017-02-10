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
from raysect.core.math.cython cimport integrate, interpolate
from numpy cimport PyArray_SimpleNew, PyArray_FILLWBYTE, NPY_FLOAT64, npy_intp, import_array

# Plank's constant * speed of light in a vacuum
DEF CONSTANT_HC = 1.9864456832693028e-25

# required by numpy c-api
import_array()


cdef class Spectrum(SpectralFunction):
    """
    radiance units: W/m^2/str/nm

    Regularly spaced samples.

    Used internally by the raytracer.

    samples lie in centre of wavelength bins.

    """

    def __init__(self, double min_wavelength, double max_wavelength, int bins):

        self._wavelength_check(min_wavelength, max_wavelength)

        if bins < 1:
            raise ValueError("Number of bins cannot be less than 1.")

        self._construct(min_wavelength, max_wavelength, bins)

    cdef inline void _wavelength_check(self, double min_wavelength, double max_wavelength):

        if min_wavelength <= 0.0 or max_wavelength <= 0.0:
            raise ValueError("Wavelength cannot be less than or equal to zero.")

        if min_wavelength >= max_wavelength:
            raise ValueError("Minimum wavelength cannot be greater or equal to the maximum wavelength.")

    cdef inline void _attribute_check(self):

        # users can modify the sample array, need to prevent segfaults in cython code
        if self.samples is None:
            raise ValueError("Cannot generate sample as the sample array is None.")

        if self.samples.shape[0] != self.bins:
            raise ValueError("Sample array length is inconsistent with the number of bins.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void _construct(self, double min_wavelength, double max_wavelength, int bins):

        cdef:
            npy_intp size, index
            double[::1] wavelengths_view

        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.bins = bins
        self.delta_wavelength = (max_wavelength - min_wavelength) / bins

        # create spectral sample bins, initialise with zero
        size = bins
        self.samples = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
        PyArray_FILLWBYTE(self.samples, 0)

        # obtain memory view
        self.samples_mv = self.samples

        # wavelengths is populated on demand
        self._wavelengths = None

    property wavelengths:

        def __get__(self):

            self._populate_wavelengths()
            return self._wavelengths

    def __len__(self):

        return self.bins

    def __getstate__(self):
        """Encodes state for pickling."""

        return (
            self.min_wavelength,
            self.max_wavelength,
            self.bins,
            self.delta_wavelength,
            self.samples,
        )

    def __setstate__(self, state):
        """Decodes state for pickling."""

        (self.min_wavelength,
         self.max_wavelength,
         self.bins,
         self.delta_wavelength,
         self.samples) = state

        self._wavelengths = None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void _populate_wavelengths(self):

        cdef:
            npy_intp size
            int index
            double[::1] w_view

        if self._wavelengths is None:

            # create and populate central wavelength array
            size = self.bins
            self._wavelengths = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
            w_view = self._wavelengths

            for index in range(self.bins):
                w_view[index] = self.min_wavelength + (0.5 + index) * self.delta_wavelength

    cpdef bint is_compatible(self, double min_wavelength, double max_wavelength, int bins):
        """
        Returns True if the stored samples are consistent with the specified
        wavelength range and sample size.

        :param min_wavelength: The minimum wavelength in nanometers.
        :param max_wavelength: The maximum wavelength in nanometers
        :param bins: The number of bins.
        :return: True if the samples are compatible with the range/samples, False otherwise.
        :rtype: boolean
        """

        return self.min_wavelength == min_wavelength and \
               self.max_wavelength == max_wavelength and \
               self.bins == bins

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double average(self, double min_wavelength, double max_wavelength):

        self._wavelength_check(min_wavelength, max_wavelength)
        self._attribute_check()

        # require wavelength information for this calculation
        self._populate_wavelengths()

        # average value obtained by integrating linearly interpolated data and normalising
        return integrate(self._wavelengths, self.samples, min_wavelength, max_wavelength) / (max_wavelength - min_wavelength)

    cpdef double integrate(self, double min_wavelength, double max_wavelength):
        """
        Calculates the radiance over the specified spectral range.

        :param min_wavelength:
        :param max_wavelength:
        :return: Radiance in W/m^2/str
        """

        self._wavelength_check(min_wavelength, max_wavelength)
        self._attribute_check()

        # this calculation requires the wavelength array
        self._populate_wavelengths()

        return integrate(self._wavelengths, self.samples, min_wavelength, max_wavelength)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef ndarray sample(self, double min_wavelength, double max_wavelength, int bins):

        cdef:
            ndarray samples
            double[::1] s_view
            npy_intp size, index
            double lower_wavelength, upper_wavelength, centre_wavelength, delta_wavelength, reciprocal

        self._wavelength_check(min_wavelength, max_wavelength)
        self._attribute_check()

        # create new sample object and obtain a memoryview for fast access
        size = bins
        samples = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
        PyArray_FILLWBYTE(samples, 0)
        s_view = samples

        # require wavelength information for this calculation
        self._populate_wavelengths()

        delta_wavelength = (max_wavelength - min_wavelength) / bins

        # re-sample by averaging data across each bin
        lower_wavelength = min_wavelength
        reciprocal = 1.0 / delta_wavelength
        for index in range(bins):

            # average value obtained by integrating linearly interpolated data and normalising
            upper_wavelength = min_wavelength + (index + 1) * delta_wavelength
            s_view[index] = reciprocal * integrate(self._wavelengths, self.samples, lower_wavelength, upper_wavelength)
            lower_wavelength = upper_wavelength

        return samples

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint is_zero(self):
        """
        Can be used to determine if all the samples are zero.

        :return: True if the spectrum is zero, False otherwise.
        """

        cdef int index

        self._attribute_check()

        for index in range(self.bins):
            if self.samples_mv[index] != 0.0:
                return False
        return True

    cpdef double total(self):
        """
        Calculates the radiance over the sampled spectral range.

        :return: Radiance in W/m^2/str
        """

        self._attribute_check()

        # this calculation requires the wavelength array
        self._populate_wavelengths()

        return integrate(self._wavelengths, self.samples, self.min_wavelength, self.max_wavelength)

    cpdef ndarray to_photons(self):
        """
        Converts the spectrum sample array from W/m^2/str/nm to Photons/s/m^2/str/nm
        and returns the data in a numpy array.

        :return: A numpy array containing the spectral samples converted to ph/s/m^2/str/nm.
        """

        cdef:
            npy_intp size
            int index
            ndarray photons
            double[::1] photons_view

        self._attribute_check()

        # this calculation requires the wavelength array
        self._populate_wavelengths()

        # create array to hold photon samples
        size = self.bins
        photons = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
        photons_view = photons

        # convert each sample to photons
        for index in range(self.bins):
            photons_view[index] = self.samples_mv[index] / photon_energy(self._wavelengths[index])

        return photons

    cpdef Spectrum new_spectrum(self):
        """
        Returns a new Spectrum compatible with the same spectral settings.

        :return: A new Spectrum object.
        """

        return new_spectrum(self.min_wavelength, self.max_wavelength, self.bins)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum copy(self):
        """
        Returns a copy of the spectrum.

        :return: A new Spectrum object.
        """

        cdef:
            Spectrum spectrum
            npy_intp index

        spectrum = self.new_spectrum()
        for index in range(self.samples_mv.shape[0]):
            spectrum.samples_mv[index] = self.samples_mv[index]

        return spectrum

    # low level scalar maths functions
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void add_scalar(self, double value):

        cdef npy_intp index
        for index in range(self.samples_mv.shape[0]):
            self.samples_mv[index] += value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void sub_scalar(self, double value):

        cdef npy_intp index
        for index in range(self.samples_mv.shape[0]):
            self.samples_mv[index] -= value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void mul_scalar(self, double value):

        cdef npy_intp index
        for index in range(self.samples_mv.shape[0]):
            self.samples_mv[index] *= value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void div_scalar(self, double value):

        cdef:
            double reciprocal
            npy_intp index

        reciprocal = 1.0 / value
        for index in range(self.samples_mv.shape[0]):
            self.samples_mv[index] *= reciprocal

    # low level array maths functions
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void add_array(self, double[::1] array):

        cdef npy_intp index
        for index in range(self.samples_mv.shape[0]):
            self.samples_mv[index] += array[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void sub_array(self, double[::1] array):

        cdef npy_intp index
        for index in range(self.samples_mv.shape[0]):
            self.samples_mv[index] -= array[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void mul_array(self, double[::1] array):

        cdef npy_intp index
        for index in range(self.samples_mv.shape[0]):
            self.samples_mv[index] *= array[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void div_array(self, double[::1] array):

        cdef npy_intp index
        for index in range(self.samples_mv.shape[0]):
            self.samples_mv[index] /= array[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void mad_scalar(self, double scalar, double[::1] array):

        cdef npy_intp index
        for index in range(self.samples_mv.shape[0]):
            self.samples_mv[index] += scalar * array[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void mad_array(self, double[::1] a, double[::1] b):

        cdef npy_intp index
        for index in range(self.samples_mv.shape[0]):
            self.samples_mv[index] += a[index] * b[index]


cdef Spectrum new_spectrum(double min_wavelength, double max_wavelength, int bins):

    cdef Spectrum v

    v = Spectrum.__new__(Spectrum)
    v._construct(min_wavelength, max_wavelength, bins)

    return v


@cython.cdivision(True)
cpdef double photon_energy(double wavelength) except -1:
    """
    Returns the energy of a photon with the specified wavelength.

    :param wavelength: Photon wavelength in nanometers.
    :return: Photon energy in Joules.
    """

    if wavelength <= 0.0:
        raise ValueError("Wavelength must be greater than zero.")

    # h * c / lambda
    return CONSTANT_HC / (wavelength * 1e-9)

