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

# todo: add getters and setters for arrays of stokes components
# todo: add maths utilities to simplify update of each stokes component e.g. mul_i_array, mul_q_scalar, mad_q_scalar etc...
#  or use an enumeration for the first array index e.g. mul_array(Q, ....) where I=0, Q=1, etc... or better yest just use s0-s3 notation and an index
# todo: add utility functions to check the stokes vector validity (i >= sqrt(q**2...))

cimport cython
from raysect.optical.stokes cimport StokesVector, new_stokesvector
from numpy cimport PyArray_SimpleNew, PyArray_FILLWBYTE, NPY_FLOAT64, npy_intp, import_array

# Plank's constant * speed of light in a vacuum
DEF CONSTANT_HC = 1.9864456832693028e-25

# required by numpy c-api
import_array()

# todo: only the most rudimentary functions have been implemented for now, these will be fleshed out before the v1 release
# todo: for v1 rewrite using array.array or malloc/free
@cython.freelist(256)
cdef class Spectrum:
    # todo: update docstring
    """
    A class for working with polarised spectra.

    Each column of the spectrum defines a Stokes vector.

    Describes the distribution of light at each wavelength in units of radiance (W/m^2/str/nm).
    Spectral samples are regularly spaced over the wavelength range and lie in the centre of
    the wavelength bins.

    :param float min_wavelength: Lower wavelength bound for this spectrum
    :param float max_wavelength: Upper wavelength bound for this spectrum
    :param int bins: Number of samples to use over the spectral range

    .. code-block:: pycon

        >>> from raysect.optical.polarised import Spectrum
        >>>
        >>> spectrum = Spectrum(400, 720, 250)
    """

    def __init__(self, double min_wavelength, double max_wavelength, int bins):

        self._wavelength_check(min_wavelength, max_wavelength)

        if bins < 1:
            raise ValueError("Number of bins cannot be less than 1.")

        self._construct(min_wavelength, max_wavelength, bins)

    def __getstate__(self):
        """Encodes state for pickling."""

        return (
            self.min_wavelength,
            self.max_wavelength,
            self.bins,
            self.delta_wavelength,
            self.samples,
            self._wavelengths,
            super().__getstate__()
        )

    def __setstate__(self, state):
        """Decodes state for pickling."""

        (
            self.min_wavelength,
            self.max_wavelength,
            self.bins,
            self.delta_wavelength,
            self.samples,
            self._wavelengths,
            super_state
        ) = state
        super().__setstate__(super_state)

        # rebuild memory views
        self.samples_mv = self.samples

    # must override automatic __reduce__ method generated by cython for the base class
    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    cdef void _wavelength_check(self, double min_wavelength, double max_wavelength):

        if min_wavelength <= 0.0 or max_wavelength <= 0.0:
            raise ValueError("Wavelength cannot be less than or equal to zero.")

        if min_wavelength >= max_wavelength:
            raise ValueError("Minimum wavelength cannot be greater or equal to the maximum wavelength.")

    cdef void _attribute_check(self):

        # users can modify the sample array, need to prevent segfaults in cython code
        if self.samples is None:
            raise ValueError("Cannot generate sample as the sample array is None.")

        if self.samples.ndim != 2:
            raise ValueError("The sample array must have the dimensions (n, 4), where n is the number of bins.")

        if self.samples.shape[0] != self.bins:
            raise ValueError("The sample array length is inconsistent with the number of bins.")

        if self.samples.shape[1] != 4:
            raise ValueError("The sample array must be 4 elements wide.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _construct(self, double min_wavelength, double max_wavelength, int bins):

        cdef:
            npy_intp size[2]
            npy_intp index

        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.bins = bins
        self.delta_wavelength = (max_wavelength - min_wavelength) / bins

        # create sample bins for Stoke's vectors, initialise with zero
        size[0] = bins
        size[1] = 4
        self.samples = PyArray_SimpleNew(2, size, NPY_FLOAT64)
        PyArray_FILLWBYTE(self.samples, 0)

        # obtain memory view
        self.samples_mv = self.samples

        # wavelengths is populated on demand
        self._wavelengths = None

    @property
    def wavelengths(self):
        """
        Wavelength array in nm

        :rtype: ndarray
        """

        self._populate_wavelengths()
        return self._wavelengths

    def __len__(self):
        """
        The number of spectral bins

        :rtype: int
        """

        return self.bins

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _populate_wavelengths(self):

        cdef:
            npy_intp size
            int index
            double[::1] wavelengths_mv

        if self._wavelengths is None:

            # create and populate central wavelength array
            size = self.bins
            self._wavelengths = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
            wavelengths_mv = self._wavelengths
            for index in range(self.bins):
                wavelengths_mv[index] = self.min_wavelength + (0.5 + index) * self.delta_wavelength

    cpdef bint is_compatible(self, double min_wavelength, double max_wavelength, int bins):
        """
        Returns True if the stored samples are consistent with the specified
        wavelength range and sample size.

        :param float min_wavelength: The minimum wavelength in nanometers
        :param float max_wavelength: The maximum wavelength in nanometers
        :param int bins: The number of bins.
        :return: True if the samples are compatible with the range/samples, False otherwise.
        :rtype: boolean
        """

        return self.min_wavelength == min_wavelength and \
               self.max_wavelength == max_wavelength and \
               self.bins == bins

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef StokesVector as_stokes(self, npy_intp index):
        """
        Returns a StokesVector for the specified bin.

        :param index: The bin index 
        :return: A StokesVector object.
        """

        if index < 0 or index > self.bins:
            raise ValueError('The bin index is out of range.')

        return new_stokesvector(
            self.samples_mv[index, 0],
            self.samples_mv[index, 1],
            self.samples_mv[index, 2],
            self.samples_mv[index, 3]
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void clear(self):
        """
        Resets the sample values in the spectrum to zero.
        """

        cdef npy_intp index
        for index in range(self.bins):
            self.samples_mv[index] = 0

    cpdef Spectrum new_spectrum(self):
        """
        Returns a new Spectrum compatible with the same spectral settings.

        :rtype: Spectrum
        """

        return new_spectrum(self.min_wavelength, self.max_wavelength, self.bins)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum copy(self):
        """
        Returns a copy of the spectrum.

        :rtype: Spectrum
        """

        cdef:
            Spectrum spectrum
            npy_intp i, j

        spectrum = self.new_spectrum()
        for i in range(self.samples_mv.shape[0]):
            for j in range(self.samples_mv.shape[1]):
                spectrum.samples_mv[i, j] = self.samples_mv[i, j]
        return spectrum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void mul_scalar(self, double value) nogil:

        cdef npy_intp i, j
        for i in range(self.samples_mv.shape[0]):
            for j in range(self.samples_mv.shape[1]):
                self.samples_mv[i, j] *= value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void div_scalar(self, double value) nogil:

        cdef:
            double reciprocal
            npy_intp i, j

        reciprocal = 1.0 / value
        for i in range(self.samples_mv.shape[0]):
            for j in range(self.samples_mv.shape[1]):
                self.samples_mv[i, j] *= reciprocal

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void mad_scalar(self, double scalar, double[:,::1] array) nogil:

        cdef npy_intp i, j
        for i in range(self.samples_mv.shape[0]):
            for j in range(self.samples_mv.shape[1]):
              self.samples_mv[i, j] += scalar * array[i, j]


cdef Spectrum new_spectrum(double min_wavelength, double max_wavelength, int bins):

    cdef Spectrum v
    v = Spectrum.__new__(Spectrum)
    v._construct(min_wavelength, max_wavelength, bins)
    return v
