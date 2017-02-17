# Copyright (c) 2016, Dr Alex Meakins, Raysect Project
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
cimport numpy as np
import numpy as np
from matplotlib import pyplot as plt

from raysect.optical.observer.base cimport PixelProcessor, Pipeline0D, Pipeline2D
from raysect.core.math cimport StatsArray3D, StatsArray1D
from raysect.optical.spectrum cimport Spectrum
from raysect.optical.observer.base.slice cimport SpectralSlice


_DEFAULT_PIPELINE_NAME = "Spectral Pipeline"
_DISPLAY_DPI = 100
_DISPLAY_SIZE = (800 / _DISPLAY_DPI, 600 / _DISPLAY_DPI)

# todo: add rendering display
cdef class SpectralPipeline0D(Pipeline0D):

    cdef:
        public str name
        public bint accumulate
        readonly StatsArray1D samples
        list _spectral_slices
        readonly int bins
        readonly double min_wavelength, max_wavelength, delta_wavelength
        readonly np.ndarray wavelengths
        object _display_figure

    def __init__(self, bint accumulate=True, str name=None):

        self.name = name or _DEFAULT_PIPELINE_NAME
        self.accumulate = accumulate
        self.samples = None
        self._spectral_slices = None

        self.min_wavelength = 0
        self.max_wavelength = 0
        self.bins = 0
        self.delta_wavelength = 0
        self.wavelengths = None

        self._display_figure = None

    cpdef object initialise(self, double min_wavelength, double max_wavelength, int spectral_bins, list spectral_slices):

        self._spectral_slices = spectral_slices
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.delta_wavelength = (max_wavelength - min_wavelength) / spectral_bins
        self.bins = spectral_bins
        self.wavelengths = np.array([min_wavelength + (0.5 + i) * self.delta_wavelength for i in range(spectral_bins)])

        # create samples buffer
        if not self.accumulate or self.samples is None or self.samples.length != spectral_bins:
            self.samples = StatsArray1D(spectral_bins)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int slice_id):
        return SpectralPixelProcessor(self._spectral_slices[slice_id])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object update(self, int slice_id, tuple packed_result, int pixel_samples):

        cdef:
            int index
            double[::1] mean, variance
            SpectralSlice slice

        # obtain result
        mean, variance = packed_result

        # accumulate samples
        slice = self._spectral_slices[slice_id]
        for index in range(slice.bins):
            self.samples.combine_samples(slice.offset + index, mean[index], variance[index], pixel_samples)

    cpdef object finalise(self):

        self._render_display()

        # workaround for interactivity for QT backend
        plt.pause(0.1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _render_display(self):

        cdef:
            np.ndarray errors
            double[::1] errors_mv
            int i

        errors = np.empty(self.samples.length)
        errors_mv = errors
        for i in range(self.samples.length):
            errors_mv[i] = self.samples.error(i)

        # create a fresh figure if the existing figure window has gone missing
        if not self._display_figure or not plt.fignum_exists(self._display_figure.number):
            self._display_figure = plt.figure(facecolor=(1, 1, 1), figsize=_DISPLAY_SIZE, dpi=_DISPLAY_DPI)
        fig = self._display_figure

        # set window title
        fig.canvas.set_window_title(self.name)

        fig.clf()
        plt.plot(self.wavelengths, self.samples.mean[:], color=(0, 0, 1))
        plt.plot(self.wavelengths, self.samples.mean[:] + errors[:], color=(0.685, 0.685, 1.0))
        plt.plot(self.wavelengths, self.samples.mean[:] - errors[:], color=(0.685, 0.685, 1.0))
        plt.title('{}'.format(self.name))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Spectral Power (W/nm)')
        fig.canvas.draw_idle()
        plt.show()

    def display(self):
        if not self.samples:
            raise ValueError("There is no spectrum to display.")
        self._render_display()


cdef class SpectralPipeline2D(Pipeline2D):

    cdef:
        public str name
        public bint accumulate
        readonly StatsArray3D frame
        tuple _pixels
        int _samples
        list _spectral_slices
        readonly int bins
        readonly double min_wavelength, max_wavelength, delta_wavelength
        readonly np.ndarray wavelengths

    def __init__(self, bint accumulate=True, str name=None):

        self.name = name or _DEFAULT_PIPELINE_NAME
        self.accumulate = accumulate
        self.frame = None
        self._pixels = None
        self._samples = 0
        self._spectral_slices = None

        self.min_wavelength = 0
        self.max_wavelength = 0
        self.bins = 0
        self.delta_wavelength = 0
        self.wavelengths = None

    cpdef object initialise(self, tuple pixels, int pixel_samples, double min_wavelength, double max_wavelength, int spectral_bins, list spectral_slices):

        nx, ny = pixels
        self._pixels = pixels
        self._samples = pixel_samples
        self._spectral_slices = spectral_slices

        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.delta_wavelength = (max_wavelength - min_wavelength) / spectral_bins
        self.bins = spectral_bins
        self.wavelengths = np.array([min_wavelength + (0.5 + i) * self.delta_wavelength for i in range(spectral_bins)])

        # create frame-buffer
        if not self.accumulate or self.frame is None or self.frame.shape != (nx, ny, spectral_bins):
            self.frame = StatsArray3D(nx, ny, spectral_bins)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int x, int y, int slice_id):
        return SpectralPixelProcessor(self._spectral_slices[slice_id])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object update(self, int x, int y, int slice_id, tuple packed_result):

        cdef:
            int index
            double[::1] mean, variance
            SpectralSlice slice

        # obtain result
        mean, variance = packed_result

        # accumulate samples
        slice = self._spectral_slices[slice_id]
        for index in range(slice.bins):
            self.frame.combine_samples(x, y, slice.offset + index, mean[index], variance[index], self._samples)

    cpdef object finalise(self):
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def display_pixel(self, int x, int y):

        cdef:
            np.ndarray errors
            double[::1] errors_mv
            int i

        errors = np.empty(self.frame.nz)
        errors_mv = errors
        for i in range(self.frame.nz):
            errors_mv[i] = self.frame.error(x, y, i)

        plt.figure()
        plt.plot(self.wavelengths, self.frame.mean[x, y, :], color=(0, 0, 1))
        plt.plot(self.wavelengths, self.frame.mean[x, y, :] + errors[:], color=(0.685, 0.685, 1.0))
        plt.plot(self.wavelengths, self.frame.mean[x, y, :] - errors[:], color=(0.685, 0.685, 1.0))
        plt.title('{} - Pixel ({}, {})'.format(self.name, x, y))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Spectral Power (W/nm)')
        plt.draw()
        plt.show()


cdef class SpectralPixelProcessor(PixelProcessor):

    cdef StatsArray1D bins

    def __init__(self, SpectralSlice slice):
        self.bins = StatsArray1D(slice.bins)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object add_sample(self, Spectrum spectrum, double etendue):

        cdef int index
        for index in range(self.bins.length):
            self.bins.add_sample(index, spectrum.samples_mv[index] * etendue)

    cpdef tuple pack_results(self):
        return self.bins.mean, self.bins.variance

