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

from time import time
cimport cython
cimport numpy as np
import matplotlib.pyplot as plt
import numpy as np
from .colormaps import viridis

from raysect.optical.observer.base cimport PixelProcessor, Pipeline2D
from raysect.core.math cimport StatsArray3D, StatsArray2D, StatsArray1D
from raysect.optical.colour cimport resample_ciexyz, spectrum_to_ciexyz, ciexyz_to_srgb
from raysect.optical.spectrum cimport Spectrum
from raysect.optical.observer.base.slice cimport SpectralSlice


cdef class SpectralPipeline2D(Pipeline2D):

    cdef:
        public double sensitivity
        public bint display_progress
        double _display_timer
        public double display_update_time
        public bint accumulate
        readonly StatsArray3D frame
        double[:,:,::1] _working_mean, _working_variance
        StatsArray3D _display_frame
        tuple _pixels
        int _samples
        list _spectral_slices

    def __init__(self, double sensitivity=1.0, bint display_progress=True, double display_update_time=15, bint accumulate=False):

        # todo: add validation
        self.sensitivity = sensitivity
        self.display_progress = display_progress
        self._display_timer = 0
        self.display_update_time = display_update_time
        self.accumulate = accumulate

        self.frame = None

        self._pixels = None
        self._samples = 0
        self._spectral_slices = None

    cpdef object initialise(self, tuple pixels, int pixel_samples, double min_wavelength, double max_wavelength, int spectral_bins, list spectral_slices):

        nx, ny = pixels
        self._pixels = pixels
        self._samples = pixel_samples
        self._spectral_slices = spectral_slices

        # create intermediate and final frame-buffers
        if not self.accumulate or self.frame is None or self.frame.shape != (nx, ny, spectral_bins):
            self.frame = StatsArray3D(nx, ny, spectral_bins)

        if self.display_progress:
            self._start_display()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int x, int y, int slice_id):
        return SpectralPixelProcessor(self._spectral_slices[slice_id])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object update(self, int x, int y, int slice_id, tuple packed_result):

        cdef:
            int bin, index
            double[::1] mean, variance
            SpectralSlice slice

        # obtain result
        mean, variance = packed_result

        # accumulate samples
        slice = self._spectral_slices[slice_id]
        for index in range(slice.bins):
            bin = slice.offset + index
            self.frame.combine_samples(x, y, bin, mean[index], variance[index], self._samples)

        # update users
        if self.display_progress:
            self._update_display(x, y)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object finalise(self):

        if self.display_progress:
            self._render_display()

    def _start_display(self):
        """
        Display live render.
        """

        # display initial frame
        self._render_display()
        self._display_timer = time()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_display(self, int x, int y):
        """
        Update live render.
        """
        # update live render display
        if (time() - self._display_timer) > self.display_update_time:

            print("SpectralPipeline2D updating display...")
            self._render_display()
            self._display_timer = time()

    def _render_display(self):

        INTERPOLATION = 'nearest'

        total = self.frame.mean.sum(axis=2)
        errors = self.frame.errors()

        plt.figure(21)
        plt.clf()
        plt.imshow(np.transpose(self.frame.mean.sum(axis=2)), aspect="equal", origin="upper", interpolation=INTERPOLATION, cmap=plt.get_cmap('gray'))
        plt.tight_layout()

        # plt.figure(22)
        # plt.clf()
        # plt.plot(self.frame.mean[28, 8, :], 'b-')
        # plt.plot(self.frame.mean[28, 8, :] + 3 * errors[28, 8, :], 'k-')
        # plt.plot(self.frame.mean[28, 8, :] - 3 * errors[28, 8, :], 'k-')
        # plt.tight_layout()

        # # plot standard error
        # plt.figure(2)
        # plt.clf()
        # plt.imshow(np.transpose(self.frame.errors().mean(axis=2)), aspect="equal", origin="upper", interpolation=INTERPOLATION, cmap=viridis)
        # plt.colorbar()
        # plt.tight_layout()
        #
        # # plot samples
        # plt.figure(3)
        # plt.clf()
        # plt.imshow(np.transpose(self.frame.samples.mean(axis=2)), aspect="equal", origin="upper", interpolation=INTERPOLATION, cmap=viridis)
        # plt.colorbar()
        # plt.tight_layout()
        #
        # plt.draw()
        # plt.show()

        # workaround for interactivity for QT backend
        plt.pause(0.1)

    def display(self):
        if self.frame:
            self._render_display()
        raise ValueError("There is no frame to display.")

    # def save(self, filename):
    #     """
    #     Save the collected samples in the camera frame to file.
    #     :param str filename: Filename and path for camera frame output file.
    #     """
    #
    #     rgb_frame = self._generate_srgb_frame(self.xyz_frame)
    #     plt.imsave(filename, np.transpose(rgb_frame, (1, 0, 2)))


cdef class SpectralPixelProcessor(PixelProcessor):

    cdef StatsArray1D _bins

    def __init__(self, SpectralSlice slice):
        self._bins = StatsArray1D(slice.bins)

    cpdef object add_sample(self, Spectrum spectrum):

        cdef int index
        for index in range(self._bins.length):
            self._bins.add_sample(index, spectrum.samples_mv[index])

    cpdef tuple pack_results(self):
        return self._bins.mean, self._bins.variance

