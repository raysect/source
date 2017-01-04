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


cdef class RGBPipeline2D(Pipeline2D):

    cdef:
        public double sensitivity
        public bint display_progress
        double _display_timer
        public double display_update_time
        public bint accumulate
        readonly StatsArray3D xyz_frame
        double[:,:,::1] _working_mean, _working_variance
        StatsArray3D _display_frame
        list _resampled_xyz
        tuple _pixels
        int _samples

    def __init__(self, double sensitivity=1.0, bint display_progress=True, double display_update_time=15, bint accumulate=False):

        # todo: add validation
        self.sensitivity = sensitivity
        self.display_progress = display_progress
        self._display_timer = 0
        self.display_update_time = display_update_time
        self.accumulate = accumulate

        self.xyz_frame = None

        self._working_mean = None
        self._working_variance = None

        self._display_frame = None

        self._resampled_xyz = None

        self._pixels = None
        self._samples = 0

    @property
    def rgb_frame(self):
        if self.xyz_frame:
            return self._generate_srgb_frame(self.xyz_frame)
        return None

    cpdef object initialise(self, tuple pixels, int pixel_samples, int spectral_samples, double lower_wavelength, double upper_wavelength, list spectral_slices):

        nx, ny = pixels
        self._pixels = pixels
        self._samples = pixel_samples

        # create intermediate and final frame-buffers
        if not self.accumulate or self.xyz_frame is None or self.xyz_frame.shape != (nx, ny, 3):
            self.xyz_frame = StatsArray3D(nx, ny, 3)

        self._working_mean = np.zeros((nx, ny, 3))
        self._working_variance = np.zeros((nx, ny, 3))

        # generate pixel processor configurations for each spectral slice
        self._resampled_xyz = [resample_ciexyz(slice.lower_wavelength, slice.upper_wavelength, slice.bins) for slice in spectral_slices]

        if self.display_progress:
            self._start_display()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int x, int y, int slice_id):
        return XYZPixelProcessor(self._resampled_xyz[slice_id])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object update(self, int x, int y, int slice_id, tuple packed_result):

        cdef:
            double[::1] mean, variance

        # unpack results
        mean, variance = packed_result

        # accumulate sub-samples
        self._working_mean[x, y, 0] += mean[0]
        self._working_mean[x, y, 1] += mean[1]
        self._working_mean[x, y, 2] += mean[2]

        self._working_variance[x, y, 0] += variance[0]
        self._working_variance[x, y, 1] += variance[1]
        self._working_variance[x, y, 2] += variance[2]

        # update users
        if self.display_progress:
            self._update_display(x, y)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object finalise(self):

        cdef int x, y

        # update final frame with working frame results
        for x in range(self.xyz_frame.shape[0]):
            for y in range(self.xyz_frame.shape[1]):
                self.xyz_frame.combine_samples(x, y, 0, self._working_mean[x, y, 0], self._working_variance[x, y, 0], self._samples)
                self.xyz_frame.combine_samples(x, y, 1, self._working_mean[x, y, 1], self._working_variance[x, y, 1], self._samples)
                self.xyz_frame.combine_samples(x, y, 2, self._working_mean[x, y, 2], self._working_variance[x, y, 2], self._samples)

        if self.display_progress:
            self._render_display(self.xyz_frame)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _generate_srgb_frame(self, StatsArray3D xyz_frame):

        cdef:
            int nx, ny, ix, iy
            np.ndarray rgb_frame
            double[:,:,::1] rgb_frame_mv
            tuple rgb_pixel

        # TODO - re-add exposure handlers
        nx, ny = self._pixels
        rgb_frame = np.zeros((nx, ny, 3))
        rgb_frame_mv = rgb_frame

        # Apply sensitivity to each pixel and convert to sRGB colour-space
        for ix in range(nx):
            for iy in range(ny):

                rgb_pixel = ciexyz_to_srgb(
                    xyz_frame.mean_mv[ix, iy, 0] * self.sensitivity,
                    xyz_frame.mean_mv[ix, iy, 1] * self.sensitivity,
                    xyz_frame.mean_mv[ix, iy, 2] * self.sensitivity
                )

                rgb_frame_mv[ix, iy, 0] = rgb_pixel[0]
                rgb_frame_mv[ix, iy, 1] = rgb_pixel[1]
                rgb_frame_mv[ix, iy, 2] = rgb_pixel[2]

        return rgb_frame

    def _start_display(self):
        """
        Display live render.
        """

        # populate live frame with current frame state
        self._display_frame = self.xyz_frame.copy()

        # display initial frame
        self._render_display(self._display_frame)
        self._display_timer = time()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_display(self, int x, int y):
        """
        Update live render.
        """

        # update display pixel by combining existing frame data with working data
        self._display_frame.mean_mv[x, y, :] = self.xyz_frame.mean_mv[x, y, :]
        self._display_frame.variance_mv[x, y, :] = self.xyz_frame.variance_mv[x, y, :]
        self._display_frame.samples_mv[x, y, :] = self.xyz_frame.samples_mv[x, y, :]

        self._display_frame.combine_samples(x, y, 0, self._working_mean[x, y, 0], self._working_variance[x, y, 0], self._samples)
        self._display_frame.combine_samples(x, y, 1, self._working_mean[x, y, 1], self._working_variance[x, y, 1], self._samples)
        self._display_frame.combine_samples(x, y, 2, self._working_mean[x, y, 2], self._working_variance[x, y, 2], self._samples)

        # update live render display
        if (time() - self._display_timer) > self.display_update_time:

            print("RGBPipeline2D updating display...")
            self._render_display(self._display_frame)
            self._display_timer = time()

    def _render_display(self, display_frame):

        INTERPOLATION = 'nearest'

        rgb_frame = self._generate_srgb_frame(display_frame)

        plt.figure(1)
        plt.clf()
        plt.imshow(np.transpose(rgb_frame, (1, 0, 2)), aspect="equal", origin="upper", interpolation=INTERPOLATION)
        plt.tight_layout()

        # # plot standard error
        # plt.figure(2)
        # plt.clf()
        # plt.imshow(np.transpose(self.xyz_frame.errors().mean(axis=2)), aspect="equal", origin="upper", interpolation=INTERPOLATION, cmap=viridis)
        # plt.colorbar()
        # plt.tight_layout()
        #
        # # plot samples
        # plt.figure(3)
        # plt.clf()
        # plt.imshow(np.transpose(self.xyz_frame.samples.mean(axis=2)), aspect="equal", origin="upper", interpolation=INTERPOLATION, cmap=viridis)
        # plt.colorbar()
        # plt.tight_layout()
        #
        # plt.draw()
        # plt.show()

        # workaround for interactivity for QT backend
        plt.pause(0.1)

    def display(self):
        if self.xyz_frame:
            self._render_display(self.xyz_frame)
        raise ValueError("There is no frame to display.")

    def save(self, filename):
        """
        Save the collected samples in the camera frame to file.
        :param str filename: Filename and path for camera frame output file.
        """

        rgb_frame = self._generate_srgb_frame(self.xyz_frame)
        plt.imsave(filename, np.transpose(rgb_frame, (1, 0, 2)))


cdef class XYZPixelProcessor(PixelProcessor):

    cdef:
        np.ndarray _resampled_xyz
        StatsArray1D _xyz

    def __init__(self, np.ndarray resampled_xyz):
        self._resampled_xyz = resampled_xyz
        self._xyz = StatsArray1D(3)

    cpdef object add_sample(self, Spectrum spectrum):

        cdef double x, y, z

        # convert spectrum to CIE XYZ and add sample to pixel buffer
        x, y, z = spectrum_to_ciexyz(spectrum, self._resampled_xyz)
        self._xyz.add_sample(0, x)
        self._xyz.add_sample(1, y)
        self._xyz.add_sample(2, z)

    cpdef tuple pack_results(self):
        return self._xyz.mean, self._xyz.variance


cdef class BayerPipeline2D(Pipeline2D):

    cdef:
        public double sensitivity
        public bint display_progress
        double _display_timer
        public double display_update_time
        public bint accumulate
        readonly StatsArray2D frame
        double[:,::1] _working_mean, _working_variance
        StatsArray2D _display_frame
        list _resampled_xyz
        tuple _pixels
        int _samples

    def __init__(self, double sensitivity=1.0, bint display_progress=True, double display_update_time=15, bint accumulate=False):

        # todo: add validation
        self.sensitivity = sensitivity
        self.display_progress = display_progress
        self._display_timer = 0
        self.display_update_time = display_update_time
        self.accumulate = accumulate

        self.frame = None

        self._working_mean = None
        self._working_variance = None

        self._display_frame = None

        self._resampled_xyz = None

        self._pixels = None
        self._samples = 0

    # @property
    # def rgb_frame(self):
    #     if self.xyz_frame:
    #         return self._generate_srgb_frame(self.xyz_frame)
    #     return None

    cpdef object initialise(self, tuple pixels, int pixel_samples, int spectral_samples, double lower_wavelength, double upper_wavelength, list spectral_slices):

        nx, ny = pixels
        self._pixels = pixels
        self._samples = pixel_samples

        # create intermediate and final frame-buffers
        if not self.accumulate or self.frame is None or self.frame.shape != (nx, ny):
            self.frame = StatsArray2D(nx, ny)

        self._working_mean = np.zeros((nx, ny))
        self._working_variance = np.zeros((nx, ny))

        # generate pixel processor configurations for each spectral slice
        self._resampled_xyz = [resample_ciexyz(slice.lower_wavelength, slice.upper_wavelength, slice.bins) for slice in spectral_slices]

        if self.display_progress:
            self._start_display()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int x, int y, int slice_id):
        return BayerPixelProcessor(x, y, self._resampled_xyz[slice_id])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object update(self, int x, int y, int slice_id, tuple packed_result):

        cdef:
            double mean, variance

        # obtain result
        mean, variance = packed_result

        # accumulate sub-samples
        self._working_mean[x, y] += mean
        self._working_variance[x, y] += variance

        # update users
        if self.display_progress:
            self._update_display(x, y)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object finalise(self):

        cdef int x, y

        # update final frame with working frame results
        for x in range(self.frame.shape[0]):
            for y in range(self.frame.shape[1]):
                self.frame.combine_samples(x, y, self._working_mean[x, y], self._working_variance[x, y], self._samples)

        if self.display_progress:
            self._render_display(self.frame)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # cpdef _generate_srgb_frame(self, StatsArray3D xyz_frame):
    #
    #     cdef:
    #         int nx, ny, ix, iy
    #         np.ndarray rgb_frame
    #         double[:,:,::1] rgb_frame_mv
    #         tuple rgb_pixel
    #
    #     # TODO - re-add exposure handlers
    #     nx, ny = self._pixels
    #     rgb_frame = np.zeros((nx, ny, 3))
    #     rgb_frame_mv = rgb_frame
    #
    #     # Apply sensitivity to each pixel and convert to sRGB colour-space
    #     for ix in range(nx):
    #         for iy in range(ny):
    #
    #             rgb_pixel = ciexyz_to_srgb(
    #                 xyz_frame.mean_mv[ix, iy, 0] * self.sensitivity,
    #                 xyz_frame.mean_mv[ix, iy, 1] * self.sensitivity,
    #                 xyz_frame.mean_mv[ix, iy, 2] * self.sensitivity
    #             )
    #
    #             rgb_frame_mv[ix, iy, 0] = rgb_pixel[0]
    #             rgb_frame_mv[ix, iy, 1] = rgb_pixel[1]
    #             rgb_frame_mv[ix, iy, 2] = rgb_pixel[2]
    #
    #     return rgb_frame

    def _start_display(self):
        """
        Display live render.
        """

        # populate live frame with current frame state
        self._display_frame = self.frame.copy()

        # display initial frame
        self._render_display(self._display_frame)
        self._display_timer = time()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _update_display(self, int x, int y):
        """
        Update live render.
        """

        # update display pixel by combining existing frame data with working data
        self._display_frame.mean_mv[x, y] = self.frame.mean_mv[x, y]
        self._display_frame.variance_mv[x, y] = self.frame.variance_mv[x, y]
        self._display_frame.samples_mv[x, y] = self.frame.samples_mv[x, y]

        self._display_frame.combine_samples(x, y, self._working_mean[x, y], self._working_variance[x, y], self._samples)

        # update live render display
        if (time() - self._display_timer) > self.display_update_time:

            print("BayerPipeline2D updating display...")
            self._render_display(self._display_frame)
            self._display_timer = time()

    def _render_display(self, display_frame):

        INTERPOLATION = 'nearest'

        plt.figure(11)
        plt.clf()
        plt.imshow(np.transpose(display_frame.mean * self.sensitivity), aspect="equal", origin="upper", interpolation=INTERPOLATION, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.tight_layout()

        # # plot standard error
        # plt.figure(12)
        # plt.clf()
        # plt.imshow(np.transpose(self.frame.errors()), aspect="equal", origin="upper", interpolation=INTERPOLATION, cmap=viridis)
        # plt.colorbar()
        # plt.tight_layout()
        #
        # # plot samples
        # plt.figure(13)
        # plt.clf()
        # plt.imshow(np.transpose(self.frame.samples), aspect="equal", origin="upper", interpolation=INTERPOLATION, cmap=viridis)
        # plt.colorbar()
        # plt.tight_layout()

        plt.draw()
        plt.show()

        # workaround for interactivity for QT backend
        plt.pause(0.1)

    def display(self):
        if self.frame:
            self._render_display(self.frame)
        raise ValueError("There is no frame to display.")

    # def save(self, filename):
    #     """
    #     Save the collected samples in the camera frame to file.
    #     :param str filename: Filename and path for camera frame output file.
    #     """
    #
    #     rgb_frame = self._generate_srgb_frame(self.frame)
    #     plt.imsave(filename, np.transpose(rgb_frame))


cdef class BayerPixelProcessor(PixelProcessor):

    cdef:
        np.ndarray _resampled_xyz
        StatsArray1D _bin
        int _filter_id

    def __init__(self, int x, int y, np.ndarray resampled_xyz):

        # select appropriate filter for pixel
        if y % 2:
            # upper row
            if x % 2:
                # left - red
                self._filter_id = 0
            else:
                # right - green
                self._filter_id = 1
        else:
            # lower row
            if x % 2:
                # left - green
                self._filter_id = 1
            else:
                # right - blue
                self._filter_id = 2

        self._resampled_xyz = resampled_xyz
        self._bin = StatsArray1D(1)

    cpdef object add_sample(self, Spectrum spectrum):

        cdef tuple xyz

        # todo: inefficient kludge
        # convert spectrum to CIE XYZ and add sample to pixel buffer
        xyz = spectrum_to_ciexyz(spectrum, self._resampled_xyz)
        self._bin.add_sample(0, xyz[self._filter_id])

    cpdef tuple pack_results(self):
        return self._bin.mean[0], self._bin.variance[0]


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

    cpdef object initialise(self, tuple pixels, int pixel_samples, int spectral_samples, double lower_wavelength, double upper_wavelength, list spectral_slices):

        nx, ny = pixels
        self._pixels = pixels
        self._samples = pixel_samples
        self._spectral_slices = spectral_slices

        # create intermediate and final frame-buffers
        if not self.accumulate or self.frame is None or self.frame.shape != (nx, ny, spectral_samples):
            self.frame = StatsArray3D(nx, ny, spectral_samples)

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

