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
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from .colormaps import viridis

cimport cython
cimport numpy as np
from raysect.core.math.cython cimport clamp
from raysect.optical.spectralfunction cimport SpectralFunction, ConstantSF
from raysect.optical.observer.base cimport PixelProcessor, Pipeline2D
from raysect.core.math cimport StatsBin, StatsArray2D
from raysect.optical.spectrum cimport Spectrum
from raysect.optical.observer.base.slice cimport SpectralSlice
from raysect.optical.observer.base.sampler cimport FrameSampler2D


cdef class MonoPipeline2D(Pipeline2D):

    cdef:
        public SpectralFunction sensitivity
        public bint display_progress
        double _display_timer
        double display_update_time  # TODO - add property
        public bint accumulate
        readonly StatsArray2D frame
        double[:,::1] _working_mean, _working_variance
        char[:,::1] _working_touched
        StatsArray2D _display_frame
        list _resampled_sensitivity
        tuple _pixels
        int _samples
        double black_point, white_point, unsaturated_fraction
        bint auto_exposure

    def __init__(self, SpectralFunction sensitivity=None, bint display_progress=True,
                 double display_update_time=15, bint accumulate=True,
                 bint auto_exposure=True, double black_point=0.0, double white_point=1.0,
                 double unsaturated_fraction=1.0):

        # todo: add validation
        self.sensitivity = sensitivity or ConstantSF(1.0)

        self.display_progress = display_progress
        self.auto_exposure = auto_exposure

        if black_point < 0:
            raise ValueError('Black point cannot be less than zero.')
        self.black_point = black_point
        if white_point < black_point:
            white_point = black_point
        self.white_point = white_point

        if not (0 < unsaturated_fraction <= 1):
            raise ValueError('Unsaturated fraction must lie in range (0, 1].')
        self.unsaturated_fraction = unsaturated_fraction

        self._display_timer = 0
        self.display_update_time = display_update_time
        self.accumulate = accumulate

        self.frame = None

        self._working_mean = None
        self._working_variance = None
        self._working_touched = None

        self._display_frame = None

        self._resampled_sensitivity = None

        self._pixels = None
        self._samples = 0

    cpdef object initialise(self, tuple pixels, int pixel_samples, double min_wavelength, double max_wavelength, int spectral_bins, list spectral_slices):

        nx, ny = pixels
        self._pixels = pixels
        self._samples = pixel_samples

        # create intermediate and final frame-buffers
        if not self.accumulate or self.frame is None or self.frame.shape != (nx, ny):
            self.frame = StatsArray2D(nx, ny)

        self._working_mean = np.zeros((nx, ny))
        self._working_variance = np.zeros((nx, ny))
        self._working_touched = np.zeros((nx, ny), dtype=np.int8)

        # generate pixel processor configurations for each spectral slice
        self._resampled_sensitivity = [self.sensitivity.sample(slice.min_wavelength, slice.max_wavelength, slice.bins) for slice in spectral_slices]

        if self.display_progress:
            self._start_display()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int x, int y, int slice_id):
        return MonoPixelProcessor(x, y, self._resampled_sensitivity[slice_id])

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

        # mark pixel as modified
        self._working_touched[x, y] = 1

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
                if self._working_touched[x, y] == 1:
                    self.frame.combine_samples(x, y, self._working_mean[x, y], self._working_variance[x, y], self._samples)

        if self.display_progress:
            self._render_display(self.frame)

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

            print("MonoPipeline2D updating display...")
            self._render_display(self._display_frame)
            self._display_timer = time()

    def _render_display(self, display_frame):

        cdef:
            int nx, ny, ix, iy
            np.ndarray image
            double[:,::1] image_mv

        INTERPOLATION = 'nearest'

        if self.auto_exposure:
            self.white_point = self._calculate_white_point(display_frame.mean)

        image = display_frame.mean.copy()
        image_mv = image
        nx = display_frame.shape[0]
        ny = display_frame.shape[1]

        # clamp data to within black and white point range, and shift zero to blackpoint
        for ix in range(nx):
            for iy in range(ny):
                image_mv[ix, iy] = clamp(image_mv[ix, iy], self.black_point, self.white_point) - self.black_point

        plt.figure(11)
        plt.clf()
        plt.imshow(np.transpose(image), aspect="equal", origin="upper", interpolation=INTERPOLATION, cmap=plt.get_cmap('gray'))
        plt.draw()
        plt.show()

        # workaround for interactivity for QT backend
        plt.pause(0.1)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double _calculate_white_point(self, np.ndarray frame):

        cdef:
            int nx, ny, pixels, ix, iy, i
            double peak_luminance
            np.ndarray luminance
            double[:,::1] fmv
            double[::1] lmv

        nx = frame.shape[1]
        ny = frame.shape[0]
        fmv = frame  # memory view

        pixels = nx * ny
        luminance = np.zeros(pixels)
        lmv = luminance  # memory view

        # calculate luminance values for frame (XYZ Y component is luminance)
        for iy in range(ny):
            for ix in range(nx):
                lmv[iy*nx + ix] = max(fmv[iy, ix] - self.black_point, 0)

        # sort by luminance
        luminance.sort()

        # if all pixels black, return default sensitivity
        for i in range(pixels):
            if lmv[i] > 0:
                break

        if i == pixels:
            return self.black_point

        # identify luminance at threshold
        peak_luminance = lmv[<int> min(pixels - 1, pixels * self.unsaturated_fraction)]

        if peak_luminance == 0:
            return self.black_point

        return peak_luminance + self.black_point

    def display(self):
        if not self.frame:
            raise ValueError("There is no frame to display.")
        self._render_display(self.frame)


    # def save(self, filename):
    #     """
    #     Save the collected samples in the camera frame to file.
    #     :param str filename: Filename and path for camera frame output file.
    #     """
    #
    #     rgb_frame = self._generate_srgb_frame(self.frame)
    #     plt.imsave(filename, np.transpose(rgb_frame))


cdef class MonoPixelProcessor(PixelProcessor):

    cdef:
        StatsBin bin
        double[::1] sensitivity, _temp

    def __init__(self, int x, int y, double[::1] sensitivity):
        self.bin = StatsBin()
        self.sensitivity = sensitivity

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object add_sample(self, Spectrum spectrum):

        cdef:
            int index
            double total = 0

        # apply sensitivity curve
        for index in range(spectrum.bins):
            total += spectrum.samples_mv[index] * self.sensitivity[index]

        self.bin.add_sample(total)

    cpdef tuple pack_results(self):
        return self.bin.mean, self.bin.variance


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MonoAdaptiveSampler2D(FrameSampler2D):

    cdef:
        MonoPipeline2D pipeline
        double fraction, ratio, cutoff
        int min_samples

    def __init__(self, MonoPipeline2D pipeline, double fraction=0.2, double ratio=10.0, int min_samples=1000, double cutoff=0.0):

        # todo: validation
        self.pipeline = pipeline
        self.fraction = fraction
        self.ratio = ratio
        self.min_samples = min_samples
        self.cutoff = cutoff

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef generate_tasks(self, tuple pixels):

        cdef:
            int nx, ny, x, y
            np.ndarray normalised
            double[:,::1] error, normalised_mv
            double percentile_error
            list tasks

        nx, ny = pixels
        frame = self.pipeline.frame
        min_samples = max(self.min_samples, frame.samples.max() / self.ratio)
        error = frame.errors()
        normalised = np.zeros((nx, ny))
        normalised_mv = normalised

        # calculated normalised standard error
        for x in range(nx):
            for y in range(ny):
                if frame.mean_mv[x, y] <= 0:
                    normalised_mv[x, y] = 0
                else:
                    normalised_mv[x, y] = error[x, y] / frame.mean_mv[x, y]

        # locate error value corresponding to fraction of frame to process
        percentile_error = np.percentile(normalised, (1 - self.fraction) * 100)

        # build tasks
        tasks = []
        for x in range(nx):
            for y in range(ny):
                if frame.samples_mv[x, y] < min_samples or normalised_mv[x, y] > max(self.cutoff, percentile_error):
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks
