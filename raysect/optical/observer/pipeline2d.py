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

from raysect.core.math import StatsArray3D, StatsArray1D
from raysect.optical.colour import resample_ciexyz, spectrum_to_ciexyz, ciexyz_to_srgb
from raysect.optical.observer.observer2d import Pipeline2D, PixelProcessor
from .colormaps import viridis


class RGBPipeline2D(Pipeline2D):

    def __init__(self, sensitivity=1.0, display_progress=True, display_update_time=5, accumulate=False):

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
        self._normalisation = None

        self._pixels = None
        self._samples = 0

    @property
    def rgb_frame(self):
        if self.xyz_frame:
            return self._generate_srgb_frame(self.xyz_frame)
        return None

    def initialise(self, pixels, pixel_samples, spectral_slices):

        nx, ny = pixels

        # create intermediate and final frame-buffers
        if not self.accumulate or self.xyz_frame is None or self.xyz_frame.shape != (nx, ny, 3):
            self.xyz_frame = StatsArray3D(nx, ny, 3)

        self._working_mean = np.zeros((nx, ny, 3))
        self._working_variance = np.zeros((nx, ny, 3))

        # generate pixel processor configurations for each spectral slice
        self._resampled_xyz = [resample_ciexyz(slice.min_wavelength, slice.max_wavelength, slice.num_samples) for slice in spectral_slices]

        self._pixels = pixels
        self._samples = pixel_samples

        if self.display_progress:
            self._start_display()

    def pixel_processor(self, slice_id):
        return XYZPixelProcessor(self._resampled_xyz[slice_id])

    def update(self, pixel_id, packed_result, slice_id):

        # obtain result
        x, y = pixel_id
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

    def finalise(self):

        # update final frame with working frame results
        for x in range(self.xyz_frame.shape[0]):
            for y in range(self.xyz_frame.shape[1]):
                self.xyz_frame.combine_samples(x, y, 0, self._working_mean[x, y, 0], self._working_variance[x, y, 0], self._samples)
                self.xyz_frame.combine_samples(x, y, 1, self._working_mean[x, y, 1], self._working_variance[x, y, 1], self._samples)
                self.xyz_frame.combine_samples(x, y, 2, self._working_mean[x, y, 2], self._working_variance[x, y, 2], self._samples)

        if self.display_progress:
            self._render_display(self.xyz_frame)

    def _generate_srgb_frame(self, xyz_frame):

        # TODO - re-add exposure handlers
        nx, ny = self._pixels
        rgb_frame = np.zeros((nx, ny, 3))

        # Apply sensitivity to each pixel and convert to sRGB colour-space
        for ix in range(nx):
            for iy in range(ny):

                rgb_pixel = ciexyz_to_srgb(
                    xyz_frame.mean[ix, iy, 0] * self.sensitivity,
                    xyz_frame.mean[ix, iy, 1] * self.sensitivity,
                    xyz_frame.mean[ix, iy, 2] * self.sensitivity
                )

                rgb_frame[ix, iy, 0] = rgb_pixel[0]
                rgb_frame[ix, iy, 1] = rgb_pixel[1]
                rgb_frame[ix, iy, 2] = rgb_pixel[2]

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

    def _update_display(self, x, y):
        """
        Update live render.
        """

        # update display pixel by combining existing frame data with working data
        self._display_frame.mean[x, y, :] = self.xyz_frame.mean[x, y, :]
        self._display_frame.variance[x, y, :] = self.xyz_frame.variance[x, y, :]
        self._display_frame.samples[x, y, :] = self.xyz_frame.samples[x, y, :]

        self._display_frame.combine_samples(x, y, 0, self._working_mean[x, y, 0], self._working_variance[x, y, 0], self._samples)
        self._display_frame.combine_samples(x, y, 1, self._working_mean[x, y, 1], self._working_variance[x, y, 1], self._samples)
        self._display_frame.combine_samples(x, y, 2, self._working_mean[x, y, 2], self._working_variance[x, y, 2], self._samples)

        # update live render display
        if (time() - self._display_timer) > self.display_update_time:

            print("RGBPipeline2D updating display...")
            self._render_display(self._display_frame)
            self._display_timer = time()

    def _render_display(self, xyz_frame):

        INTERPOLATION = 'nearest'

        rgb_frame = self._generate_srgb_frame(xyz_frame)

        plt.figure(1)
        plt.clf()
        plt.imshow(np.transpose(rgb_frame, (1, 0, 2)), aspect="equal", origin="upper", interpolation=INTERPOLATION)
        plt.tight_layout()

        # plot standard error
        plt.figure(2)
        plt.clf()
        plt.imshow(np.transpose(xyz_frame.errors().mean(axis=2)), aspect="equal", origin="upper", interpolation=INTERPOLATION, cmap=viridis)
        plt.colorbar()
        plt.tight_layout()

        plt.draw()
        plt.show()

        # plot samples
        plt.figure(3)
        plt.clf()
        plt.imshow(np.transpose(xyz_frame.samples.mean(axis=2)), aspect="equal", origin="upper", interpolation=INTERPOLATION, cmap=viridis)
        plt.colorbar()
        plt.tight_layout()

        plt.draw()
        plt.show()

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


class XYZPixelProcessor(PixelProcessor):

    def __init__(self, resampled_xyz):
        self._resampled_xyz = resampled_xyz
        self._xyz = StatsArray1D(3)

    def add_sample(self, spectrum):
        # convert spectrum to CIE XYZ and add sample to pixel buffer
        x, y, z = spectrum_to_ciexyz(spectrum, self._resampled_xyz)
        self._xyz.add_sample(0, x)
        self._xyz.add_sample(1, y)
        self._xyz.add_sample(2, z)

    def pack_results(self):

        mean = (self._xyz.mean[0], self._xyz.mean[1], self._xyz.mean[2])
        variance = (self._xyz.variance[0], self._xyz.variance[1], self._xyz.variance[2])
        return mean, variance

