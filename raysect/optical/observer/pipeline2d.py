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

import matplotlib.pyplot as plt
import numpy as np
from time import time

from raysect.optical.colour import resample_ciexyz, spectrum_to_ciexyz, ciexyz_to_srgb
from raysect.optical.observer.frame import Frame2D, Pixel
from raysect.optical.observer.observer2d import Pipeline2D, PixelProcessor


class RGBPipeline2D(Pipeline2D):

    def __init__(self, sensitivity=1.0, display_progress=True, display_update_time=5):
        self.sensitivity = sensitivity
        self.display_progress = display_progress
        self._display_timer = 0
        self.display_update_time = display_update_time

    def initialise(self, pixels, spectral_fragments):

        # create intermediate and final frame-buffers
        # if not self.accumulate:
        self.xyz_frame = Frame2D(pixels, channels=3)
        self.rgb_frame = np.zeros((pixels[0], pixels[1], 3))

        # generate pixel processors for each fragment
        self._resampled_xyz = {}
        for fragment in spectral_fragments:
            id, _, _, num_samples, min_wavelength, max_wavelength = fragment
            self._resampled_xyz[id] = resample_ciexyz(min_wavelength, max_wavelength, num_samples)

        self._start_display()

    def pixel_processor(self, fragment):
        id, _, _, _, _, _ = fragment
        return XYZPixelProcessor(self._resampled_xyz[id])

    def update(self, pixel_id, packed_result, fragment):

        # obtain result
        x, y = pixel_id
        mean, variance, samples = packed_result

        self.xyz_frame.combine_samples(x, y, 0, mean[0], variance[0], samples[0])
        self.xyz_frame.combine_samples(x, y, 1, mean[1], variance[1], samples[1])
        self.xyz_frame.combine_samples(x, y, 2, mean[2], variance[2], samples[2])

        # update users
        self._update_display()
        # self._update_statistics(channel, x, y, sample_ray_count)

    def finalise(self):

        self._generate_srgb_frame()

        if self.display_progress:
            self.display()

    def _generate_srgb_frame(self):

        # TODO - re-add exposure handlers

        # Apply sensitivity to each pixel and convert to sRGB colour-space
        nx, ny, _ = self.rgb_frame.shape
        for ix in range(nx):
            for iy in range(ny):

                rgb = ciexyz_to_srgb(
                    self.xyz_frame.value[ix, iy, 0] * self.sensitivity,
                    self.xyz_frame.value[ix, iy, 1] * self.sensitivity,
                    self.xyz_frame.value[ix, iy, 2] * self.sensitivity
                )

                self.rgb_frame[ix, iy, 0] = rgb[0]
                self.rgb_frame[ix, iy, 1] = rgb[1]
                self.rgb_frame[ix, iy, 2] = rgb[2]

    def _start_display(self):
        """
        Display live render.
        """

        self._display_timer = 0
        if self.display_progress:
            self.display()
            self._display_timer = time()

    def _update_display(self):
        """
        Update live render.
        """

        # update live render display
        if self.display_progress and (time() - self._display_timer) > self.display_update_time:

            print("RGBPipeline2D updating display...")
            self._generate_srgb_frame()
            self.display()
            self._display_timer = time()

    def display(self):
        plt.clf()
        plt.imshow(np.transpose(self.rgb_frame, (1, 0, 2)), aspect="equal", origin="upper")
        plt.draw()
        plt.show()

        # workaround for interactivity for QT backend
        plt.pause(0.1)

    def save(self, filename):
        """
        Save the collected samples in the camera frame to file.
        :param str filename: Filename and path for camera frame output file.
        """
        plt.imsave(filename, np.transpose(self.rgb_frame, (1, 0, 2)))


class XYZPixelProcessor(PixelProcessor):

    def __init__(self, resampled_xyz):
        self._resampled_xyz = resampled_xyz
        self._xyz = Pixel(channels=3)

    def add_sample(self, spectrum):
        # convert spectrum to CIE XYZ and add sample to pixel buffer
        x, y, z = spectrum_to_ciexyz(spectrum, self._resampled_xyz)
        self._xyz.add_sample(0, x)
        self._xyz.add_sample(1, y)
        self._xyz.add_sample(2, z)

    def pack_results(self):

        mean = (self._xyz.value[0], self._xyz.value[1], self._xyz.value[2])
        variance = (self._xyz.variance[0], self._xyz.variance[1], self._xyz.variance[2])
        samples = (self._xyz.samples[0], self._xyz.samples[1], self._xyz.samples[2])

        return mean, variance, samples
