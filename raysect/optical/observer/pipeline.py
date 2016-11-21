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

from raysect.optical.colour import resample_ciexyz, spectrum_to_ciexyz
from raysect.optical.observer.frame import Frame2D, Pixel


class Pipeline2D:
    """
    base class defining the core interfaces to define an image processing pipeline
    """

    def initialise(self, pixels, ray_templates):
        """
        setup internal buffers (e.g. frames)
        reset internal statistics as appropriate
        etc..

        :return:
        """
        pass

    def pixel_processor(self, channel):
        pass

    def update(self, pixel, packed_result, channel):
        pass

    def finalise(self):
        pass


class PixelProcessor:

    def add_sample(self, spectrum):
        pass

    def pack_results(self):
        pass


class RGBPipeline2D(Pipeline2D):

    def initialise(self, pixels, ray_templates):

        # create intermediate and final frame-buffers
        # if not self.accumulate:
        self.xyz_frame = Frame2D(pixels, channels=3)
        self.rgb_frame = np.zeros((pixels[0], pixels[1], 3))

        # generate resampled XYZ curves for ray spectral ranges
        self._resampled_xyz = [resample_ciexyz(ray.min_wavelength, ray.max_wavelength, ray.num_samples) for ray in ray_templates]

        # TODO - add statistics and display initialisation

    def pixel_processor(self, channel):
        return XYZPixelProcessor(self._resampled_xyz[channel])

    def update(self, pixel_id, packed_result, channel):

        # obtain result
        x, y = pixel_id
        mean, variance, samples = packed_result

        self.xyz_frame.combine_samples(x, y, 0, mean[0], variance[0], samples[0])
        self.xyz_frame.combine_samples(x, y, 1, mean[1], variance[1], samples[1])
        self.xyz_frame.combine_samples(x, y, 2, mean[2], variance[2], samples[2])

        # update users
        # self._update_display()
        # self._update_statistics(channel, x, y, sample_ray_count)

    def finalise(self):

        plt.figure(1)
        plt.clf()
        img = np.transpose(10 * self.xyz_frame.value/self.xyz_frame.value.max(), (1, 0, 2))
        img[img > 1.0] = 1.0
        plt.imshow(img, aspect="equal", origin="upper")
        plt.draw()
        plt.show()

        # workaround for interactivity for QT backend
        plt.pause(0.1)


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
