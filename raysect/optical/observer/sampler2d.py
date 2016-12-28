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

from random import shuffle
from .base import FrameSampler2D


class FullFrameSampler2D(FrameSampler2D):

    def generate_tasks(self, pixels):

        tasks = []
        nx, ny = pixels
        for iy in range(ny):
            for ix in range(nx):
                tasks.append((ix, iy))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


# TODO - add Adaptive Sampler
# class AdaptiveSampler2D(FrameSampler):
#
#     def generate_tasks(self, pixels):
#
#         # build task list -> task is simply the pixel location
#         tasks = []
#         nx, ny = pixels
#         f = self.xyz_frame
#         i = f.value > 0
#         norm_frame_var = np.zeros((pixels[0], pixels[1], 3))
#         norm_frame_var[i] = f.variance[i]
#         # norm_frame_var[i] = np.minimum(f.variance[i], f.variance[i] / f.value[i]**2)
#         max_frame_samples = f.samples.max()
#         percentile_frame_variance = np.percentile(norm_frame_var, 80)
#         for iy in range(ny):
#             for ix in range(nx):
#                 min_pixel_samples = self.xyz_frame.samples[ix, iy, :].min()
#                 max_pixel_variance = norm_frame_var[ix, iy, :].max()
#                 if min_pixel_samples < 1000*self.spectral_rays or \
#                     min_pixel_samples <= 0.1 * max_frame_samples or \
#                     max_pixel_variance >= percentile_frame_variance:
#                     tasks.append((ix, iy))

