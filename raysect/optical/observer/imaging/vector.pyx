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


import numpy as np
cimport numpy as np

from raysect.core cimport Point3D, Vector3D, Point2D
from raysect.core.math.random cimport point_square
from raysect.optical cimport Ray
from raysect.optical.observer.base cimport Observer2D
from raysect.optical.observer.sampler2d import FullFrameSampler2D, RGBAdaptiveSampler2D
from raysect.optical.observer.pipeline import RGBPipeline2D


cdef class VectorCamera(Observer2D):
    """
    An observer that uses a specified set of pixel vectors.

    A simple camera that uses calibrated vectors for each pixel to sample the scene.
    Arguments and attributes are inherited from the base Observer2D sensor class.

    :param np.ndarray pixel_origins: Numpy array of Point3Ds describing the origin points
      of each pixel. Must have same shape as the pixel dimensions.
    :param np.ndarray pixel_directions: Numpy array of Vector3Ds describing the sampling
      direction vectors of each pixel. Must have same shape as the pixel dimensions.
    :param float sensitivity: The sensitivity of each pixel (default=1.0)
    :param FrameSampler2D frame_sampler: The frame sampling strategy (default=FullFrameSampler2D()).
    :param list pipelines: The list of pipelines that will process the spectrum measured
      at each pixel by the camera (default=RGBPipeline2D()).
    :param kwargs: **kwargs and properties from Observer2D and _ObserverBase.
    """

    def __init__(self, pixel_origins, pixel_directions, frame_sampler=None, pipelines=None, sensitivity=None, parent=None, transform=None, name=None):

        # defaults to an adaptively sampled RGB pipeline
        if not pipelines and not frame_sampler:
            rgb = RGBPipeline2D()
            pipelines = [rgb]
            frame_sampler = RGBAdaptiveSampler2D(rgb)
        else:
            pipelines = pipelines or [RGBPipeline2D()]
            frame_sampler = frame_sampler or FullFrameSampler2D()

        pixel_origins = np.array(pixel_origins)
        pixel_directions = np.array(pixel_directions)
        pixels = pixel_origins.shape

        if len(pixels) != 2:
            raise ValueError("Pixel arrays must have 2 dimensions.")

        if pixel_origins.shape != pixel_directions.shape:
            raise ValueError("Pixel arrays must have equal shapes.")

        super().__init__(pixel_origins.shape, frame_sampler, pipelines, parent=parent, transform=transform, name=name)

        # camera configuration
        self.pixel_origins = pixel_origins
        self.pixel_directions = pixel_directions
        self._sensitivity = sensitivity or 1.0

    @property
    def sensitivity(self):
        """
        The sensitivity applied to each pixel.

        If sensitivity=1.0 all spectral units are in radiance.

        :rtype: float
        """
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        if value <= 0:
            raise ValueError("Sensitivity must be greater than zero.")
        self._sensitivity = value

    cpdef list _generate_rays(self, int x, int y, Ray template, int ray_count):

        cdef:
            list rays
            Point3D origin
            Vector3D direction, v1, v2, v3, v4, v14, v23
            Ray ray
            int i
            Point2D sample_point
            double alpha, beta, gamma

        # assemble rays
        origin = self.pixel_origins[x, y]
        direction = self.pixel_directions[x, y]

        rays = []

        # Any pixels not at the edge of the image will have sub-sampling (anti-aliasing) applied
        if 0 < x < self._pixels[0] - 1 and 0 < y < self._pixels[1] - 1:

            v1 = self.pixel_directions[x-1, y-1]
            v2 = self.pixel_directions[x-1, y+1]
            v3 = self.pixel_directions[x+1, y+1]
            v4 = self.pixel_directions[x+1, y-1]

            for i in range(self._pixel_samples):

                # Generate new sample point in unit square
                sample_point = point_square()

                v14 = v1.slerp(v4, sample_point.x)
                v23 = v2.slerp(v3, sample_point.x)
                sample_vector = v14.slerp(v23, sample_point.y)

                ray = template.copy(origin, sample_vector.normalise())
                rays.append((ray, 1.0))

        # if at edge of image, accept aliasing
        else:
            for i in range(self._pixel_samples):

                ray = template.copy(origin, direction)

                # projected area weight is normal.incident which simplifies
                # to incident.z here as the normal is (0, 0 ,1)
                rays.append((ray, 1.0))

        return rays

    cpdef double _pixel_sensitivity(self, int x, int y):
        return self._sensitivity
