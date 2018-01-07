# Copyright (c) 2017, Dr Alex Meakins, Raysect Project
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

from raysect.optical.observer.sampler2d import FullFrameSampler2D, RGBAdaptiveSampler2D
from raysect.optical.observer.pipeline import RGBPipeline2D

import numpy as np

from raysect.core cimport Point3D, Vector3D
from raysect.optical cimport Ray
from raysect.optical.observer.base cimport Observer2D
cimport numpy as np


cdef class OpenCVCamera(Observer2D):
    """
    An observer based on the OpenCV camera model.

    A simple analytic camera that uses calibrated camera parameters to re-generate the
    pixel vectors. The following parameters need to be supplied.

    * pinhole and barrel distortion terms :math:`(k_1, k_2, p_1, p_2, k_3)`.
    * focal lengths :math:`(f_x, f_y)` in pixel coordinates.
    * optical centres :math:`(c_x, c_y)` in pixel coordinates.

    See the `OpenCV documentation
    <https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html>`_ for more details.

    Arguments and attributes are inherited from the base Observer2D sensor class.

    :param tuple distortion: pinhole and barrel distortion terms :math:`(k_1, k_2, p_1, p_2, k_3)`.
    :param tuple focal_lengths: focal lengths :math:`(f_x, f_y)` in pixel coordinates.
    :param tuple focal_lengths: optical centres :math:`(c_x, c_y)` in pixel coordinates.
    :param FrameSampler2D frame_sampler: The frame sampling strategy (default=FullFrameSampler2D()).
    :param list pipelines: The list of pipelines that will process the spectrum measured
      at each pixel by the camera (default=RGBPipeline2D()).
    :param kwargs: **kwargs and properties from Observer2D and _ObserverBase.
    """

    cdef:
        double image_delta, image_start_x, image_start_y

    def __init__(self, distortion, focal_lengths, optical_centres, frame_sampler=None, pipelines=None, etendue=None, parent=None, transform=None, name=None):
        pass

    #     # defaults to an adaptively sampled RGB pipeline
    #     if not pipelines and not frame_sampler:
    #         rgb = RGBPipeline2D()
    #         pipelines = [rgb]
    #         frame_sampler = RGBAdaptiveSampler2D(rgb)
    #     else:
    #         pipelines = pipelines or [RGBPipeline2D()]
    #         frame_sampler = frame_sampler or FullFrameSampler2D()
    #
    #     pixel_origins = np.array(pixel_origins)
    #     pixel_directions = np.array(pixel_directions)
    #     pixels = pixel_origins.shape
    #
    #     if len(pixels) != 2:
    #         raise ValueError("Pixel arrays must have 2 dimensions.")
    #
    #     if pixel_origins.shape != pixel_directions.shape:
    #         raise ValueError("Pixel arrays must have equal shapes.")
    #
    #     super().__init__(pixel_origins.shape, frame_sampler, pipelines, parent=parent, transform=transform, name=name)
    #
    #     # camera configuration
    #     self.pixel_origins = pixel_origins
    #     self.pixel_directions = pixel_directions
    #     self._etendue = etendue or 1.0
    #
    # @property
    # def etendue(self):
    #     """
    #     The etendue applied to each pixel.
    #
    #     If etendue=1.0 all spectral units are in radiance.
    #
    #     :rtype: float
    #     """
    #     return self._etendue
    #
    # @etendue.setter
    # def etendue(self, value):
    #     if value <= 0:
    #         raise ValueError("Etendue must be greater than zero.")
    #     self._etendue = value
    #
    # cpdef list _generate_rays(self, int x, int y, Ray template, int ray_count):
    #
    #     cdef:
    #         list rays
    #         Point3D origin
    #         Vector3D direction
    #         Ray ray
    #         int i
    #
    #     # assemble rays
    #     origin = self.pixel_origins[x, y]
    #     direction = self.pixel_directions[x, y]
    #
    #     # assemble rays
    #     rays = []
    #     for i in range(self._pixel_samples):
    #
    #         ray = template.copy(origin, direction)
    #
    #         # projected area weight is normal.incident which simplifies
    #         # to incident.z here as the normal is (0, 0 ,1)
    #         rays.append((ray, 1.0))
    #
    #     return rays
    #
    # cpdef double _pixel_etendue(self, int x, int y):
    #     return self._etendue
