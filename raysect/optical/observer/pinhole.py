# cython: language_level=3

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

from math import tan, pi as PI
import numpy as np

from raysect.core import Point3D, Vector3D
from .point_generator import Rectangle
from .sensor import Imaging


class PinholeCamera(Imaging):
    """
    An observer that models an idealised pinhole camera.

    A simple camera that launches rays from the observer's origin point over a
    specified field of view.

    Arguments and attributes are inherited from the base Imaging sensor class.

    :param double fov: The field of view of the camera in degrees (default is 90 degrees).
    """

    def __init__(self, pixels=(512, 512), fov=90, sensitivity=1.0, spectral_samples=21, spectral_rays=1,
                 pixel_samples=100, process_count=0, parent=None, transform=None, name=None):

        super().__init__(pixels=pixels, sensitivity=sensitivity, spectral_samples=spectral_samples,
                         spectral_rays=spectral_rays, pixel_samples=pixel_samples, process_count=process_count,
                         parent=parent, transform=transform, name=name)

        self._fov = fov
        self._update_image_geometry()
        self.point_generator = Rectangle(self.image_delta, self.image_delta)

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):
        if len(pixels) != 2:
            raise ValueError("Pixel dimensions of camera frame-buffer must be a tuple "
                             "containing the x and y pixel counts.")
        self._pixels = pixels

        # reset frames
        self.xyz_frame = np.zeros((self._pixels[1], self._pixels[0], 3))
        self.rgb_frame = np.zeros((self._pixels[1], self._pixels[0], 3))
        self.accumulated_samples = 0

        # update pixel geometry
        self._update_image_geometry()

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, fov):
        if fov <= 0:
            raise ValueError("Field of view angle can not be less than or equal to 0 degrees.")
        self._fov = fov

    def _update_image_geometry(self):

        max_pixels = max(self._pixels)

        if max_pixels > 1:

            # Get width of image plane at a distance of 1m from aperture.
            image_max_width = 2 * tan(PI / 180 * 0.5 * self._fov)

            # set pixel step size in image plane
            self.image_delta = image_delta = image_max_width / max_pixels

            self.image_start_x = 0.5 * self._pixels[1] * image_delta
            self.image_start_y = 0.5 * self._pixels[0] * image_delta

        else:
            raise RuntimeError("Number of Pinhole camera Pixels must be > 1.")

    def _generate_rays(self, ix, iy, ray_template):

        # generate pixel transform
        pixel_x = self.image_start_x - self.image_delta * ix
        pixel_y = self.image_start_y - self.image_delta * iy
        pixel_centre = Point3D(pixel_x, pixel_y, 1)

        points = self.point_generator(self.pixel_samples)

        # assemble rays
        rays = []
        for point in points:

            # calculate point in virtual image plane to be used for ray direction
            origin = Point3D()
            direction = Vector3D(
                point.x + pixel_centre.x,
                point.y + pixel_centre.y,
                point.z + pixel_centre.z
            ).normalise()

            ray = ray_template.copy(origin, direction)

            # projected area weight is normal.incident which simplifies
            # to incident.z here as the normal is (0, 0 ,1)
            rays.append((ray, direction.z))

        return rays
