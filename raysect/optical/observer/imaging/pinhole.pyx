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

from raysect.optical.observer.sampler2d import FullFrameSampler2D
from raysect.optical.observer.pipeline import RGBPipeline2D, RGBAdaptiveSampler2D

from raysect.core cimport Point3D, new_point3d, Vector3D, new_vector3d, PointSampler, RectangleSampler
from raysect.optical cimport Ray
from libc.math cimport M_PI as pi, tan
from raysect.optical.observer.base cimport Observer2D


# todo: complete docstrings
# todo: add properties
cdef class PinholeCamera(Observer2D):
    """
    An observer that models an idealised pinhole camera.

    A simple camera that launches rays from the observer's origin point over a
    specified field of view.

    :param double fov: The field of view of the camera in degrees (default: 45 degrees).
    :param double etendue: The etendue of each pixel (default: 1.0)
    """

    cdef:
        double _etendue, _fov, image_delta, image_start_x, image_start_y
        PointSampler point_sampler

    def __init__(self, pixels, fov=None, etendue=None, frame_sampler=None, pipelines=None, parent=None, transform=None, name=None):

        # defaults to an adaptively sampled RGB pipeline
        if not pipelines and not frame_sampler:
            rgb = RGBPipeline2D()
            pipelines = [rgb]
            frame_sampler = RGBAdaptiveSampler2D(rgb)
        else:
            pipelines = pipelines or [RGBPipeline2D()]
            frame_sampler = frame_sampler or FullFrameSampler2D()

        super().__init__(pixels, frame_sampler, pipelines, parent=parent, transform=transform, name=name)

        # note that the fov property triggers a call to _update_image_geometry()
        self.fov = fov or 45
        self.etendue = etendue or 1.0

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        if value <= 0 or value > 90:
            raise ValueError("The field-of-view angle must lie in the range (0, 90].")
        self._fov = value
        self._update_image_geometry()

    @property
    def etendue(self):
        return self._etendue

    @etendue.setter
    def etendue(self, value):
        if value <= 0:
            raise ValueError("Etendue must be greater than zero.")
        self._etendue = value

    cdef inline object _update_image_geometry(self):

        max_pixels = max(self.pixels)

        if max_pixels > 1:

            # Get width of image plane at a distance of 1m from aperture.
            image_max_width = 2 * tan(pi / 180 * 0.5 * self._fov)

            # set pixel step size in image plane
            self.image_delta = image_delta = image_max_width / max_pixels

            self.image_start_x = 0.5 * self.pixels[0] * image_delta
            self.image_start_y = 0.5 * self.pixels[1] * image_delta

            # rebuild point generator
            self.point_sampler = RectangleSampler(self.image_delta, self.image_delta)

        else:
            raise RuntimeError("Number of Pinhole camera Pixels must be > 1.")

    cpdef list _generate_rays(self, int x, int y, Ray template, int ray_count):

        cdef:
            double pixel_x, pixel_y
            list points, rays
            Point3D pixel_centre, point, origin
            Vector3D direction
            Ray ray

        # generate pixel transform
        pixel_x = self.image_start_x - self.image_delta * x
        pixel_y = self.image_start_y - self.image_delta * y
        pixel_centre = new_point3d(pixel_x, pixel_y, 1)

        points = self.point_sampler(ray_count)

        # assemble rays
        rays = []
        for point in points:

            # calculate point in virtual image plane to be used for ray direction
            origin = new_point3d(0, 0, 0)
            direction = new_vector3d(
                point.x + pixel_centre.x,
                point.y + pixel_centre.y,
                point.z + pixel_centre.z
            ).normalise()

            ray = template.copy(origin, direction)

            # projected area weight is normal.incident which simplifies
            # to incident.z here as the normal is (0, 0 ,1)
            rays.append((ray, direction.z))

        return rays

    cpdef double _pixel_etendue(self, int x, int y):
        return self._etendue
