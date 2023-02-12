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


from raysect.optical.observer.sampler2d import FullFrameSampler2D
from raysect.optical.observer.pipeline import RGBPipeline2D

from raysect.core cimport RectangleSampler3D, HemisphereCosineSampler
from raysect.optical cimport Ray, AffineMatrix3D, Point3D, Vector3D, translate
from libc.math cimport M_PI
from raysect.optical.observer.base cimport Observer2D


cdef class CCDArray(Observer2D):
    """
    An observer that models an idealised CCD-like imaging sensor.

    The CCD is a regular array of square pixels. Each pixel samples red, green
    and blue channels (behaves like a Foveon imaging sensor). The CCD sensor
    width is specified with the width parameter. The CCD height is calculated
    from the width and the number of vertical and horizontal pixels. The
    default width and sensor ratio approximates a 35mm camera sensor.

    :param tuple pixels: A tuple of pixel dimensions for the camera (default=(720, 480)).
    :param float width: The CCD sensor x-width in metres (default=35mm).
    :param list pipelines: The list of pipelines that will process the spectrum measured
      at each pixel by the camera (default=RGBPipeline2D()).
    :param kwargs: **kwargs and properties from Observer2D and _ObserverBase.
    """

    def __init__(self, pixels=(720, 480), width=0.035, parent=None, transform=None, name=None, pipelines=None):

        # initial values to prevent undefined behaviour when setting via self.width
        self._width = 0.035
        self._pixels = (720, 480)

        pipelines = pipelines or [RGBPipeline2D()]

        super().__init__(pixels, FullFrameSampler2D(), pipelines, parent=parent, transform=transform, name=name)

        # setting width triggers calculation of image geometry calculations
        self.width = width
        self.vector_sampler = HemisphereCosineSampler()

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, value):
        pixels = tuple(value)
        if len(pixels) != 2:
            raise ValueError("Pixels must be a 2 element tuple defining the x and y resolution.")
        x, y = pixels
        if x <= 0:
            raise ValueError("Number of x pixels must be greater than 0.")
        if y <= 0:
            raise ValueError("Number of y pixels must be greater than 0.")
        self._pixels = pixels
        self._update_image_geometry()

    @property
    def width(self):
        """
        The CCD sensor x-width in metres.

        :rtype: float
        """
        return self._width

    @width.setter
    def width(self, width):
        if width <= 0:
            raise ValueError("width can not be less than or equal to 0 meters.")
        self._width = width
        self._update_image_geometry()

    cdef object _update_image_geometry(self):

        self.image_delta = self._width / self._pixels[0]
        self.image_start_x = 0.5 * self._pixels[0] * self.image_delta
        self.image_start_y = 0.5 * self._pixels[1] * self.image_delta
        self.point_sampler = RectangleSampler3D(self.image_delta, self.image_delta)
        self._pixel_area = (self._width / self._pixels[0])**2

    cpdef list _generate_rays(self, int ix, int iy, Ray template, int ray_count):

        cdef:
            double pixel_x, pixel_y
            list origin_points, direction_vectors, rays
            Point3D origin
            Vector3D direction
            Ray ray
            AffineMatrix3D pixel_to_local

        # generate pixel transform
        pixel_x = self.image_start_x - self.image_delta * (ix + 0.5)
        pixel_y = self.image_start_y - self.image_delta * (iy + 0.5)
        pixel_to_local = translate(pixel_x, pixel_y, 0)

        # generate origin and direction vectors
        origin_points = self.point_sampler.samples(ray_count)
        direction_vectors = self.vector_sampler.samples(ray_count)

        # assemble rays
        rays = []
        for origin, direction in zip(origin_points, direction_vectors):

            # transform to local space from pixel space
            origin = origin.transform(pixel_to_local)
            direction = direction.transform(pixel_to_local)

            ray = template.copy(origin, direction)

            # cosine weighted distribution
            # projected area cosine is implicit in distribution
            # weight = (1 / 2*pi) * (pi / cos(theta)) * cos(theta) = 0.5
            rays.append((ray, 0.5))

        return rays

    cpdef double _pixel_sensitivity(self, int x, int y):
        return self._pixel_area * 2 * M_PI
