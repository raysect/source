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

from raysect.core cimport Point3D, Vector3D, new_vector3d, translate, RectangleSampler, PointSampler, AffineMatrix3D
from raysect.optical cimport Ray
from raysect.optical.observer.base cimport Observer2D


cdef class OrthographicCamera(Observer2D):
    """
    A camera observing an orthogonal (orthographic) projection of the scene, avoiding perspective effects.

    Arguments and attributes are inherited from the base Imaging sensor class.

    :param double width: width of the orthographic area to observe in meters, the height is deduced from the 'pixels'
       attribute.
    """

    cdef:
        double image_delta, image_start_x, image_start_y, _width, _etendue
        PointSampler _point_sampler

    def __init__(self, pixels, width, etendue=None, frame_sampler=None, pipelines=None, parent=None, transform=None, name=None):

        # defaults to an adaptively sampled RGB pipeline
        if not pipelines and not frame_sampler:
            rgb = RGBPipeline2D()
            pipelines = [rgb]
            frame_sampler = RGBAdaptiveSampler2D(rgb)
        else:
            pipelines = pipelines or [RGBPipeline2D()]
            frame_sampler = frame_sampler or FullFrameSampler2D()

        super().__init__(pixels, frame_sampler, pipelines, parent=parent, transform=transform, name=name)

        self.etendue = etendue or 1.0
        self.width = width

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        if width <= 0:
            raise ValueError("Width must be greater than 0.")
        self._width = width
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

        self.image_delta = self._width / self._pixels[0]
        self.image_start_x = 0.5 * self._pixels[0] * self.image_delta
        self.image_start_y = 0.5 * self._pixels[1] * self.image_delta
        self._point_sampler = RectangleSampler(self.image_delta, self.image_delta)

    cpdef list _generate_rays(self, int ix, int iy, Ray template, int ray_count):

        cdef:
            double pixel_x, pixel_y
            list points, rays
            Point3D origin
            Ray ray
            AffineMatrix3D to_local

        # generate pixel transform
        pixel_x = self.image_start_x - self.image_delta * ix
        pixel_y = self.image_start_y - self.image_delta * iy
        to_local = translate(pixel_x, pixel_y, 0)

        # generate origin and direction vectors
        points = self._point_sampler(self._pixel_samples)

        # assemble rays
        rays = []
        for origin in points:

            # transform to local space from pixel space
            origin = origin.transform(to_local)
            ray = template.copy(origin, new_vector3d(0, 0, 1))

            # rays fired along normal hence projected area weight is 1.0
            rays.append((ray, 1.0))

        return rays

    cpdef double _pixel_etendue(self, int ix, int iy):
        return self._etendue


