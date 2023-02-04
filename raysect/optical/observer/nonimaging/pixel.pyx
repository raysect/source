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

from libc.math cimport M_PI as PI

from raysect.core.math.sampler cimport RectangleSampler3D, HemisphereCosineSampler
from raysect.optical cimport Ray
from raysect.optical.observer.base cimport Observer0D
from raysect.optical.observer.pipeline.spectral import SpectralPowerPipeline0D
cimport cython


cdef class Pixel(Observer0D):
    """
    A pixel observer that samples rays from a hemisphere and rectangular area.

    :param list pipelines: The list of pipelines that will process the spectrum measured
      by this pixel (default=SpectralPipeline0D()).
    :param float x_width: The rectangular collection area's width along the
      x-axis in local coordinates (default=1cm).
    :param float y_width: The rectangular collection area's width along the
      y-axis in local coordinates (default=1cm).
    :param kwargs: **kwargs from Observer0D and _ObserverBase

    .. code-block:: pycon

        >>> from raysect.optical import World
        >>> from raysect.optical.observer import Pixel, PowerPipeline0D
        >>>
        >>> world = World()
        >>> power = PowerPipeline0D(accumulate=False)
        >>> observing_plane = Pixel([power], x_width=2.0, y_width=2.0,
                                    min_wavelength=400, max_wavelength=720,
                                    spectral_bins=1, pixel_samples=250,
                                    parent=world, transform=rotate(0, 0, 0)*translate(0, 0, -1))
        >>> observing_plane.observe()
    """

    def __init__(self, pipelines=None, x_width=None, y_width=None, parent=None, transform=None, name=None,
                 render_engine=None, pixel_samples=None, samples_per_task=None, spectral_rays=None, spectral_bins=None,
                 min_wavelength=None, max_wavelength=None, ray_extinction_prob=None, ray_extinction_min_depth=None,
                 ray_max_depth=None, ray_importance_sampling=None, ray_important_path_weight=None, quiet=False):

        pipelines = pipelines or [SpectralPowerPipeline0D()]

        super().__init__(pipelines, parent=parent, transform=transform, name=name, render_engine=render_engine,
                         pixel_samples=pixel_samples, samples_per_task=samples_per_task, spectral_rays=spectral_rays,
                         spectral_bins=spectral_bins, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                         ray_extinction_prob=ray_extinction_prob, ray_extinction_min_depth=ray_extinction_min_depth,
                         ray_max_depth=ray_max_depth, ray_importance_sampling=ray_importance_sampling,
                         ray_important_path_weight=ray_important_path_weight, quiet=quiet)

        self._x_width = 0.01
        self._y_width = 0.01
        self._vector_sampler = HemisphereCosineSampler()
        self._solid_angle = 2 * PI

        self.x_width = x_width or 0.01
        self.y_width = y_width or 0.01

    @property
    def x_width(self):
        """
        The rectangular collection area's width along the x-axis in local coordinates.

        :rtype: float
        """
        return self._x_width

    @x_width.setter
    def x_width(self, value):
        if value <= 0:
            raise RuntimeError("Pixel x-width must be greater than zero.")
        self._x_width = value
        self._point_sampler = RectangleSampler3D(width=self._x_width, height=self._y_width)
        self._collection_area = self._x_width * self._y_width

    @property
    def y_width(self):
        """
        The rectangular collection area's width along the y-axis in local coordinates.

        :rtype: float
        """
        return self._y_width

    @y_width.setter
    def y_width(self, value):
        if value <= 0:
            raise RuntimeError("Pixel y-width must be greater than zero.")
        self._y_width = value
        self._point_sampler = RectangleSampler3D(width=self._x_width, height=self._y_width)
        self._collection_area = self._x_width * self._y_width

    @property
    def collection_area(self):
        """
        The pixel's collection area in m^2.

        :rtype: float
        """
        return self._collection_area

    @property
    def solid_angle(self):
        """
        The pixel's solid angle in steradians str.

        :rtype: float
        """
        return self._solid_angle

    @property
    def sensitivity(self):
        """
        The pixel's sensitivity measured in units of per area per solid angle (m^-2 str^-1).

        :rtype: float
        """
        return self._pixel_sensitivity()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list _generate_rays(self, Ray template, int ray_count):

        cdef:
            list rays, origins, directions
            int n
            double weight

        origins = self._point_sampler.samples(ray_count)
        directions = self._vector_sampler.samples(ray_count)

        rays = []
        for n in range(ray_count):

            # cosine weighted distribution
            # projected area cosine is implicit in distribution
            # weight = 1 / (2 * pi) * (pi / cos(theta)) * cos(theta) = 0.5
            rays.append((template.copy(origins[n], directions[n]), 0.5))

        return rays

    cpdef double _pixel_sensitivity(self):
        return self._solid_angle * self._collection_area
