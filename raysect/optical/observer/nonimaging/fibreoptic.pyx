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

from libc.math cimport cos, M_PI as PI

from raysect.core cimport Vector3D
from raysect.core.math.sampler cimport DiskSampler3D, ConeUniformSampler
from raysect.optical cimport Ray
from raysect.optical.observer.base cimport Observer0D
from raysect.optical.observer.pipeline.spectral import SpectralPowerPipeline0D
cimport cython


# 1 / (2 * PI)
DEF RECIP_2_PI = 0.15915494309189535


# TODO - provide a function for angular fall off for collection, instead of acceptance cone.
# TODO - current samples the cone with uniform sampling, need a cosine weighted cone sampler.
cdef class FibreOptic(Observer0D):
    """
    An optical fibre observer that samples rays from an acceptance cone and circular area at the fibre tip.

    Rays are sampled over a circular area at the fibre tip and a conical solid angle
    defined by the acceptance_angle parameter.

    :param list pipelines: The list of pipelines that will process the spectrum measured
      by this optical fibre (default=SpectralPipeline0D()).
    :param float acceptance_angle: The angle in degrees between the z axis and the cone surface which defines the fibres
       solid angle sampling area.
    :param float radius: The radius of the fibre tip in metres. This radius defines a circular area at the fibre tip
       which will be sampled over.
    :param kwargs: **kwargs from Observer0D and _ObserverBase

    .. code-block:: pycon

        >>> from raysect.optical.observer import FibreOptic, RadiancePipeline0D, PowerPipeline0D
        >>>
        >>> power = PowerPipeline0D()
        >>> radiance = RadiancePipeline0D()
        >>> fibre = FibreOptic([power, radiance], acceptance_angle=10, radius=0.0005,
                                spectral_bins=500, pixel_samples=1000,
                                transform=translate(0, 0, -5), parent=world)
        >>> fibre.observe()
    """

    def __init__(self, pipelines=None, acceptance_angle=None, radius=None, parent=None, transform=None, name=None,
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

        acceptance_angle = acceptance_angle or 5.0
        radius = radius or 0.001

        self.acceptance_angle = acceptance_angle
        self.radius = radius

    @property
    def acceptance_angle(self):
        """
        The angle in degrees between the z axis and the cone surface which defines the fibres
        solid angle sampling area.

        :rtype: float
        """
        return self._acceptance_angle

    @acceptance_angle.setter
    def acceptance_angle(self, value):
        if not 0 < value <= 90:
            raise RuntimeError("Acceptance angle must be between 0 and 90 degrees.")
        self._acceptance_angle = value
        self._vector_sampler = ConeUniformSampler(value)
        self._solid_angle = 2 * PI * (1 - cos(value / 180 * PI))

    @property
    def radius(self):
        """
        The radius of the fibre tip in metres. This radius defines a circular area at the fibre tip
        which will be sampled over.

        :rtype: float
        """
        return self._radius

    @radius.setter
    def radius(self, value):
        if value <= 0:
            raise RuntimeError("Fibre radius must be greater than 0.")
        self._radius = value
        self._point_sampler = DiskSampler3D(value)
        self._collection_area = PI * value * value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list _generate_rays(self, Ray template, int ray_count):

        cdef:
            list rays, origins, directions
            double pdf
            Vector3D direction
            int n

        origins = self._point_sampler.samples(ray_count)
        directions = self._vector_sampler.samples_with_pdfs(ray_count)

        rays = []
        for n in range(ray_count):

            # projected area weight is normal.incident which simplifies
            # to incident.z here as the normal is (0, 0 ,1)
            # weight = 1/(Omega) * 1/(omega_sample_pdf) * 1/(Area) * 1/(x_sample_pdf) * cos(theta)
            # Note: 1/area * 1/area_pdf cancels when doing uniform area point sampling
            # Note: 1/(Omega) * 1/(omega_sample_pdf) cancels when doing uniform vector sampling
            # Therefore, weight = cos(theta) term only.
            direction, pdf = directions[n]
            rays.append((template.copy(origins[n], direction), direction.z))

        return rays

    @property
    def collection_area(self):
        """
        The fibre's collection area in m^2.

        :rtype: float
        """
        return self._collection_area

    @property
    def solid_angle(self):
        """
        The fibre's solid angle in steradians str.

        :rtype: float
        """
        return self._solid_angle

    @property
    def sensitivity(self):
        """
        The fibre's sensitivity measured in units of per area per solid angle (m^-2 str^-1).

        :rtype: float
        """
        return self._pixel_sensitivity()

    cpdef double _pixel_sensitivity(self):
        return self._solid_angle * self._collection_area
