# cython: language_level=3

# Copyright (c) 2014-2017, Dr Alex Meakins, Raysect Project
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

from libc.math cimport cos, M_PI as pi

from raysect.core.math.sampler cimport DiskSampler, ConeSampler
from raysect.optical cimport Ray, new_point3d, new_vector3d
from raysect.optical.observer.base cimport Observer0D
cimport cython


# TODO - provide a function for angular fall off for collection, instead of acceptance cone.
# TODO - current samples the cone with uniform sampling, need a cosine weighted cone sampler.
cdef class FibreOptic(Observer0D):
    """
    An optical fibre observer that samples rays from an acceptance cone and circular area at the fibre tip.
    Inherits arguments and attributes from the base NonImaging sensor class. Rays are sampled over a circular area at
    the fibre tip and a conical solid angle defined by the acceptance_angle parameter.
    :param float acceptance_angle: The angle in degrees between the z axis and the cone surface which defines the fibres
       soild angle sampling area.
    :param float radius: The radius of the fibre tip in metres. This radius defines a circular area at the fibre tip
       which will be sampled over.
    """

    cdef:
        double _acceptance_angle, _radius, _solid_angle, _collection_area
        DiskSampler _point_sampler
        ConeSampler _vector_sampler

    def __init__(self, pipelines, acceptance_angle=None, radius=None, parent=None, transform=None, name=None,
                 render_engine=None, pixel_samples=None, samples_per_task=None, spectral_rays=None, spectral_bins=None,
                 min_wavelength=None, max_wavelength=None, ray_extinction_prob=None, ray_extinction_min_depth=None,
                 ray_max_depth=None, ray_importance_sampling=None, ray_important_path_weight=None):

        super().__init__(pipelines, parent=parent, transform=transform, name=name, render_engine=render_engine,
                         pixel_samples=pixel_samples, samples_per_task=samples_per_task, spectral_rays=spectral_rays,
                         spectral_bins=spectral_bins, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                         ray_extinction_prob=ray_extinction_prob, ray_extinction_min_depth=ray_extinction_min_depth,
                         ray_max_depth=ray_max_depth, ray_importance_sampling=ray_importance_sampling,
                         ray_important_path_weight=ray_important_path_weight)

        acceptance_angle = acceptance_angle or 5
        radius = radius or 0.001

        self.acceptance_angle = acceptance_angle
        self.radius = radius

    @property
    def acceptance_angle(self):
        return self._acceptance_angle

    @acceptance_angle.setter
    def acceptance_angle(self, value):
        if not 0 <= value <= 90:
            raise RuntimeError("Acceptance angle must be between 0 and 90 degrees.")
        self._acceptance_angle = value
        self._vector_sampler = ConeSampler(value)
        self._solid_angle = 2 * pi * (1 - cos(value/180*pi))

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value <= 0:
            raise RuntimeError("Fibre radius must be greater than 0.")
        self._radius = value
        self._point_sampler = DiskSampler(value)
        self._collection_area = pi * value * value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list _generate_rays(self, Ray template, int ray_count):

        cdef:
            list rays, origins, directions
            int n
            double weight

        origins = self._point_sampler(ray_count)
        directions = self._vector_sampler(ray_count)

        rays = []
        for n in range(ray_count):

            # projected area weight is normal.incident which simplifies
            # to incident.z here as the normal is (0, 0 ,1)
            weight = directions[n].z
            rays.append((template.copy(origins[n], directions[n]), weight))

        return rays

    cpdef double _pixel_etendue(self):
        return self._solid_angle * self._collection_area
