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

from raysect.optical.observer import SpectralPowerPipeline0D

from raysect.optical cimport Ray, new_point3d, new_vector3d
from raysect.optical.observer.base cimport Observer0D


# TODO: complete docstrings
cdef class SightLine(Observer0D):
    """
    A simple line of sight observer.

    Fires a single ray oriented along the observer's z axis in world space.

    :param float sensitivity: Optional user specified sensitivity. Defaults to sensitivity=1.0
      in which case the returned units will always be in radiance (W/m^2/str/nm)
    :param list pipelines: The list of pipelines that will process the spectrum measured
      by this line of sight (default=SpectralPipeline0D()).
    :param kwargs: **kwargs and instance properties from Observer0D and _ObserverBase

    .. code-block:: pycon

        >>> from raysect.optical import World
        >>> from raysect.optical.observer import SightLine, PowerPipeline0D
        >>>
        >>> world = World()
        >>> power = PowerPipeline0D(accumulate=False)
        >>> los = SightLine([power], min_wavelength=400, max_wavelength=720,
                            parent=world, transform=rotate(0, 0, 0)*translate(0, 0, -1))
        >>> los.observe()
    """

    def __init__(self, sensitivity=None, pipelines=None, parent=None, transform=None, name=None,
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

        self.sensitivity = sensitivity or 1.0

    @property
    def sensitivity(self):
        """
        User specified sensitivity (str^-1/m^-2)

        If sensitivity=1.0 the spectral units will always be in radiance (W/m^2/str/nm)

        :rtype: float
        """
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        if value <= 0:
            raise ValueError('Sensitivity must be greater than zero.')
        self._sensitivity = value

    cpdef list _generate_rays(self, Ray template, int ray_count):

        cdef:
            list rays
            int n

        rays = []
        for n in range(ray_count):
            rays.append((template.copy(new_point3d(0, 0, 0), new_vector3d(0, 0, 1)), 1.0))
        return rays

    cpdef double _pixel_sensitivity(self):
        return self._sensitivity
