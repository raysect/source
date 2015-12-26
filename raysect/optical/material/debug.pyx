# cython: language_level=3

# Copyright (c) 2014-2015, Dr Alex Meakins, Raysect Project
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

"""
This module contains materials to aid with debugging.
"""

from raysect.core.math.point cimport Point3D
from raysect.core.math.normal cimport Normal3D
from raysect.optical.spectrum cimport Spectrum
from raysect.core.math.affinematrix cimport AffineMatrix3D
from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.scenegraph.world cimport World
from raysect.optical.ray cimport Ray
from raysect.optical.colour import d65_white


cdef class Light(NullVolume):
    """
    A Lambertian surface material illuminated by a distant light source.

    This debug material lights the primitive from the world direction specified
    by a vector passed to the light_direction parameter. An optional intensity
    and emission spectrum may be supplied. By default the light spectrum is the
    D65 white point spectrum.

    :param light_direction: A world space Vector3D defining the light direction.
    :param intensity: The light intensity (default is 1.0).
    :param spectrum: A SpectralFunction defining the light spectrum (default is D65 white).
    """

    def __init__(self, Vector3D light_direction, double intensity=1.0, SpectralFunction spectrum=None):

        self.light_direction = light_direction.normalise()
        self.intensity = max(0, intensity)

        if spectrum is None:
            self.spectrum = d65_white
        else:
            self.spectrum = spectrum

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D to_local, AffineMatrix3D to_world):

        cdef Spectrum spectrum

        spectrum = ray.new_spectrum()
        if self.intensity != 0.0:
            diffuse_intensity = self.intensity * max(0, -self.light_direction.transform(to_local).dot(normal))
            spectrum.samples[:] = diffuse_intensity * self.spectrum.sample_multiple(ray.min_wavelength, ray.max_wavelength, ray.num_samples)
        return spectrum