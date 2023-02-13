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

"""
This module contains materials to aid with debugging.
"""

from raysect.optical import d65_white
from raysect.optical cimport Point3D, Normal3D, AffineMatrix3D, Spectrum, World, Primitive, Ray, new_vector3d, Intersection
cimport cython


cdef class Light(NullVolume):
    """
    A Lambertian surface material illuminated by a distant light source.

    This debug material lights the primitive from the world direction specified
    by a vector passed to the light_direction parameter. An optional intensity
    and emission spectrum may be supplied. By default the light spectrum is the
    D65 white point spectrum.

    :param Vector3D light_direction: A world space Vector3D defining the light direction.
    :param float intensity: The light intensity in units of radiance (default is 1.0).
    :param SpectralFunction spectrum: A SpectralFunction defining the light's
      emission spectrum (default is D65 white).
    """

    def __init__(self, Vector3D light_direction, double intensity=1.0, SpectralFunction spectrum=None):

        super().__init__()
        self.light_direction = light_direction.normalise()
        self.intensity = max(0, intensity)

        if spectrum is None:
            self.spectrum = d65_white
        else:
            self.spectrum = spectrum

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef Spectrum spectrum

        # todo: optimise
        spectrum = ray.new_spectrum()
        if self.intensity != 0.0:
            diffuse_intensity = self.intensity * max(0, -self.light_direction.transform(world_to_primitive).dot(normal))
            spectrum.samples[:] = diffuse_intensity * self.spectrum.sample(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_bins())
        return spectrum


cdef class PerfectReflectingSurface(Material):
    """
    A material that is perfectly reflecting.
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef:
            Vector3D incident, reflected
            double temp, ci
            Ray reflected_ray

        # convert ray direction normal to local coordinates
        incident = ray.direction.transform(world_to_primitive)

        # ensure vectors are normalised for reflection calculation
        incident = incident.normalise()
        normal = normal.normalise()

        # calculate cosine of angle between incident and normal
        ci = normal.dot(incident)

        # reflection
        temp = 2 * ci
        reflected = new_vector3d(incident.x - temp * normal.x,
                                 incident.y - temp * normal.y,
                                 incident.z - temp * normal.z)

        # convert reflected ray direction to world space
        reflected = reflected.transform(primitive_to_world)

        # spawn reflected ray and trace
        # note, we do not use the supplied exiting parameter as the normal is
        # not guaranteed to be perpendicular to the surface for meshes
        if ci > 0.0:

            # incident ray is pointing out of surface, reflection is therefore inside
            reflected_ray = ray.spawn_daughter(inside_point.transform(primitive_to_world), reflected)

        else:

            # incident ray is pointing in to surface, reflection is therefore outside
            reflected_ray = ray.spawn_daughter(outside_point.transform(primitive_to_world), reflected)

        return reflected_ray.trace(world)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # do nothing!
        return spectrum

