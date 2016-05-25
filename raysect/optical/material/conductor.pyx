# cython: language_level=3

# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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

cimport cython
from numpy import array
from numpy cimport ndarray
from libc.math cimport fabs
from raysect.optical cimport Point3D, Vector3D, new_vector3d, Normal3D, AffineMatrix3D, World, Primitive, InterpolatedSF, Spectrum, Ray


cdef class Conductor(Material):
    """
    Conductor material.

    The conductor material simulates the interaction of light with a
    homogeneous conducting material, such as, gold, silver or aluminium.

    This material implements the Fresnel equations for a conducting surface. To
    use the material, the complex refractive index of the conductor must be
    supplied.

    :param SpectralFunction index: Real component of refractive index - $n(\lambda)$.
    :param extinction: Imaginary component of refractive index (extinction) - $k(\lambda)$.
    """

    def __init__(self, SpectralFunction index, SpectralFunction extinction):

        super().__init__()
        self.index = index
        self.extinction = extinction

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Vector3D incident, reflected
            double temp, ci
            ndarray n, k, reflection_coefficient
            Ray reflected_ray
            Spectrum spectrum
            double[::1] s_view, n_view, k_view
            int i

        # convert ray direction normal to local coordinates
        incident = ray.direction.transform(world_to_primitive)

        # ensure vectors are normalised for reflection calculation
        incident = incident.normalise()
        normal = normal.normalise()

        # calculate cosine of angle between incident and normal
        ci = normal.dot(incident)

        # sample refractive index and absorption
        n = self.index.sample_multiple(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_num_samples())
        k = self.extinction.sample_multiple(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_num_samples())

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

        spectrum = reflected_ray.trace(world)

        # calculate reflection coefficients at each wavelength and apply
        ci = fabs(ci)
        s_view = spectrum.samples
        n_view = n
        k_view = k
        for i in range(spectrum.num_samples):
            s_view[i] *= self._fresnel(ci, n_view[i], k_view[i])

        return spectrum

    @cython.cdivision(True)
    cdef inline double _fresnel(self, double ci, double n, double k) nogil:

        cdef double c12, k0, k1, k2, k3

        ci2 = ci * ci
        k0 = n * n + k * k
        k1 = k0 * ci2 + 1
        k2 = 2 * n * ci
        k3 = k0 + ci2

        return 0.5 * ((k1 - k2) / (k1 + k2) + (k3 - k2) / (k3 + k2))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # do nothing!
        # TODO: make it solid - return black or calculate attenuation from extinction?
        return spectrum


