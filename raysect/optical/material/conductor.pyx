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

from numpy cimport ndarray
from raysect.core.math.random cimport uniform
from raysect.optical cimport Point3D, Normal3D, AffineMatrix3D, Primitive, World, Ray, new_vector3d, Intersection
from libc.math cimport M_PI, sqrt, fabs, atan, cos, sin
cimport cython


cdef class Conductor(Material):
    """
    Conductor material.

    The conductor material simulates the interaction of light with a
    homogeneous conducting material, such as, gold, silver or aluminium.

    This material implements the Fresnel equations for a conducting surface. To
    use the material, the complex refractive index of the conductor must be
    supplied.

    :param SpectralFunction index: Real component of the refractive
      index - :math:`n(\lambda)`.
    :param SpectralFunction extinction: Imaginary component of the
      refractive index (extinction) - :math:`k(\lambda)`.

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import InterpolatedSF
        >>> from raysect.optical.material import Conductor
        >>>
        >>> wavelength = np.array(...)
        >>> index = InterpolatedSF(wavelength, np.array(...))
        >>> extinction = InterpolatedSF(wavelength, np.array(...))
        >>>
        >>> metal = Conductor(index, extinction)
    """

    def __init__(self, SpectralFunction index, SpectralFunction extinction):

        super().__init__()
        self.index = index
        self.extinction = extinction

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef:
            Vector3D incident, reflected
            double temp, ci
            ndarray reflection_coefficient
            Ray reflected_ray
            Spectrum spectrum
            double[::1] n, k
            int i

        # convert ray direction normal to local coordinates
        incident = ray.direction.transform(world_to_primitive)

        # ensure vectors are normalised for reflection calculation
        incident = incident.normalise()
        normal = normal.normalise()

        # calculate cosine of angle between incident and normal
        ci = normal.dot(incident)

        # sample refractive index and absorption
        n = self.index.sample_mv(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_bins())
        k = self.extinction.sample_mv(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_bins())

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
        for i in range(spectrum.bins):
            spectrum.samples_mv[i] *= self._fresnel(ci, n[i], k[i])

        return spectrum

    @cython.cdivision(True)
    cdef double _fresnel(self, double ci, double n, double k) nogil:

        cdef double c12, k0, k1, k2, k3

        ci2 = ci * ci
        k0 = n * n + k * k
        k1 = k0 * ci2 + 1
        k2 = 2 * n * ci
        k3 = k0 + ci2

        return 0.5 * ((k1 - k2) / (k1 + k2) + (k3 - k2) / (k3 + k2))

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # do nothing!
        return spectrum


# TODO: generalise microfacet models
cdef class RoughConductor(ContinuousBSDF):
    """
    This is implementing Cook-Torrence with conducting fresnel microfacets.

    Smith shadowing and GGX facet distribution used to model roughness.

    :param SpectralFunction index: Real component of the refractive
      index - :math:`n(\lambda)`.
    :param SpectralFunction extinction: Imaginary component of the
      refractive index (extinction) - :math:`k(\lambda)`.
    :param float roughness: The roughness parameter in range (0, 1]. 0 is
      perfectly specular, 1 is perfectly rough.

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import InterpolatedSF
        >>> from raysect.optical.material import RoughConductor
        >>>
        >>> wavelength = np.array(...)
        >>> index = InterpolatedSF(wavelength, np.array(...))
        >>> extinction = InterpolatedSF(wavelength, np.array(...))
        >>>
        >>> rough_metal = RoughConductor(index, extinction, 0.25)
    """

    def __init__(self, SpectralFunction index, SpectralFunction extinction, double roughness):

        super().__init__()
        self.index = index
        self.extinction = extinction
        self.roughness = roughness

    @property
    def roughness(self):
        return self._roughness

    @roughness.setter
    def roughness(self, value):
        if value <= 0 or value > 1:
            raise ValueError("Surface roughness must lie in the range (0, 1].")
        self._roughness = value

    @cython.cdivision(True)
    cpdef double pdf(self, Vector3D s_incoming, Vector3D s_outgoing, bint back_face):

        cdef Vector3D s_half

        # calculate half vector
        s_half = new_vector3d(
            s_incoming.x + s_outgoing.x,
            s_incoming.y + s_outgoing.y,
            s_incoming.z + s_outgoing.z
        )

        # catch ill defined half vector
        if s_half.get_length() == 0.0:
            # should never produce a none zero BSDF value therefore safe to return zero as pdf
            return 0.0

        s_half = s_half.normalise()
        return 0.25 * self._d(s_half) * fabs(s_half.z / s_outgoing.dot(s_half))

    @cython.cdivision(True)
    cpdef Vector3D sample(self, Vector3D s_incoming, bint back_face):

        cdef:
            double e1, e2
            double theta, phi, temp
            Vector3D facet_normal

        e1 = uniform()
        e2 = uniform()

        theta = atan(self._roughness * sqrt(e1) / sqrt(1 - e1))
        phi = 2 * M_PI * e2

        facet_normal = new_vector3d(
            cos(phi) * sin(theta),
            sin(phi) * sin(theta),
            cos(theta)
        )

        temp = 2 * s_incoming.dot(facet_normal)
        return new_vector3d(
            temp * facet_normal.x - s_incoming.x,
            temp * facet_normal.y - s_incoming.y,
            temp * facet_normal.z - s_incoming.z
        )

    @cython.cdivision(True)
    cpdef Spectrum evaluate_shading(self, World world, Ray ray, Vector3D s_incoming, Vector3D s_outgoing,
                                    Point3D w_reflection_origin, Point3D w_transmission_origin, bint back_face,
                                    AffineMatrix3D world_to_surface, AffineMatrix3D surface_to_world,
                                    Intersection intersection):

        cdef:
            Vector3D s_half
            Spectrum spectrum
            Ray reflected

        # outgoing ray is sampling incident light so s_outgoing = incident

        # material does not transmit
        if s_outgoing.z <= 0:
            return ray.new_spectrum()

        # ignore parallel rays which could cause a divide by zero later
        if s_incoming.z == 0:
            return ray.new_spectrum()

        # calculate half vector
        s_half = new_vector3d(
            s_incoming.x + s_outgoing.x,
            s_incoming.y + s_outgoing.y,
            s_incoming.z + s_outgoing.z
        ).normalise()

        # generate and trace ray
        reflected = ray.spawn_daughter(w_reflection_origin, s_outgoing.transform(surface_to_world))
        spectrum = reflected.trace(world)

        # evaluate lighting with Cook-Torrance bsdf (optimised)
        spectrum.mul_scalar(self._d(s_half) * self._g(s_incoming, s_outgoing) / (4 * s_incoming.z))
        return self._f(spectrum, s_outgoing, s_half)

    @cython.cdivision(True)
    cdef double _d(self, Vector3D s_half):

        cdef double r2, h2, k

        # ggx distribution
        r2 = self._roughness * self._roughness
        h2 = s_half.z * s_half.z
        k = h2 * (r2 - 1) + 1
        return r2 / (M_PI * k * k)

    cdef double _g(self, Vector3D s_incoming, Vector3D s_outgoing):
        # Smith's geometric shadowing model
        return self._g1(s_incoming) * self._g1(s_outgoing)

    @cython.cdivision(True)
    cdef double _g1(self, Vector3D v):
        # Smith's geometric component (G1) for GGX distribution
        cdef double r2 = self._roughness * self._roughness
        return 2 * v.z / (v.z + sqrt(r2 + (1 - r2) * (v.z * v.z)))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef Spectrum _f(self, Spectrum spectrum, Vector3D s_outgoing, Vector3D s_normal):

        cdef:
            double[::1] n, k
            double ci
            int i

        # sample refractive index and absorption
        n = self.index.sample_mv(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
        k = self.extinction.sample_mv(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)

        ci = s_normal.dot(s_outgoing)
        for i in range(spectrum.bins):
            spectrum.samples_mv[i] *= self._fresnel_conductor(ci, n[i], k[i])

        return spectrum

    @cython.cdivision(True)
    cdef double _fresnel_conductor(self, double ci, double n, double k) nogil:

        cdef double c12, k0, k1, k2, k3

        ci2 = ci * ci
        k0 = n * n + k * k
        k1 = k0 * ci2 + 1
        k2 = 2 * n * ci
        k3 = k0 + ci2
        return 0.5 * ((k1 - k2) / (k1 + k2) + (k3 - k2) / (k3 + k2))

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # no volume contribution
        return spectrum
