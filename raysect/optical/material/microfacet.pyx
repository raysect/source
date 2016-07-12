# cython: language_level=3

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

from raysect.core.math.random cimport vector_hemisphere_cosine
from raysect.optical cimport Point3D, Vector3D, AffineMatrix3D, Primitive, World, Ray, Spectrum, SpectralFunction, ConstantSF, new_vector3d
from raysect.optical.material cimport ContinuousBSDF
from numpy cimport ndarray
from libc.math cimport M_1_PI, M_PI, sqrt
cimport cython


# Implemented as a rough metal for now.
cdef class MicroFacet(ContinuousBSDF):
    """
    This is implementing Torrence Sparrow.
    """

    cdef:
        public SpectralFunction index
        public SpectralFunction extinction
        double _roughness

    def __init__(self, SpectralFunction index, SpectralFunction extinction, double roughness):

        # todo: add validation
        super().__init__()
        self.index = index
        self.extinction = extinction
        self._roughness = roughness

    cpdef double pdf(self, Vector3D s_incoming, Vector3D s_outgoing, bint back_face):
        """
        temporarily cosine weighted hemispherical sampling
        not a great strategy for this...!
        """

        # todo: replace with ggx importance sampling pdf
        cdef double cos_theta

        # normal is aligned with +ve Z so dot products with the normal are simply the z component of the other vector
        cos_theta = s_outgoing.z

        # clamp probability to zero on far side of surface
        if cos_theta < 0:
            return 0

        return cos_theta * M_1_PI

    cpdef Vector3D sample(self, Vector3D s_incoming, bint back_face):
        """
        temporarily cosine weighted hemispherical sampling
        not a great strategy for this...!
        """

        # todo: replace with ggx importance sampling
        return vector_hemisphere_cosine()

    @cython.cdivision(True)
    cpdef Spectrum evaluate_shading(self, World world, Ray ray, Vector3D s_incoming, Vector3D s_outgoing,
                                    Point3D w_reflection_origin, Point3D w_transmission_origin, bint back_face,
                                    AffineMatrix3D world_to_surface, AffineMatrix3D surface_to_world):

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
        return self._f(spectrum, s_outgoing)

    @cython.cdivision(True)
    cdef inline double _d(self, Vector3D s_half):

        cdef double r2, h2, k

        # ggx distribution
        r2 = self._roughness * self._roughness
        h2 = s_half.z * s_half.z
        k = h2 * (r2 - 1) + 1
        return r2 / (M_PI * k * k)

    cdef inline double _g(self, Vector3D s_incoming, Vector3D s_outgoing):
        # Smith's geometric shadowing model
        return self._g1(s_incoming) * self._g1(s_outgoing)

    @cython.cdivision(True)
    cdef inline double _g1(self, Vector3D v):
        # Smith's geometric component (G1) for GGX distribution
        cdef double r2 = self._roughness * self._roughness
        return 2 * v.z / (v.z + sqrt(r2 + (1 - r2) * (v.z * v.z)))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline Spectrum _f(self, Spectrum spectrum, Vector3D s_outgoing):

        cdef:
            double[::1] s, n, k
            int i

        # sample refractive index and absorption
        n = self.index.sample(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.num_samples)
        k = self.extinction.sample(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.num_samples)

        s = spectrum.samples
        for i in range(spectrum.num_samples):
            s[i] *= self._fresnel_conductor(s_outgoing.z, n[i], k[i])

        return spectrum

    @cython.cdivision(True)
    cdef inline double _fresnel_conductor(self, double ci, double n, double k) nogil:

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