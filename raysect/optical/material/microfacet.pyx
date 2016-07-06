# cython: language_level=3

# Copyright (c) 2015, Dr Alex Meakins, Raysect Project
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
from raysect.optical cimport Point3D, Vector3D, AffineMatrix3D, Primitive, World, Ray, Spectrum, SpectralFunction, ConstantSF, Normal3D
from raysect.optical.material cimport ContinuousBSDF
from numpy cimport ndarray
from libc.math cimport M_1_PI, M_PI, fabs
cimport cython


# cdef class FacetDistribution:
#
#     cpdef double pdf(self, Vector3D incoming, Vector3D outgoing, bint back_face):
#         raise NotImplementedError()
#
#     cpdef Vector3D sample(self, Vector3D incoming, bint back_face):
#         raise NotImplementedError()
#
#
# cdef class FacetReflectivity:
#     pass
#
#
# cdef class FresnelTerm:
#     pass


# Implemented as a rough metal for now.
cdef class MicroFacet(ContinuousBSDF):
    """
    This is implementing Torence Sparrow.
    """

    # cdef FacetDistribution distribution

    cpdef double pdf(self, Vector3D incoming, Vector3D outgoing, bint back_face):
        return self.distribution.pdf(incoming, outgoing, back_face)


    cpdef Vector3D sample(self, Vector3D incoming, bint back_face):
        return self.distribution.sample(incoming, back_face)

    # TODO - s_incoming and s_outgoing are back to front. s_outgoing should be direction back to observer.
    cpdef Spectrum evaluate_shading(self, World world, Ray ray, Vector3D s_incoming, Vector3D s_outgoing,
                                    Point3D w_inside_point, Point3D w_outside_point, bint back_face,
                                    AffineMatrix3D world_to_surface, AffineMatrix3D surface_to_world):

        cdef:
            Vector3D s_half
            Spectrum spectrum
            Ray reflected
            ndarray reflectivity

        # are incident and reflected on the same side?
        if (back_face and s_outgoing.z >= 0) or (not back_face and s_outgoing.z <= 0):
            # different sides, return empty spectrum
            return ray.new_spectrum()


        # calculate half vector
        s_half = s_incoming + s_outgoing

        # catch case when incoming and outgoing vectors are exactly opposite
        if s_half.get_length() == 0:
            return ray.new_spectrum()

        s_half = s_half.normalise()


        # generate and trace ray
        if back_face:The scattering of electromagnetic waves from rough surfaces
            reflected = ray.spawn_daughter(w_inside_point, s_outgoing.transform(surface_to_world))
        else:
            reflected = ray.spawn_daughter(w_outside_point, s_outgoing.transform(surface_to_world))

        spectrum = reflected.trace(world)


        # sample refractive index and absorption
        n = self.index.sample(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_num_samples())
        k = self.extinction.sample(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_num_samples())


        # perform Torrance-Sparrow calculation

        # calculate cosine of angle of incident vector VS surface normal accounting for side of surface
        if back_face:
            ci = -s_outgoing.z
            s_normal = Normal3D(0, 0, -1)
        else:
            ci = s_outgoing.z
            s_normal = Normal3D(0, 0, 1)

        # scalar factor for projection of incoming light
        spectrum.mul_scalar(1 / (4 * M_PI * ci))

        # Geometric masking and shadowing term
        spectrum.mul_scalar(self._g(s_incoming, s_outgoing, s_normal, s_half))

        # Fresnel reflectivity coefficients at each wavelength and apply
        s_view = spectrum.samples
        n_view = n
        k_view = k
        for i in range(spectrum.num_samples):
            s_view[i] *= self._fresnel(ci, n_view[i], k_view[i])

        # Microfacet distribution term
        spectrum.mul_scalar(self._d(s_half))

        return spectrum


    cdef double _g(self, Vector3D s_incoming, Vector3D s_outgoing, Vector3D s_normal, Vector3D s_half):

        cdef double k, a, b
        k = 2 * (s_normal.dot(s_half)) / (s_incoming.dot(s_half))
        a = k * (s_normal.dot(s_incoming))
        b = k * (s_normal.dot(s_outgoing))
        return min(1, a, b)

    @cython.cdivision(True)
    cdef inline double _f(self, double ci, double n, double k) nogil:

        cdef double c12, k0, k1, k2, k3

        ci2 = ci * ci
        k0 = n * n + k * k
        k1 = k0 * ci2 + 1
        k2 = 2 * n * ci
        k3 = k0 + ci2

        return 0.5 * ((k1 - k2) / (k1 + k2) + (k3 - k2) / (k3 + k2))

    cdef double _d(self, Vector3D s_half):
        pass


    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # no volume contribution
        return spectrum



