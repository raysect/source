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
from raysect.optical cimport Point3D, Vector3D, AffineMatrix3D, Primitive, World, Ray, Spectrum, SpectralFunction, ConstantSF
from raysect.optical.material cimport ContinuousBSDF
from numpy cimport ndarray
from libc.math cimport M_1_PI, fabs
cimport cython


cdef class FacetDistribution:

    cpdef double pdf(self, Vector3D incoming, Vector3D outgoing, bint back_face):
        raise NotImplementedError()

    cpdef Vector3D sample(self, Vector3D incoming, bint back_face):
        raise NotImplementedError()


cdef class FacetReflectivity:
    pass


cdef class FresnelTerm:
    pass


# Implemented as a rough metal for now.
cdef class MicroFacet(ContinuousBSDF):
    """
    This is implementing Torence Sparrow.
    """

    cdef FacetDistribution distribution

    cpdef double pdf(self, Vector3D incoming, Vector3D outgoing, bint back_face):
        return self.distribution.pdf(incoming, outgoing, back_face)


    cpdef Vector3D sample(self, Vector3D incoming, bint back_face):
        return self.distribution.sample(incoming, back_face)

    cpdef Spectrum evaluate_shading(self, World world, Ray ray, Vector3D s_incoming, Vector3D s_outgoing,
                                    Point3D w_inside_point, Point3D w_outside_point, bint back_face,
                                    AffineMatrix3D world_to_surface, AffineMatrix3D surface_to_world): #, bint back_face

        cdef:
            Spectrum spectrum
            Ray reflected
            ndarray reflectivity



        # # are incident and reflected on the same side?
        # if (back_face and s_outgoing.z >= 0) or (not back_face and s_outgoing.z <= 0):
        #     # different sides, return empty spectrum
        #     return ray.new_spectrum()
        #
        # # generate and trace ray
        # if back_face:
        #     reflected = ray.spawn_daughter(w_inside_point, s_outgoing.transform(surface_to_world))
        # else:
        #     reflected = ray.spawn_daughter(w_outside_point, s_outgoing.transform(surface_to_world))
        #
        # spectrum = reflected.trace(world)
        #
        # # obtain samples of reflectivity
        # reflectivity = self.reflectivity.sample(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.num_samples)
        # spectrum.mul_array(reflectivity)
        # spectrum.mul_scalar(fabs(s_outgoing.z) * M_1_PI)
        # return spectrum

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # no volume contribution
        return spectrum



