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

from raysect.core.math.affinematrix cimport AffineMatrix3D
from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.math.point cimport Point3D
from raysect.core.math.vector cimport Vector3D
from raysect.core.math.normal cimport Normal3D
from raysect.core.math.random cimport vector_hemisphere_cosine
from raysect.core.math.cython cimport transform
from raysect.core.intersection cimport Intersection
from raysect.optical.material.material cimport ContinuousPDF
from raysect.optical.scenegraph.world cimport World
from raysect.optical.ray cimport Ray
from raysect.optical.spectrum cimport Spectrum
from raysect.optical.spectralfunction cimport SpectralFunction, ConstantSF
from numpy cimport ndarray
from libc.math cimport M_PI as PI, fabs
cimport cython

cdef class Lambert(ContinuousPDF):

    cdef SpectralFunction reflectivity

    def __init__(self, SpectralFunction reflectivity=None):

        super().__init__()
        if reflectivity is None:
            reflectivity = ConstantSF(0.5)
        self.reflectivity = reflectivity

    # TODO: optimise this code
    cpdef double pdf(self, Vector3D incoming, Vector3D outgoing):

        if incoming.z > 0:
            # THis means ray entering surface
            if outgoing.z < 0:
                return 0

            return outgoing.z

        else:
            # Ray is leaving surface
            if outgoing.z > 0:
                return 0

            return -outgoing.z

    # TODO: fix documentation
    cpdef Vector3D sample(self, Vector3D incoming):

        # obtain new surface space vector from cosine-weighted hemisphere
        outgoing = vector_hemisphere_cosine()

        if incoming.z < 0:
            return outgoing.neg()
        return outgoing

    # TODO: add inside and outside_point to arguments
    cpdef Spectrum evaluate_shading(self, World world, Ray ray, Vector3D s_incoming, Vector3D s_outgoing,
                                    Point3D w_inside_point, Point3D w_outside_point,
                                    AffineMatrix3D world_to_surface, AffineMatrix3D surface_to_world):

        exiting = s_incoming.z < 0

        # are incident and reflected on the same side?
        if (exiting and s_outgoing.z >= 0) or (not exiting and s_outgoing.z <= 0):
            # different sides, return empty spectrum
            return ray.new_spectrum()

        # generate and trace ray
        if exiting:
            reflected = ray.spawn_daughter(w_inside_point, s_outgoing.transform(surface_to_world))
        else:
            reflected = ray.spawn_daughter(w_outside_point, s_outgoing.transform(surface_to_world))

        spectrum = reflected.trace(world)

        # obtain samples of reflectivity
        reflectivity = self.reflectivity.sample_multiple(spectrum.min_wavelength,
                                                         spectrum.max_wavelength,
                                                         spectrum.num_samples)

        spectrum.mul_array(reflectivity)
        spectrum.mul_scalar(fabs(s_incoming.z) / PI)

        return spectrum

    # cpdef Spectrum bsdf(self, World world, Ray ray, Intersection intersection, Vector3D direction):
    #     # are incident and reflected on the same side?
    #     dn = direction.dot(intersection.normal)
    #     if (intersection.exiting and dn >= 0) or (not intersection.exiting and dn <= 0):
    #         # different sides, return empty spectrum
    #         return ray.new_spectrum()
    #
    #     # generate and trace ray
    #     if intersection.exiting:
    #         reflected = ray.spawn_daughter(intersection.inside_point.transform(intersection.primitive_to_world), direction.transform(intersection.primitive_to_world))
    #     else:
    #         reflected = ray.spawn_daughter(intersection.outside_point.transform(intersection.primitive_to_world), direction.transform(intersection.primitive_to_world))
    #
    #     spectrum = reflected.trace(world)
    #
    #     # obtain samples of reflectivity
    #     reflectivity = self.reflectivity.sample_multiple(spectrum.min_wavelength,
    #                                                      spectrum.max_wavelength,
    #                                                      spectrum.num_samples)
    #
    #     spectrum.mul_array(reflectivity)
    #     spectrum.mul_scalar(fabs(ray.direction.dot(intersection.normal.transform(intersection.primitive_to_world))))
    #     spectrum.mul_scalar(1 / PI)
    #
    #     return spectrum

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # no volume contribution
        return spectrum



