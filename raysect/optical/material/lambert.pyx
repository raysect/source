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

from raysect.core.math.affinematrix cimport AffineMatrix
from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.scenegraph.world cimport World
from raysect.optical.ray cimport Ray
from raysect.core.math.point cimport Point
from raysect.core.math.vector cimport Vector
from raysect.core.math.affinematrix cimport new_affinematrix
from raysect.optical.spectrum cimport Spectrum
from raysect.core.math.normal cimport Normal
from raysect.optical.spectralfunction cimport SpectralFunction, ConstantSF
from raysect.core.math.random cimport vector_hemisphere_cosine
from libc.math cimport M_PI as PI

cdef class Lambert(NullVolume):

    def __init__(self, SpectralFunction reflectivity=None):

        if reflectivity is None:
            reflectivity = ConstantSF(0.5)

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point hit_point,
                                    bint exiting, Point inside_point, Point outside_point,
                                    Normal normal, AffineMatrix world_to_local, AffineMatrix local_to_world):

        cdef:
            Ray reflected
            Vector v_normal, v_tangent, v_bitangent, direction
            AffineMatrix surface_to_local
            Spectrum spectrum

        # obtain samples of reflectivity
        # todo: reflectivity

        # generate an orthogonal basis about surface normal
        v_normal = normal.as_vector()
        v_tangent = normal.orthogonal()
        v_bitangent = v_normal.cross(v_tangent)

        # generate inverse surface transform matrix
        # TODO: MOVE THIS TO A UTILITY FUNCTION AND TEST - math.cython.transforms <- high performance but no safety
        surface_to_local = new_affinematrix(v_tangent.x, v_bitangent.x, v_normal.x, 0.0,
                                            v_tangent.y, v_bitangent.y, v_normal.y, 0.0,
                                            v_tangent.z, v_bitangent.z, v_normal.z, 0.0,
                                            0.0, 0.0, 0.0, 1.0)

        # TODO: MOVE THIS TO A UTILITY FUNCTION AND TEST - math.cython.transforms <- high performance but no safety
        # local_to_surface = new_affinematrix(v_tangent.x, v_tangent.y, v_tangent.z, 0.0,
        #                                     v_bitangent.x, v_bitangent.y, v_bitangent.z, 0.0,
        #                                     v_normal.x, v_normal.y, v_normal.z, 0.0,
        #                                     0.0, 0.0, 0.0, 1.0)

        # obtain new world space ray vector from cosine-weighted hemisphere
        direction = vector_hemisphere_cosine()
        direction = direction.transform(surface_to_local)
        # normalisation = direction.dot(normal)

        # avoid a divide by zero, kill rays that are fully orthogonal to direction
        # if normalisation == 0.0:
        #     return ray.new_spectrum()

        # generate and trace ray
        if exiting:
            reflected = ray.spawn_daughter(inside_point.transform(local_to_world), -direction.transform(local_to_world))
        else:
            reflected = ray.spawn_daughter(outside_point.transform(local_to_world), direction.transform(local_to_world))

        spectrum = reflected.trace(world)

        # apply reflectivity and normalisation
        # todo: reflectivity, fudged here as 0.5
        spectrum.mul_scalar(0.5)

        # TODO: figure out normalisation, this isn't right
        #spectrum.mul_scalar(2*PI / normalisation)

        return spectrum







