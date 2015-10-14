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

from raysect.optical.material.material cimport Material
from raysect.core.math.affinematrix cimport AffineMatrix
from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.scenegraph.world cimport World
from raysect.optical.ray cimport Ray
from raysect.core.math.point cimport Point
from raysect.core.math.vector cimport Vector
from raysect.core.math.affinematrix cimport new_affinematrix
from raysect.optical.spectrum cimport Spectrum
from raysect.core.math.normal cimport Normal
from raysect.core.math.random cimport vector_hemisphere_cosine


cdef class Roughen(Material):
    """
    Modifies the surface normal to approximate a rough surface.

    This is a modifier material, it takes another material (the base material)
    as an argument.

    The roughen modifier works by randomly deflecting the surface normal about
    its true position before passing the intersection parameters on to the base
    material.

    The deflection is calculated by interpolating between the existing normal
    and a vector sampled from a cosine weighted hemisphere. The strength of the
    interpolation, and hence the roughness of the surface, is controlled by the
    roughness argument. The roughness argument takes a value in the range
    [0, 1] where 1 is a fully rough, lambert-like surface and 0 is a smooth,
    untainted surface.

    :param material: The base material.
    :param roughness: A double value in the range [0, 1].
    """

    cdef:
        Material material
        double roughness

    def __init__(self, Material material not None, double roughness):

        if roughness < 0 or roughness > 1.0:
            raise ValueError("Roughness must be a floating point value in the range [0, 1] where 1 is full roughness.")

        self.roughness = roughness
        self.material = material

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point hit_point,
                                    bint exiting, Point inside_point, Point outside_point,
                                    Normal normal, AffineMatrix world_to_local, AffineMatrix local_to_world):

        cdef:
            Ray reflected
            Vector v_normal, v_tangent, v_bitangent, v_random
            AffineMatrix surface_to_local

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

        # Generate a new normal about the original normal by lerping between a random vector and the original normal.
        # The lerp strength is determined by the roughness. Calculation performed in surface space, so the original
        # normal is aligned with the z-axis.
        v_random = vector_hemisphere_cosine()
        normal.x = self.roughness * v_random.x
        normal.y = self.roughness * v_random.y
        normal.z = self.roughness * v_random.z + (1 - self.roughness)
        normal = normal.transform(surface_to_local).normalise()

        return self.material.evaluate_surface(world, ray, primitive, hit_point, exiting, inside_point, outside_point,
                                              normal, world_to_local, local_to_world)

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point start_point, Point end_point,
                                   AffineMatrix to_local, AffineMatrix to_world):

        return self.material.evaluate_volume(spectrum, world, ray, primitive, start_point, end_point, to_local, to_world)




