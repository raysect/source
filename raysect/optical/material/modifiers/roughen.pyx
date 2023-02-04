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

from raysect.core.math.sampler cimport HemisphereCosineSampler
from raysect.optical cimport Point3D, Vector3D, Normal3D, new_normal3d, AffineMatrix3D, new_affinematrix3d, Primitive, World, Ray, Spectrum, Intersection
from raysect.optical.material cimport Material

# sets the maximum number of attempts to find a valid perturbed normal
# it is highly unlikely (REALLY!) this number will ever be reached, it is just there for my paranoia
# in the worst case 50% of the random hemisphere will always generate a valid solution... so P(fail) < 0.5^50!
DEF SAMPLE_ATTEMPTS = 50


cdef HemisphereCosineSampler hemisphere_sampler = HemisphereCosineSampler()


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

        super().__init__()

        if roughness < 0 or roughness > 1.0:
            raise ValueError("Roughness must be a floating point value in the range [0, 1] where 1 is full roughness.")

        self.roughness = roughness
        self.material = material

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef:
            Ray reflected
            Vector3D s_incident, s_random
            Normal3D s_normal
            AffineMatrix3D surface_to_primitive
            int attempt

        # generate surface transforms
        primitive_to_surface, surface_to_primitive = self._generate_surface_transforms(normal)

        # convert ray direction to surface space
        s_incident = ray.direction.transform(world_to_primitive).transform(primitive_to_surface)

        # attempt to find a valid (intersectable by ray) surface perturbation
        s_normal = new_normal3d(0, 0, 1)
        for attempt in range(SAMPLE_ATTEMPTS):

            # Generate a new normal about the original normal by lerping between a random vector and the original normal.
            # The lerp strength is determined by the roughness. Calculation performed in surface space, so the original
            # normal is aligned with the z-axis.
            s_random = hemisphere_sampler.sample()
            s_normal.x = self.roughness * s_random.x
            s_normal.y = self.roughness * s_random.y
            s_normal.z = self.roughness * s_random.z + (1 - self.roughness)

            # Only accept the new normal if it does not change the side of the surface the incident ray is on.
            # An incident ray could not hit a surface facet that is facing away from it.
            # If (incident.normal) * (incident.perturbed_normal) < 0 the ray has swapped sides.
            # Note: normal in surface space is Normal3D(0, 0, 1), therefore incident.normal is just incident.z.
            if (s_incident.z * s_incident.dot(s_normal)) > 0:

                # we have found a valid perturbation, re-assign normal
                normal = s_normal.transform(surface_to_primitive).normalise()
                break

        return self.material.evaluate_surface(world, ray, primitive, hit_point, exiting, inside_point, outside_point,
                                              normal, world_to_primitive, primitive_to_world, intersection)

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D to_local, AffineMatrix3D to_world):

        return self.material.evaluate_volume(spectrum, world, ray, primitive, start_point, end_point, to_local, to_world)

    cdef tuple _generate_surface_transforms(self, Normal3D normal):
        """
        Calculates and populates the surface space transform attributes.
        """

        cdef:
            Vector3D tangent, bitangent
            AffineMatrix3D primitive_to_surface, surface_to_primitive

        tangent = normal.orthogonal()
        bitangent = normal.cross(tangent)

        primitive_to_surface = new_affinematrix3d(
            tangent.x, tangent.y, tangent.z, 0.0,
            bitangent.x, bitangent.y, bitangent.z, 0.0,
            normal.x, normal.y, normal.z, 0.0,
            0.0, 0.0, 0.0, 1.0
        )

        surface_to_primitive = new_affinematrix3d(
            tangent.x, bitangent.x, normal.x, 0.0,
            tangent.y, bitangent.y, normal.y, 0.0,
            tangent.z, bitangent.z, normal.z, 0.0,
            0.0, 0.0, 0.0, 1.0
        )

        return primitive_to_surface, surface_to_primitive



