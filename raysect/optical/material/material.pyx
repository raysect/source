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

from raysect.core.math.random cimport probability

# TODO: move surface_to_primitive calculation to material from intersection, convert eval_surface API back to list of intersection parameters

cdef class Material(CoreMaterial):

    def __init__(self):
        super().__init__()
        self._importance = 0.0

    property importance:

        def __get__(self):
            return self._importance

        def __set__(self, value):
            if value < 0:
                raise ValueError("Material sampling importance cannot be less than zero.")
            self._importance = value
            self.notify_material_change()

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Intersection intersection):
        raise NotImplementedError("Material virtual method evaluate_surface() has not been implemented.")

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):
        raise NotImplementedError("Material virtual method evaluate_volume() has not been implemented.")


cdef class ContinuousPDF(Material):

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Intersection intersection):

        cdef:
            double pdf, pdf_importance, pdf_bsdf

            # primitive space
            Point3D p_hit_point
            Vector3D p_direction

            # world space
            Point3D w_hit_point
            Vector3D w_direction

        if ray.importance_sampling and world.has_important_primitives():

            w_hit_point = intersection.hit_point.transform(intersection.primitive_to_world)

            # multiple importance sampling
            important_path_weight = 0.5  # TODO: make an attribute

            if probability(important_path_weight):

                # sample important path pdf
                w_direction = world.important_direction_sample(w_hit_point)
                p_direction = w_direction.transform(intersection.world_to_primitive)

            else:

                # sample bsdf pdf
                p_direction = self.sample(world, ray, intersection)
                w_direction = p_direction.transform(intersection.primitive_to_world)

            # compute combined pdf
            pdf_important = world.important_direction_pdf(w_hit_point, w_direction)
            pdf_bsdf = self.pdf(world, ray, intersection, p_direction)
            pdf = important_path_weight * pdf_important + (1 - important_path_weight) * pdf_bsdf

            # # impossible paths must have zero valued spectrum
            # if pdf == 0.0:
            #     return ray.new_spectrum()

            # evaluate bsdf and normalise
            spectrum = self.bsdf(world, ray, intersection, p_direction)
            spectrum.mul_scalar(1 / pdf)
            return spectrum

        else:

            # bsdf sampling
            p_direction = self.sample(world, ray, intersection)
            spectrum = self.bsdf(world, ray, intersection, p_direction)
            pdf = self.pdf(world, ray, intersection, p_direction)
            spectrum.mul_scalar(1 / pdf)
            return spectrum

    cpdef double pdf(self, World world, Ray ray, Intersection intersection, Vector3D direction):
        raise NotImplementedError("ContinuousMaterial virtual method pdf() has not been implemented.")

    cpdef Vector3D sample(self, World world, Ray ray, Intersection intersection):
        raise NotImplementedError("ContinuousMaterial virtual method sample() has not been implemented.")

    cpdef Spectrum bsdf(self, World world, Ray ray, Intersection intersection, Vector3D direction):
        raise NotImplementedError("ContinuousMaterial virtual method bsdf() has not been implemented.")


cdef class NullSurface(Material):

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Intersection intersection):

        cdef:
            Point3D origin
            Ray daughter_ray

        # are we entering or leaving surface?
        if intersection.exiting:
            origin = intersection.outside_point.transform(intersection.primitive_to_world)
        else:
            origin = intersection.inside_point.transform(intersection.primitive_to_world)

        daughter_ray = ray.spawn_daughter(origin, ray.direction)

        # do not count null surfaces in ray depth
        daughter_ray.depth -= 1

        # prevent extinction on a null surface
        return daughter_ray.trace(world, keep_alive=True)
