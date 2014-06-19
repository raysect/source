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

cdef class RayRGB(Ray):

    def __init__(self, Point origin = Point([0,0,0]), Vector direction = Vector([0,0,1]),
                 double max_distance = float('inf'), double max_depth = 15):

        super().__init__(origin, direction, max_distance)

        self.max_depth = max_depth
        self.depth = 0

    cpdef RGB trace(self, World world):

        cdef RGB spectrum
        cdef Intersection intersection
        cdef MaterialRGB material
        cdef SurfaceRGB sresponce
        cdef VolumeRGB vresponce
        cdef list primitives
        cdef Primitive primitive
        cdef Point entry_point, exit_point

        spectrum = RGB(0, 0, 0)

        if self.depth >= self.max_depth:

            return spectrum

        intersection = world.hit(self)

        if intersection is not None:

            # surface contribution
            material = intersection.primitive.material
            sresponce = material.evaluate_surface(world, self,
                                                 intersection.primitive,
                                                 intersection.hit_point,
                                                 intersection.exiting,
                                                 intersection.inside_point,
                                                 intersection.outside_point,
                                                 intersection.normal,
                                                 intersection.to_local,
                                                 intersection.to_world)

            spectrum.r = sresponce.intensity.r
            spectrum.g = sresponce.intensity.g
            spectrum.b = sresponce.intensity.b

            # volume contribution - TODO: deal max_distance if no intersection occurs.
            primitives = world.contains(self.origin)

            entry_point = self.origin
            exit_point = intersection.hit_point.transform(intersection.to_world)

            for primitive in primitives:

                vresponce = primitive.material.evaluate_volume(world,
                                                              self,
                                                              entry_point,
                                                              exit_point,
                                                              primitive.to_local(),
                                                              primitive.to_root())

                spectrum.r = (1 - vresponce.attenuation.r) * spectrum.r + vresponce.intensity.r
                spectrum.g = (1 - vresponce.attenuation.g) * spectrum.g + vresponce.intensity.g
                spectrum.b = (1 - vresponce.attenuation.b) * spectrum.b + vresponce.intensity.b

        return spectrum

    cpdef RayRGB spawn_daughter(self, Point origin, Vector direction):

        cdef RayRGB ray

        ray = RayRGB(origin, direction)

        ray.max_depth = self.max_depth
        ray.depth = self.depth + 1

        return ray
