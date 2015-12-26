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

cdef class Material(CoreMaterial):

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D to_local, AffineMatrix3D to_world):

        raise NotImplementedError("Material virtual method evaluate_surface() has not been implemented.")

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D to_local, AffineMatrix3D to_world):

        raise NotImplementedError("Material virtual method evaluate_volume() has not been implemented.")


cdef class NullSurface(Material):

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D to_local, AffineMatrix3D to_world):

        cdef:
            Point3D origin
            Ray daughter_ray

        # are we entering or leaving surface?
        if exiting:

            # ray leaving surface, use exterior point for ray origin
            origin = outside_point.transform(to_world)

        else:

            # ray entering surface, use interior point for ray origin
            origin = inside_point.transform(to_world)

        daughter_ray = ray.spawn_daughter(origin, ray.direction)

        # do not count null surfaces in ray depth
        daughter_ray.depth -= 1

        # prevent extinction on a null surface
        return daughter_ray.trace(world, keep_alive=True)


cdef class NullVolume(Material):

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D to_local, AffineMatrix3D to_world):

        # do nothing!
        return spectrum