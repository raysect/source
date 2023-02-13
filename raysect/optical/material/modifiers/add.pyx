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

from raysect.optical cimport Point3D, Normal3D, AffineMatrix3D, Primitive, World, Ray, Spectrum, Intersection
from raysect.optical.material cimport Material


cdef class Add(Material):
    """
    Adds the response of two materials.

    This modifier is used to sum together the behaviours of two different
    materials. The surface response is the simple sum of the two surface
    material responses. The volume response is more nuanced. The volume method
    of material 1 is applied first, followed by the volume method of material
    2. Depending on the choice of volume material, this may result in a simple
    summation or a more complex interaction.

    The Add modifier should be used with caution, it is possible to produce
    unphysical material combinations that violate energy conservation. It is
    the responsibility of the user to ensure the material combination is
    physically valid.

    By default both the volume and surface responses are combined. This may be
    configured with the surface_only and volume_only parameters. If summation
    is disabled the response from material 1 is returned.

    Add can be used to introduce a surface emission component to a non-emitting
    surface. For example, A hot metal surface can be approximated by adding a
    black body emitter to a metal material:

        material = Add(
            Iron(),
            UniformSurfaceEmitter(BlackBody(800)),
            surface_only=True
        )

    Combining volumes is more complex and must only be used with materials that
    are mathematically commutative, for example two volume emitters or two
    absorbing volumes.

    :param m1: The first material.
    :param m2: The second material.
    :param surface_only: Only blend the surface response (default=False).
    :param volume_only: Only blend the volume response (default=False).
    """

    cdef:
        Material m1, m2
        bint surface_only, volume_only

    def __init__(self, Material m1 not None, Material m2 not None, bint surface_only=False, bint volume_only=False):

        super().__init__()

        if surface_only and volume_only:
            raise ValueError("Surface only and volume only cannot be enabled at the same time.")

        self.m1 = m1
        self.m2 = m2

        # ensure the highest importance is passed through
        self.importance = max(m1.importance, m2.importance)

        self.surface_only = surface_only
        self.volume_only = volume_only

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef Spectrum s1, s2

        # sample material 1
        s1 = self.m1.evaluate_surface(world, ray, primitive, hit_point, exiting, inside_point, outside_point, normal, world_to_primitive, primitive_to_world, intersection)
        if self.volume_only:
            return s1

        # sample material 2
        s2 = self.m2.evaluate_surface(world, ray, primitive, hit_point, exiting, inside_point, outside_point, normal, world_to_primitive, primitive_to_world, intersection)

        # combine
        s1.add_spectrum(s2)
        return s1

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D to_local, AffineMatrix3D to_world):

        # sample material 1
        self.m1.evaluate_volume(spectrum, world, ray, primitive, start_point, end_point, to_local, to_world)
        if self.surface_only:
            return spectrum

        # sample material 2
        self.m2.evaluate_volume(spectrum, world, ray, primitive, start_point, end_point, to_local, to_world)
        return spectrum
