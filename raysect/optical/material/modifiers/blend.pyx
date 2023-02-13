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

from raysect.core.math.random cimport probability
from raysect.optical cimport Point3D, Normal3D, AffineMatrix3D, Primitive, World, Ray, Spectrum, Intersection
from raysect.optical.material cimport Material


cdef class Blend(Material):
    """
    Blend combines the behaviours of two materials.

    This modifier is used to blend together the behaviours of two different
    materials. Which material handles the interaction for an incoming ray is
    determined by a random choice, weighted by the ratio argument. Low values
    of ratio bias the selection towards material 1, high values to material 2.

    It is the responsibility of the user to ensure the material combination is
    physically valid.

    By default both the volume and surface responses are blended. This may be
    configured with the surface_only and volume_only parameters. If blending
    is disabled the response from material 1 is returned.

    Blend can be used to approximate finely sputtered surfaces consisting of a
    mix of materials. For example it can be used to crudely approximate a gold
    coated glass surface:

        material = Blend(schott('N-BK7'), Gold(), 0.1, surface_only=True)

    :param m1: The first material.
    :param m2: The second material.
    :param ratio: A double value in the range (0, 1).
    :param surface_only: Only blend the surface response (default=False).
    :param volume_only: Only blend the volume response (default=False).
    """

    cdef:
        Material m1, m2
        double ratio
        bint surface_only, volume_only

    def __init__(self, Material m1 not None, Material m2 not None, double ratio, bint surface_only=False, bint volume_only=False):

        super().__init__()

        if ratio <= 0 or ratio >= 1.0:
            raise ValueError("Ratio must be a floating point value in the range (0, 1).")

        self.m1 = m1
        self.m2 = m2
        self.ratio = ratio

        # ensure the highest importance is passed through
        self.importance = max(m1.importance, m2.importance)

        self.surface_only = surface_only
        self.volume_only = volume_only

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        if not self.volume_only and probability(self.ratio):
            return self.m2.evaluate_surface(world, ray, primitive, hit_point, exiting, inside_point, outside_point,
                                                  normal, world_to_primitive, primitive_to_world, intersection)
        else:
            return self.m1.evaluate_surface(world, ray, primitive, hit_point, exiting, inside_point, outside_point,
                                                  normal, world_to_primitive, primitive_to_world, intersection)

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D to_local, AffineMatrix3D to_world):

        if not self.surface_only and probability(self.ratio):
            return self.m2.evaluate_volume(spectrum, world, ray, primitive, start_point, end_point, to_local, to_world)
        else:
            return self.m1.evaluate_volume(spectrum, world, ray, primitive, start_point, end_point, to_local, to_world)

