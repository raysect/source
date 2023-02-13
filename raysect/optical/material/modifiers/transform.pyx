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


cdef class VolumeTransform(Material):
    """
    Translate/rotate the volume material relative to the primitive.

    Applies an affine transform to the start and end points of the volume
    response calculation. This modifier is intended for use with volume
    texture materials, allowing them to be translated/rotated.

    As a modifier material, it takes another material (the base material) as an
    argument. Using a supplied an affine transform, this material will modify
    the start and end coordinate of the volume integration.

    :param material: The base material.
    :param transform: An affine transform.
    """

    cdef:
        Material _material
        AffineMatrix3D _transform, _transform_inv

    def __init__(self, Material material not None, AffineMatrix3D transform=None):

        super().__init__()

        self.material = material
        self.transform = transform or AffineMatrix3D()

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, Material m not None):
        self._material = m

    @property
    def transform(self):
        return

    @transform.setter
    def transform(self, AffineMatrix3D m not None):
        self._transform = m
        self._transform_inv = m.inverse()

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        return self.material.evaluate_surface(world, ray, primitive, hit_point, exiting, inside_point, outside_point,
                                              normal, world_to_primitive, primitive_to_world, intersection)

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef AffineMatrix3D m = primitive_to_world.mul(self._transform_inv.mul(world_to_primitive))
        start_point = start_point.transform(m)
        end_point = end_point.transform(m)
        return self.material.evaluate_volume(spectrum, world, ray, primitive,
                                             start_point, end_point, world_to_primitive, primitive_to_world)
