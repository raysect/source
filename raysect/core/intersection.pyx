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

from raysect.core.math.affinematrix cimport AffineMatrix3D, new_affinematrix3d
from raysect.core.math.vector cimport Vector3D


cdef class Intersection:
    """
    Describes the result of a ray-primitive intersection.

    The inside and outside points are launch points for rays emitted from the hit point on the surface. Rays cannot be
    launched from the hit point directly as they risk re-intersecting the same surface due to numerical accuracy. The
    inside and outside points are slightly displaced from the primitive surface at a sufficient distance to prevent
    re-intersection due to numerical accuracy issues. The inside_point is shifted backwards into the surface relative to
    the surface normal. The outside_point is equivalently shifted away from the surface in the direction of the surface
    normal.

    :param Ray ray: The incident ray object (world space).
    :param double ray_distance: The distance of the intersection along the ray path.
    :param Primitive primitive: The intersected primitive object.
    :param Point3D hit_point: The point of intersection between the ray and the primitive (primitive local space).
    :param Point3D inside_point: The interior ray launch point (primitive local space).
    :param Point3D outside_point: The exterior ray launch point (primitive local space).
    :param Normal3D normal: The surface normal (primitive local space)
    :param bint exiting: True if the ray is exiting the surface, False otherwise.
    :param AffineMatrix3D world_to_primitive: A world to primitive local transform matrix.
    :param AffineMatrix3D primitive_to_world: A primitive local to world transform matrix.
    """

    def __init__(self, Ray ray, double ray_distance, Primitive primitive,
                 Point3D hit_point, Point3D inside_point, Point3D outside_point,
                 Normal3D normal, bint exiting, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        self._construct(ray, ray_distance, primitive, hit_point, inside_point, outside_point, normal, exiting, world_to_primitive, primitive_to_world)

    cdef inline void _construct(self, Ray ray, double ray_distance, Primitive primitive,
                                Point3D hit_point, Point3D inside_point, Point3D outside_point,
                                Normal3D normal, bint exiting, AffineMatrix3D world_to_primitive,
                                AffineMatrix3D primitive_to_world):

        self.ray = ray
        self.ray_distance = ray_distance
        self.exiting = exiting
        self.primitive = primitive
        self.hit_point = hit_point
        self.inside_point = inside_point
        self.outside_point = outside_point
        self.normal = normal
        self.world_to_primitive = world_to_primitive
        self.primitive_to_world = primitive_to_world
        self._primitive_to_surface = None
        self._surface_to_primitive = None

    cpdef AffineMatrix3D primitive_to_surface(self):
        """
        Returns a transform from surface space to primitive local space.

        In surface space the +ve z-axis is aligned with surface normal.

        :return: An AffineMatrix.
        """

        if self._primitive_to_surface is None:
            self._generate_surface_transforms()
        return self._primitive_to_surface

    cpdef AffineMatrix3D surface_to_primitive(self):
        """
        Returns a transform from primitive local space to surface space.

        In surface space the +ve z-axis is aligned with surface normal.

        :return: An AffineMatrix.
        """


        if self._surface_to_primitive is None:
            self._generate_surface_transforms()
        return self._surface_to_primitive

    cdef inline void _generate_surface_transforms(self):
        """
        Calculates and populates the surface space transform attributes.
        """

        cdef Vector3D normal, tangent, bitangent

        # TODO: when UV information added, align the x-axis with the u-coordinate and y-axis with the v-coordinate
        normal = self.normal.as_vector()
        tangent = self.normal.orthogonal()
        bitangent = normal.cross(tangent)

        self._primitive_to_surface = new_affinematrix3d(
            tangent.x, tangent.y, tangent.z, 0.0,
            bitangent.x, bitangent.y, bitangent.z, 0.0,
            normal.x, normal.y, normal.z, 0.0,
            0.0, 0.0, 0.0, 1.0
        )

        self._surface_to_primitive = new_affinematrix3d(
            tangent.x, bitangent.x, normal.x, 0.0,
            tangent.y, bitangent.y, normal.y, 0.0,
            tangent.z, bitangent.z, normal.z, 0.0,
            0.0, 0.0, 0.0, 1.0
        )

    def __repr__(self):

        return "Intersection({}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
            self.ray, self.ray_distance, self.primitive,
            self.hit_point, self.inside_point, self.outside_point,
            self.normal, self.exiting,
            self.world_to_primitive, self.primitive_to_world)
