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

from raysect.core.math.point cimport new_point3d

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

cdef class Ray:
    """
    Describes a line in space with an origin and direction.

    :param Point3D origin: Point defining origin (default is Point3D(0, 0, 0)).
    :param Vector3D direction: Vector defining direction (default is Vector3D(0, 0, 1)).
    :param double max_distance: The terminating distance of the ray.
    """

    def __init__(self, Point3D origin=None, Vector3D direction=None, double max_distance=INFINITY):

        if origin is None:
            origin = Point3D(0, 0, 0)

        if direction is None:
            direction = Vector3D(0, 0, 1)

        self.origin = origin
        """Point3D defining origin (default is Point3D(0, 0, 0))."""
        self.direction = direction
        """Vector3D defining direction (default is Vector3D(0, 0, 1))."""
        self.max_distance = max_distance
        """The terminating distance of the ray."""

    def __repr__(self):

        return "Ray({}, {}, {})".format(self.origin, self.direction, self.max_distance)

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.origin, self.direction, self.max_distance

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.origin, self.direction, self.max_distance = state

    cpdef Point3D point_on(self, double t):
        """
        Returns the point on the ray at the specified parametric distance from the ray origin.

        Positive values correspond to points forward of the ray origin, along the ray direction.

        :param t: The distance along the ray.
        :return: A point at distance t along the ray direction measured from its origin.
        """
        cdef:
            Point3D origin = self.origin
            Vector3D direction = self.direction

        return new_point3d(origin.x + t * direction.x,
                           origin.y + t * direction.y,
                           origin.z + t * direction.z)

    cpdef Ray copy(self, Point3D origin=None, Vector3D direction=None):

        if origin is None:
            origin = self.origin.copy()

        if direction is None:
            direction =self.direction.copy()

        return new_ray(origin, direction, self.max_distance)


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
    :param AffineMatrix3D to_local: A world to primitive local transform matrix.
    :param AffineMatrix3D to_world: A primitive local to world transform matrix.
    """

    def __init__(self, Ray ray, double ray_distance, Primitive primitive,
                 Point3D hit_point, Point3D inside_point, Point3D outside_point,
                 Normal3D normal, bint exiting, AffineMatrix3D to_local, AffineMatrix3D to_world):

        self.ray = ray
        """The incident ray object (world space)."""
        self.ray_distance = ray_distance
        """The distance of the intersection along the ray path."""
        self.exiting = exiting
        """True if the ray is exiting the surface, False otherwise."""
        self.primitive = primitive
        """The intersected primitive object."""
        self.hit_point = hit_point
        """The point of intersection between the ray and the primitive (primitive local space)."""
        self.inside_point = inside_point
        """The interior ray launch point (primitive local space)."""
        self.outside_point = outside_point
        """The exterior ray launch point (primitive local space)."""
        self.normal = normal
        """The surface normal (primitive local space)"""
        self.to_local = to_local
        """A world to primitive local transform matrix."""
        self.to_world = to_world
        """A primitive local to world transform matrix."""

    def __repr__(self):

        return "Intersection({}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
            self.ray, self.ray_distance, self.primitive,
            self.hit_point, self.inside_point, self.outside_point,
            self.normal, self.exiting, self.to_local, self.to_world)


cdef class Material


