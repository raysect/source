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

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

cdef class Ray:
    """
    Describes a line in space with an origin and direction.

    :param Point origin: Point defining origin (default is Point(0, 0, 0)).
    :param Vector direction: Vector defining origin (default is Point(0, 0, 0)).
    :param double max_distance: The terminating distance of the ray.

    """

    def __init__(self, Point origin=None, Vector direction=None, double max_distance=INFINITY):

        if origin is None:
            origin = Point(0, 0, 0)

        if direction is None:
            direction = Vector(0, 0, 1)

        self.origin = origin
        """Point defining origin (default is Point(0, 0, 0))."""
        self.direction = direction
        """Vector defining origin (default is Point(0, 0, 0))."""
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
    :param Point hit_point: The point of intersection between the ray and the primitive (primitive local space).
    :param Point inside_point: The interior ray launch point (primitive local space).
    :param Point outside_point: The exterior ray launch point (primitive local space).
    :param Normal normal: The surface normal (primitive local space)
    :param bint exiting: True if the ray is exiting the surface, False otherwise.
    :param AffineMatrix to_local: A world to primitive local transform matrix.
    :param AffineMatrix to_world: A primitive local to world transform matrix.
    """

    def __init__(self, Ray ray, double ray_distance, Primitive primitive,
                 Point hit_point, Point inside_point, Point outside_point,
                 Normal normal, bint exiting, AffineMatrix to_local, AffineMatrix to_world):

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


# TODO - Better name like "Properties" since models built on raysect.core may need to store information on primitives.
# TODO - This should probably be moved to the Primitive class.
cdef class Material


