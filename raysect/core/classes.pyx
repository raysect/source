# cython: language_level=3

#Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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

    def __init__(self,
                 Point origin not None = Point(0, 0, 0),
                 Vector direction not None = Vector(0, 0, 1),
                 double max_distance = INFINITY):

        self.origin = origin
        self.direction = direction
        self.max_distance = max_distance

    def __repr__(self):

        return "Ray({}, {}, {})".format(self.origin, self.direction, self.max_distance)

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.origin, self.direction, self.max_distance

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.origin, self.direction, self.max_distance = state



cdef class Intersection:

    def __init__(self, Ray ray, double ray_distance, Primitive primitive,
                 Point hit_point, Point inside_point, Point outside_point,
                 Normal normal, bint exiting, AffineMatrix to_local, AffineMatrix to_world):

        self.ray = ray
        self.ray_distance = ray_distance
        self.exiting = exiting
        self.primitive = primitive
        self.hit_point = hit_point
        self.inside_point = inside_point
        self.outside_point = outside_point
        self.normal = normal
        self.to_local = to_local
        self.to_world = to_world

    def __repr__(self):

        return "Intersection({}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
            self.ray, self.ray_distance, self.primitive,
            self.hit_point, self.inside_point, self.outside_point,
            self.normal, self.exiting, self.to_local, self.to_world)


cdef class Material


