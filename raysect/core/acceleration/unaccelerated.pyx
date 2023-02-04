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

# TODO: add docstrings
cimport cython
from raysect.core.ray cimport Ray
from raysect.core.scenegraph cimport Primitive
from raysect.core.math cimport Point3D
from raysect.core.intersection cimport Intersection
from raysect.core.acceleration.boundprimitive cimport BoundPrimitive

cdef class Unaccelerated(Accelerator):

    def __init__(self):

        self.primitives = []
        self.world_box = BoundingBox3D()

    cpdef build(self, list primitives):

        cdef:
            Primitive primitive
            BoundPrimitive accel_primitive

        self.primitives = []
        self.world_box = BoundingBox3D()

        for primitive in primitives:

            accel_primitive = BoundPrimitive(primitive)
            self.primitives.append(accel_primitive)
            self.world_box.union(accel_primitive.box)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Intersection hit(self, Ray ray):

        cdef:
            double distance
            Intersection intersection, closest_intersection
            BoundPrimitive primitive

        # does the ray intersect the space containing the primitives
        if not self.world_box.hit(ray):

            return None

        # find the closest primitive-ray intersection
        closest_intersection = None

        # intial search distance is maximum possible ray extent
        distance = ray.max_distance

        for primitive in self.primitives:

            intersection = primitive.hit(ray)

            if intersection is not None:

                if intersection.ray_distance < distance:

                    distance = intersection.ray_distance
                    closest_intersection = intersection

        return closest_intersection

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list contains(self, Point3D point):

        cdef:
            list enclosing_primitives
            BoundPrimitive primitive

        if not self.world_box.contains(point):

            return []

        enclosing_primitives = []

        for primitive in self.primitives:

            if primitive.contains(point):

                enclosing_primitives.append(primitive.primitive)

        return enclosing_primitives