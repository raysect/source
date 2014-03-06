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

# TODO: clean up this code
# TODO: add docstrings

from raysect.core.acceleration.boundingbox cimport BoundingBox
from raysect.core.scenegraph.primitive cimport Primitive

cdef class Unaccelerated(Accelerator):

    def __init__(self):

        self.bounding_boxes = []

    cpdef build(self, list primitives):

        cdef Primitive p

        self.bounding_boxes = []
        for p in primitives:

            self.bounding_boxes.append(p.bounding_box())

    cpdef Intersection hit(self, Ray ray):

        cdef BoundingBox b
        cdef Intersection i, closest_intersection

        # find the closest primitive-ray intersection
        closest_intersection = None

        # intial search distance is maximum possible ray extent
        d = ray.max_distance

        # check each box for a hit
        for b in self.bounding_boxes:

            # obtain intersection
            if b.hit(ray):

                # box hit so test primitive
                i = b.primitive.hit(ray)
                if i is not None:

                    # check if new intersection is closer and outside the ray's minimum range
                    if (i.ray_distance > ray.min_distance) and (i.ray_distance < d):

                        d = i.ray_distance
                        closest_intersection = i

        return closest_intersection

    cpdef list inside(self, Point point):

        cdef BoundingBox b
        cdef list primitives

        primitives = []

        for b in self.bounding_boxes:

            if b.inside(point):

                if b.primitive.inside(point):

                    primitives.append(b.primitive)

        return primitives