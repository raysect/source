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

from raysect.optical.spectrum cimport new_spectrum

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

cdef class Ray(CoreRay):

    def __init__(self,
                 Point origin = Point([0,0,0]),
                 Vector direction = Vector([0,0,1]),
                 double min_wavelength = 375,
                 double max_wavelength = 785,
                 int samples = 40,
                 double max_distance = INFINITY,
                 int max_depth = 15):

        if samples < 1:

                raise("Number of samples can not be less than 1.")

        if min_wavelength <= 0.0 or max_wavelength <= 0.0:

            raise ValueError("Wavelength can not be less than or equal to zero.")

        if min_wavelength >= max_wavelength:

            raise ValueError("Minimum wavelength can not be greater or eaual to the maximum wavelength.")

        super().__init__(origin, direction, max_distance)

        self._samples = samples
        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength

        self.max_depth = max_depth
        self.depth = 0

        self._update()

    property samples:

        def __get__(self):

            return self._samples

        def __set__(self, int samples):

            if samples < 1:

                raise ValueError("Number of samples can not be less than 1.")

            self._samples = samples
            self._update()

    property min_wavelength:

        def __get__(self):

            return self._min_wavelength

        def __set__(self, double min_wavelength):

            if min_wavelength <= 0.0:

                raise ValueError("Wavelength can not be less than or equal to zero.")

            if min_wavelength >= self._max_wavelength:

                raise ValueError("Minimum wavelength can not be greater or equal to the maximum wavelength.")

            self._min_wavelength = min_wavelength
            self._update()

    property max_wavelength:

        def __get__(self):

            return self._max_wavelength

        def __set__(self, double max_wavelength):

            if max_wavelength <= 0.0:

                raise ValueError("Wavelength can not be less than or equal to zero.")

            if self.min_wavelength >= max_wavelength:

                raise ValueError("Maximum wavelength can not be less than or equal to the minimum wavelength.")

            self._max_wavelength = max_wavelength
            self._update()

    cpdef Spectrum trace(self, World world):

        cdef Spectrum spectrum
        cdef Intersection intersection
        cdef list primitives
        cdef Primitive primitive
        cdef Point start_point, end_point
        cdef Material material

        spectrum = new_spectrum(self.min_wavelength, self.max_wavelength, self.samples)

        # limit ray recursion depth
        if self.depth >= self.max_depth:

            return spectrum

        # does the ray intersect with any of the primitives in the world?
        intersection = world.hit(self)
        if intersection is not None:

            # request surface contribution to spectrum from primitive material
            material = intersection.primitive.material
            spectrum = material.evaluate_surface(
                           world, self,
                           intersection.primitive,
                           intersection.hit_point,
                           intersection.exiting,
                           intersection.inside_point,
                           intersection.outside_point,
                           intersection.normal,
                           intersection.to_local,
                           intersection.to_world)

            # identify any primitive volumes the ray is propagating through
            primitives = world.inside(self.origin)
            if len(primitives) > 0:

                # the start and end points for volume contribution calculations
                # defined such that start to end is in the direction of light
                # propagation - from source to observer
                start_point = intersection.hit_point.transform(intersection.to_world)
                end_point = self.origin

                # accumulate volume contributions to the spectum
                for primitive in primitives:

                    material = primitive.material
                    spectrum = material.evaluate_volume(
                                   spectrum,
                                   world,
                                   self,
                                   primitive,
                                   start_point,
                                   end_point,
                                   primitive.to_local(),
                                   primitive.to_root())

        return spectrum

    cpdef Ray spawn_daughter(self, Point origin, Vector direction):

        cdef Ray ray

        ray = Ray.__new__(Ray)

        ray.origin = origin
        ray.direction = direction
        ray._samples = self._samples
        ray._min_wavelength = self._min_wavelength
        ray._max_wavelength = self._max_wavelength
        ray.refraction_wavelength = self.refraction_wavelength
        ray.max_distance = self.max_distance
        ray.max_depth = self.max_depth
        ray.depth = self.depth + 1

        return ray

    cdef inline void _update(self):

        # set refraction wavelength to central wavelength of ray's spectral range
        self.refraction_wavelength = 0.5 * (self._min_wavelength + self._max_wavelength)

