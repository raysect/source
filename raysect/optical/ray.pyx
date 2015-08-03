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
                 Point origin = Point(0,0,0),
                 Vector direction = Vector(0,0,1),
                 double min_wavelength = 375,
                 double max_wavelength = 785,
                 int num_samples = 40,
                 double max_distance = INFINITY,
                 int max_depth = 15):

        if num_samples < 1:
            raise ValueError("Number of samples can not be less than 1.")

        if min_wavelength <= 0.0 or max_wavelength <= 0.0:
            raise ValueError("Wavelength must be greater than to zero.")

        if min_wavelength >= max_wavelength:
            raise ValueError("Minimum wavelength must be less than the maximum wavelength.")

        super().__init__(origin, direction, max_distance)

        self._num_samples = num_samples
        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength

        self.max_depth = max_depth
        self.depth = 0

        # ray statistics
        self.ray_count = 0
        self._primary_ray = None

    def __getstate__(self):
        """Encodes state for pickling."""

        return (
            super().__getstate__(),
            self._num_samples,
            self._min_wavelength,
            self._max_wavelength,
            self.max_depth,
            self.depth,
            self.ray_count,
            self._primary_ray
        )

    def __setstate__(self, state):
        """Decodes state for pickling."""

        (super_state,
         self._num_samples,
         self._min_wavelength,
         self._max_wavelength,
         self.max_depth,
         self.depth,
         self.ray_count,
         self._primary_ray) = state

        super().__setstate__(super_state)

    property num_samples:

        def __get__(self):

            return self._num_samples

        def __set__(self, int num_samples):

            if num_samples < 1:
                raise ValueError("Number of samples can not be less than 1.")

            self._num_samples = num_samples

    cdef inline int get_num_samples(self):
        return self._num_samples

    property min_wavelength:

        def __get__(self):

            return self._min_wavelength

        def __set__(self, double min_wavelength):

            if min_wavelength <= 0.0:
                raise ValueError("Wavelength can not be less than or equal to zero.")

            if min_wavelength >= self._max_wavelength:
                raise ValueError("Minimum wavelength can not be greater or equal to the maximum wavelength.")

            self._min_wavelength = min_wavelength

    cdef inline double get_min_wavelength(self):

        return self._min_wavelength

    property max_wavelength:

        def __get__(self):

            return self._max_wavelength

        def __set__(self, double max_wavelength):

            if max_wavelength <= 0.0:
                raise ValueError("Wavelength can not be less than or equal to zero.")

            if self.min_wavelength >= max_wavelength:
                raise ValueError("Maximum wavelength can not be less than or equal to the minimum wavelength.")

            self._max_wavelength = max_wavelength

    cdef inline double get_max_wavelength(self):

        return self._max_wavelength

    cpdef Spectrum new_spectrum(self):
        """
        Returns a new Spectrum compatible with the ray spectral settings.
        """

        return new_spectrum(self._min_wavelength, self._max_wavelength, self._num_samples)

    cpdef Spectrum trace(self, World world):

        cdef:
            Spectrum spectrum
            Intersection intersection
            list primitives
            Primitive primitive
            Point start_point, end_point
            Material material

        # reset ray statistics
        if self._primary_ray is None:

            # this is the primary ray, count starts at 1 as the primary ray is the first ray
            self.ray_count = 1

        # create a new spectrum object compatible with the ray
        spectrum = self.new_spectrum()

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
                intersection.to_world
            )

            # identify any primitive volumes the ray is propagating through
            primitives = world.contains(self.origin)
            if len(primitives) > 0:

                # the start and end points for volume contribution calculations
                # defined such that start to end is in the direction of light
                # propagation - from source to observer
                start_point = intersection.hit_point.transform(intersection.to_world)
                end_point = self.origin

                # accumulate volume contributions to the spectrum
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
                        primitive.to_root()
                    )

        return spectrum

    cpdef Ray spawn_daughter(self, Point origin, Vector direction):

        cdef Ray ray

        ray = Ray.__new__(Ray)

        ray.origin = origin
        ray.direction = direction
        ray._num_samples = self._num_samples
        ray._min_wavelength = self._min_wavelength
        ray._max_wavelength = self._max_wavelength
        ray.max_distance = self.max_distance
        ray.max_depth = self.max_depth
        ray.depth = self.depth + 1

        # track ray statistics
        if self._primary_ray is None:

            # primary ray
            self.ray_count += 1
            ray._primary_ray = self

        else:

            # secondary ray
            self._primary_ray.ray_count += 1
            ray._primary_ray = self._primary_ray

        return ray

    cpdef Point get_point_at_distance(self, float distance):
        """ Get the point at a given distance alone the ray direction measured from the ray origin.

        Example use case, during a hit_intersection on a primitive you solve for the distances t0, t1 along the ray that
        define a hit intersection. Use this function to return the hit points in local coordinates.

        :param distance: A float value that defines the distance along the ray
        :return: A point at distance d along the ray direction measured from its origin.
        """
        cdef Point origin = self.origin
        cdef Vector direction = self.direction

        return Point(origin.x + distance * direction.x,
                     origin.y + distance * direction.y,
                     origin.z + distance * direction.z)


