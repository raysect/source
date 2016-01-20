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
from raysect.core.math.random cimport probability
from raysect.core.math.utility cimport clamp
from raysect.core.classes cimport Intersection
from raysect.core.scenegraph.primitive cimport Primitive
from raysect.optical.material.material cimport Material
cimport cython

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999


cdef class Ray(CoreRay):

    def __init__(self,
                 Point3D origin = Point3D(0, 0, 0),
                 Vector3D direction = Vector3D(0, 0, 1),
                 double min_wavelength = 375,
                 double max_wavelength = 785,
                 int num_samples = 40,
                 double max_distance = INFINITY,
                 double extinction_prob = 0.1,
                 int min_depth = 3,
                 int max_depth = 100):

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

        self.extinction_prob = extinction_prob
        self.min_depth = min_depth
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
            self._extinction_prob,
            self._min_depth,
            self._max_depth,
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
         self._extinction_prob,
         self._min_depth,
         self._max_depth,
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

    property extinction_prob:

        def __get__(self):
            return self._extinction_prob

        def __set__(self, double extinction_prob):
            self._extinction_prob = clamp(extinction_prob, 0.0, 1.0)

    property min_depth:

        def __get__(self):
            return self._min_depth

        def __set__(self, int min_depth):
            if min_depth <= 1:
                raise ValueError("The minimum depth cannot be less than 1.")
            self._min_depth = min_depth

    property max_depth:

        def __get__(self):
            return self._max_depth

        def __set__(self, int max_depth):
            if max_depth < self._min_depth:
                raise ValueError("The maximum depth cannot be less than the minimum depth.")
            self._max_depth = max_depth

    cpdef Spectrum new_spectrum(self):
        """
        Returns a new Spectrum compatible with the ray spectral settings.
        """

        return new_spectrum(self._min_wavelength, self._max_wavelength, self._num_samples)

    @cython.cdivision(True)
    cpdef Spectrum trace(self, World world, bint keep_alive=False):
        """
        Traces a single ray path through the world.

        :param world: World object defining the scene.
        :param keep_alive: If true, disables Russian roulette termination of the ray.
        :return: A Spectrum object.
        """

        cdef:
            Spectrum spectrum
            Intersection intersection
            list primitives
            Primitive primitive
            Point3D start_point, end_point
            Material material
            double normalisation

        # reset ray statistics
        if self._primary_ray is None:

            # this is the primary ray, count starts at 1 as the primary ray is the first ray
            self.ray_count = 1

        # create a new spectrum object compatible with the ray
        spectrum = self.new_spectrum()

        # limit ray recursion depth with Russian roulette
        # set normalisation to ensure the sampling remains unbiased
        if keep_alive or self.depth < self._min_depth:
            normalisation = 1.0
        else:
            if self.depth >= self._max_depth or probability(self._extinction_prob):
                return spectrum
            else:
                normalisation = 1 / (1 - self._extinction_prob)

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

        # apply normalisation to ensure the sampling remains unbiased
        spectrum.mul_scalar(normalisation)
        return spectrum

    @cython.cdivision(True)
    cpdef Spectrum sample(self, World world, int count):
        """
        Samples the radiance directed along the ray direction.

        This methods calls trace repeatedly to obtain a statistical sample of
        the radiance directed along the ray direction from the world. The count
        parameter specifies the number of samples to obtain. The mean spectrum
        accumulated from these samples is returned.

        :param world: World object defining the scene.
        :param count: Number of samples to take.
        :return: A Spectrum object.
        """

        cdef:
            Spectrum spectrum, sample
            double normalisation

        if count < 1:
            raise ValueError("Samples must be >= 1.")

        spectrum = self.new_spectrum()
        normalisation = 1 / <double> count
        while count:
            sample = self.trace(world)
            spectrum.mad_scalar(normalisation, sample.samples)
            count -= 1

        return spectrum

    cpdef Ray spawn_daughter(self, Point3D origin, Vector3D direction):
        """
        Spawns a new daughter of the ray.

        A daughter ray has the same spectral configuration as the source ray,
        however the ray depth is increased by 1.

        :param origin: A Point3D defining the ray origin.
        :param direction: A vector defining the ray direction.
        :return: A Ray object.
        """

        cdef Ray ray

        ray = Ray.__new__(Ray)

        ray.origin = origin
        ray.direction = direction
        ray._num_samples = self._num_samples
        ray._min_wavelength = self._min_wavelength
        ray._max_wavelength = self._max_wavelength
        ray.max_distance = self.max_distance
        ray._extinction_prob = self._extinction_prob
        ray._min_depth = self._min_depth
        ray._max_depth = self._max_depth
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

    # TODO: PROFILE ME, ray--> cython new_ray for optical ray
    cpdef Ray copy(self, Point3D origin=None, Vector3D direction=None):

        if origin is None:
            origin = self.origin.copy()

        if direction is None:
            direction =self.direction.copy()

        return new_ray(
            origin, direction,
            self._min_wavelength, self._max_wavelength, self._num_samples,
            self.max_distance,
            self._extinction_prob, self._min_depth, self._max_depth
        )
