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

from libc.math cimport M_PI as PI, asin, cos

from raysect.core cimport Intersection
from raysect.core.math.random cimport probability
from raysect.core.math.cython cimport clamp
from raysect.optical.material.material cimport Material
from raysect.optical.spectrum cimport new_spectrum
from raysect.optical.scenegraph cimport Primitive
cimport cython

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999


cdef class Ray(CoreRay):
    """
    Optical Ray class for optical applications, inherits from core Ray class.

    Provides the trace(world) method.

    :param Point3D origin: Point defining ray’s origin (default=Point3D(0, 0, 0))
    :param Vector3D direction: Vector defining ray’s direction (default=Vector3D(0, 0, 1))
    :param float min_wavelength: Lower wavelength bound for observed spectrum
    :param float max_wavelength: Upper wavelength bound for observed spectrum
    :param int bins: Number of samples to use over the spectral range
    :param float max_distance: The terminating distance of the ray
    :param float extinction_prob: Probability of path extinction at every
      material surface interaction (default=0.1)
    :param int extinction_min_depth: Minimum number of paths before triggering
      extinction probability (default=3)
    :param int max_depth: Maximum number of material interactions before
      terminating ray trajectory.
    :param bool importance_sampling: Toggles use of importance sampling for
      important primitives. See help documentation on importance sampling,
      (default=True).
    :param float important_path_weight: Weight to use for important paths when
      using importance sampling.

    .. code-block:: pycon

        >>> from raysect.core import Point3D, Vector3D
        >>> from raysect.optical import World, Ray
        >>>
        >>> world = World()
        >>>
        >>> ray = Ray(origin=Point3D(0, 0, -5),
        >>>           direction=Vector3D(0, 0, 1),
        >>>           min_wavelength=375,
        >>>           max_wavelength=785,
        >>>           bins=100)
        >>>
        >>> spectrum = ray.trace(world)
        >>> spectrum
        <raysect.optical.spectrum.Spectrum at 0x7f5b08b6e048>
    """

    def __init__(self,
                 Point3D origin = Point3D(0, 0, 0),
                 Vector3D direction = Vector3D(0, 0, 1),
                 double min_wavelength = 375,
                 double max_wavelength = 785,
                 int bins = 40,
                 double max_distance = INFINITY,
                 double extinction_prob = 0.1,
                 int extinction_min_depth = 3,
                 int max_depth = 100,
                 bint importance_sampling=True,
                 double important_path_weight=0.25):

        if bins < 1:
            raise ValueError("Number of bins cannot be less than 1.")

        if min_wavelength <= 0.0 or max_wavelength <= 0.0:
            raise ValueError("Wavelength must be greater than to zero.")

        if min_wavelength >= max_wavelength:
            raise ValueError("Minimum wavelength must be less than the maximum wavelength.")

        if important_path_weight < 0 or important_path_weight > 1.0:
            raise ValueError("Important path weight must be in the range [0, 1].")

        super().__init__(origin, direction, max_distance)

        self._bins = bins
        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength

        self.extinction_prob = extinction_prob
        self.extinction_min_depth = extinction_min_depth
        self.max_depth = max_depth
        self.depth = 0

        self.importance_sampling = importance_sampling
        self._important_path_weight = important_path_weight

        # ray statistics
        self.ray_count = 0
        self._primary_ray = None

    def __getstate__(self):
        """Encodes state for pickling."""

        return (
            super().__getstate__(),
            self._bins,
            self._min_wavelength,
            self._max_wavelength,
            self._extinction_prob,
            self._extinction_min_depth,
            self._max_depth,
            self.depth,
            self.importance_sampling,
            self._important_path_weight,
            self.ray_count,
            self._primary_ray
        )

    def __setstate__(self, state):
        """Decodes state for pickling."""

        (super_state,
         self._bins,
         self._min_wavelength,
         self._max_wavelength,
         self._extinction_prob,
         self._extinction_min_depth,
         self._max_depth,
         self.depth,
         self.importance_sampling,
         self._important_path_weight,
         self.ray_count,
         self._primary_ray) = state

        super().__setstate__(super_state)

    @property
    def bins(self):
        """
        Number of spectral bins across wavelength range.

        :rtype: int
        """
        return self._bins

    @bins.setter
    def bins(self, int bins):

        if bins < 1:
            raise ValueError("Number of bins cannot be less than 1.")
        self._bins = bins

    cdef int get_bins(self) nogil:
        return self._bins

    @property
    def min_wavelength(self):
        """
        Lower bound on wavelength range.

        :rtype: float
        """
        return self._min_wavelength

    @min_wavelength.setter
    def min_wavelength(self, double min_wavelength):

        if min_wavelength <= 0.0:
            raise ValueError("Wavelength must be greater than zero.")

        if min_wavelength > self._max_wavelength:
            raise ValueError("Maximum wavelength must be greater than minimum wavelength.")

        self._min_wavelength = min_wavelength

    cdef double get_min_wavelength(self) nogil:
        return self._min_wavelength

    @property
    def max_wavelength(self):
        """
        Upper bound on wavelength range.

        :rtype: float
        """
        return self._max_wavelength

    @max_wavelength.setter
    def max_wavelength(self, double max_wavelength):

        if max_wavelength <= 0.0:
            raise ValueError("Wavelength must be greater than zero.")

        if self.min_wavelength > max_wavelength:
            raise ValueError("Maximum wavelength must be greater than minimum wavelength.")

        self._max_wavelength = max_wavelength

    cdef double get_max_wavelength(self) nogil:
        return self._max_wavelength

    @property
    def wavelength_range(self):
        """
        Upper and lower wavelength range.

        :rtype: tuple
        """
        return self._min_wavelength, self._max_wavelength

    @wavelength_range.setter
    def wavelength_range(self, tuple range):

        min_wavelength, max_wavelength = range
        if min_wavelength <=0.0 or max_wavelength <= 0.0:
            raise ValueError("Wavelength must be greater than zero.")

        if self.min_wavelength > max_wavelength:
            raise ValueError("Maximum wavelength must be greater than minimum wavelength.")

        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength

    @property
    def extinction_prob(self):
        """
        Probability of path extinction at every material surface interaction.

        :rtype: float
        """
        return self._extinction_prob

    @extinction_prob.setter
    def extinction_prob(self, double extinction_prob):
        self._extinction_prob = clamp(extinction_prob, 0.0, 1.0)

    @property
    def extinction_min_depth(self):
        """
        Minimum number of paths before triggering extinction probability.

        :rtype: int
        """
        return self._extinction_min_depth

    @extinction_min_depth.setter
    def extinction_min_depth(self, int extinction_min_depth):
        if extinction_min_depth < 1:
            raise ValueError("The minimum extinction depth cannot be less than 1.")
        self._extinction_min_depth = extinction_min_depth

    @property
    def max_depth(self):
        """
        Maximum number of material interactions before terminating ray trajectory.

        :rtype: int
        """
        return self._max_depth

    @max_depth.setter
    def max_depth(self, int max_depth):
        if max_depth < self._extinction_min_depth:
            raise ValueError("The maximum depth cannot be less than the minimum depth.")
        self._max_depth = max_depth

    @property
    def important_path_weight(self):
        """
        Weight to use for important paths when using importance sampling.

        :rtype: float
        """
        return self._important_path_weight

    @important_path_weight.setter
    def important_path_weight(self, double important_path_weight):

        if important_path_weight < 0 or important_path_weight > 1.0:
            raise ValueError("Important path weight must be in the range [0, 1].")

        self._important_path_weight = important_path_weight

    cdef double get_important_path_weight(self) nogil:
        return self._important_path_weight

    cpdef Spectrum new_spectrum(self):
        """
        Returns a new Spectrum compatible with the ray spectral settings.

        :rtype: Spectrum

        .. code-block:: pycon

            >>> from raysect.core import Point3D, Vector3D
            >>> from raysect.optical import Ray
            >>>
            >>> ray = Ray(origin=Point3D(0, 0, -5),
            >>>           direction=Vector3D(0, 0, 1),
            >>>           min_wavelength=375,
            >>>           max_wavelength=785,
            >>>           bins=100)
            >>>
            >>> ray.new_spectrum()
            <raysect.optical.spectrum.Spectrum at 0x7f5b08b6e1b0>
        """

        return new_spectrum(self._min_wavelength, self._max_wavelength, self._bins)

    @cython.cdivision(True)
    cpdef Spectrum trace(self, World world, bint keep_alive=False):
        """
        Traces a single ray path through the world.

        :param World world: World object defining the scene.
        :param bool keep_alive: If true, disables Russian roulette termination of the ray.
        :return: The resulting Spectrum object collected by the ray.
        :rtype: Spectrum

        .. code-block:: pycon

            >>> from raysect.core import Point3D, Vector3D
            >>> from raysect.optical import World, Ray
            >>>
            >>> world = World()
            >>>
            >>> ray = Ray(origin=Point3D(0, 0, -5),
            >>>           direction=Vector3D(0, 0, 1),
            >>>           min_wavelength=375,
            >>>           max_wavelength=785,
            >>>           bins=100)
            >>>
            >>> spectrum = ray.trace(world)
            >>> spectrum
            <raysect.optical.spectrum.Spectrum at 0x7f5b08b6e048>
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

        # limit ray recursion depth with Russian roulette
        # set normalisation to ensure the sampling remains unbiased
        if keep_alive or self.depth < self._extinction_min_depth:
            normalisation = 1.0
        else:
            if self.depth >= self._max_depth or probability(self._extinction_prob):
                return self.new_spectrum()
            else:
                normalisation = 1 / (1 - self._extinction_prob)

        # does the ray intersect with any of the primitives in the world?
        intersection = world.hit(self)
        if intersection is None:
            return self.new_spectrum()

        # sample material
        spectrum = self._sample_surface(intersection, world)
        spectrum = self._sample_volumes(spectrum, intersection, world)

        # apply normalisation to ensure the sampling remains unbiased
        spectrum.mul_scalar(normalisation)
        return spectrum

    @cython.cdivision(True)
    cdef Spectrum _sample_surface(self, Intersection intersection, World world):

        cdef Material material

        # request surface contribution to spectrum from primitive material
        material = intersection.primitive.get_material()
        return material.evaluate_surface(world,
                                         self,
                                         intersection.primitive,
                                         intersection.hit_point,
                                         intersection.exiting,
                                         intersection.inside_point,
                                         intersection.outside_point,
                                         intersection.normal,
                                         intersection.world_to_primitive,
                                         intersection.primitive_to_world,
                                         intersection)

    cdef Spectrum _sample_volumes(self, Spectrum spectrum, Intersection intersection, World world):

        cdef:
            list primitives
            Point3D start_point, end_point
            Primitive primitive
            Material material

        # identify any primitive volumes the ray is propagating through
        primitives = world.contains(self.origin)
        if len(primitives) > 0:

            # the start and end points for volume contribution calculations
            # defined such that start to end is in the direction of light
            # propagation - from source to observer
            start_point = intersection.hit_point.transform(intersection.primitive_to_world)
            end_point = self.origin

            # accumulate volume contributions to the spectrum
            for primitive in primitives:

                material = primitive.get_material()
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

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef Spectrum sample(self, World world, int count):
        """
        Samples the radiance directed along the ray direction.

        This methods calls trace repeatedly to obtain a statistical sample of
        the radiance directed along the ray direction from the world. The count
        parameter specifies the number of samples to obtain. The mean spectrum
        accumulated from these samples is returned.

        :param World world: World object defining the scene.
        :param int count: Number of samples to take.
        :return: The accumulated spectrum collected by the ray.
        :rtype: Spectrum

        .. code-block:: pycon

            >>> from raysect.core import Point3D, Vector3D
            >>> from raysect.optical import World, Ray
            >>>
            >>> world = World()
            >>>
            >>> ray = Ray(origin=Point3D(0, 0, -5),
            >>>           direction=Vector3D(0, 0, 1),
            >>>           min_wavelength=375,
            >>>           max_wavelength=785,
            >>>           bins=100)
            >>>
            >>> ray.sample(world, 10)
            <raysect.optical.spectrum.Spectrum at 0x7f5b08b6e318>
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
            spectrum.mad_scalar(normalisation, sample.samples_mv)
            count -= 1

        return spectrum

    cpdef Ray spawn_daughter(self, Point3D origin, Vector3D direction):
        """
        Spawns a new daughter of the ray.

        A daughter ray has the same spectral configuration as the source ray,
        however the ray depth is increased by 1.

        :param Point3D origin: A Point3D defining the ray origin.
        :param Vector3D direction: A vector defining the ray direction.
        :return: A daughter Ray object.
        :rtype: Ray
        """

        cdef Ray ray

        ray = Ray.__new__(Ray)

        ray.origin = origin
        ray.direction = direction
        ray._bins = self._bins
        ray._min_wavelength = self._min_wavelength
        ray._max_wavelength = self._max_wavelength
        ray.max_distance = self.max_distance
        ray._extinction_prob = self._extinction_prob
        ray._extinction_min_depth = self._extinction_min_depth
        ray._max_depth = self._max_depth
        ray.importance_sampling = self.importance_sampling
        ray._important_path_weight = self._important_path_weight
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
        """
        Obtain a new Ray object with the same configuration settings.

        :param Point3D origin: New Ray's origin position.
        :param Vector3D direction: New Ray's direction.
        :rtype: Ray

        .. code-block:: pycon

            >>> from raysect.core import Point3D, Vector3D
            >>> from raysect.optical import Ray
            >>>
            >>> ray = Ray(origin=Point3D(0, 0, -5),
            >>>           direction=Vector3D(0, 0, 1),
            >>>           min_wavelength=375,
            >>>           max_wavelength=785,
            >>>           bins=100)
            >>>
            >>> ray.copy()
            Ray(Point3D(0.0, 0.0, -5.0), Vector3D(0.0, 0.0, 1.0), inf)
        """

        if origin is None:
            origin = self.origin.copy()

        if direction is None:
            direction =self.direction.copy()

        return new_ray(
            origin, direction,
            self._min_wavelength, self._max_wavelength, self._bins,
            self.max_distance,
            self._extinction_prob, self._extinction_min_depth, self._max_depth,
            self.importance_sampling,
            self._important_path_weight
        )

