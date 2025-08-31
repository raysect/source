# cython: language_level=3

# Copyright (c) 2014-2025, Dr Alex Meakins, Raysect Project
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

from libc.math cimport INFINITY

from raysect.core cimport Intersection, Point3D, Vector3D
from raysect.core.math.random cimport probability

from raysect.optical.ray cimport Ray
from raysect.optical.scenegraph cimport World, Primitive
from raysect.optical.spectrum cimport Spectrum
from raysect.optical.material.material cimport Material

cimport cython


cdef class LoggingRay(Ray):
    """
    LoggingRay is a subclass of Ray that provides additional logging functionality.

    A LoggingRay records all intersections it encounters during its path through the scene.

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

    :ivar list[Intersection] log: List of intersections encountered by the ray.
    :ivar list[Point3D] path_vertices: List of vertices defining the ray's path.

    >>> from raysect.core import Point3D, Vector3D
    >>> from raysect.primitive import Box
    >>> from raysect.optical import World
    >>> from raysect.optical.material import NullMaterial
    >>> from raysect.optical.loggingray import LoggingRay
    >>>
    >>> world = World()
    >>> box = Box(Point3D(-1, -1, 0), Point3D(1, 1, 1),
    >>>           material=NullMaterial(), parent=world)
    >>>
    >>> ray = LoggingRay(origin=Point3D(0, 0, -5),
    >>>                  direction=Vector3D(0, 0, 1))
    >>>
    >>> ray.trace(world)
    >>> ray.log
    [Intersection(...), Intersection(...)]
    >>>
    >>> ray.path_vertices
    [Point3D(0.0, 0.0, -5.0), Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, 1.0)]
    """

    cdef public list log

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

        super().__init__(origin, direction, min_wavelength, max_wavelength,
                         bins, max_distance, extinction_prob, extinction_min_depth,
                         max_depth, importance_sampling, important_path_weight)

        self.log = None

    def __getstate__(self):
        """Encodes state for pickling."""

        return (
            super().__getstate__(),
            self.log
        )

    def __setstate__(self, state):
        """Decodes state for pickling."""

        (super_state, self.log) = state

        super().__setstate__(super_state)

    @cython.cdivision(True)
    cpdef Spectrum trace(self, World world, bint keep_alive=False):
        """
        Traces a single ray path through the world.

        This method is responsible for tracing the ray through the scene, handling
        all interactions with the environment.

        :param World world: World object defining the scene.
        :param bool keep_alive: If true, disables Russian roulette termination of the ray.
        :return: A Spectrum object.
        :rtype: Spectrum

        >>> world = World()
        >>> ray = LoggingRay(origin=Point3D(0, 0, -5),
        >>>                  direction=Vector3D(0, 0, 1))
        >>> spectrum = ray.trace(world)
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
            self.log = []

        # create a new spectrum object compatible with the ray
        spectrum = self.new_spectrum()

        # limit ray recursion depth with Russian roulette
        # set normalisation to ensure the sampling remains unbiased
        if keep_alive or self.depth < self._extinction_min_depth:
            normalisation = 1.0
        else:
            if self.depth >= self._max_depth or probability(self._extinction_prob):
                return spectrum
            else:
                normalisation = 1 / (1 - self._extinction_prob)

        # does the ray intersect with any of the primitives in the world?
        intersection = world.hit(self)
        if intersection is not None:
            self.log.append(intersection)
            spectrum = self._sample_surface(intersection, world)
            spectrum = self._sample_volumes(spectrum, intersection, world)

        # apply normalisation to ensure the sampling remains unbiased
        spectrum.mul_scalar(normalisation)
        return spectrum

    cpdef LoggingRay spawn_daughter(self, Point3D origin, Vector3D direction):
        """
        Obtain a new LoggingRay object with the same configuration settings.

        :param Point3D origin: New Ray's origin position.
        :param Vector3D direction: New Ray's direction.
        :rtype: LoggingRay
        """

        cdef LoggingRay ray

        ray = LoggingRay.__new__(LoggingRay)

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
        ray.log = self.log

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

    cpdef LoggingRay copy(self, Point3D origin=None, Vector3D direction=None):
        """
        Obtain a new LoggingRay object with the same configuration settings.

        :param Point3D origin: New Ray's origin position.
        :param Vector3D direction: New Ray's direction.
        :rtype: LoggingRay
        """
        if origin is None:
            origin = self.origin.copy()

        if direction is None:
            direction =self.direction.copy()

        return LoggingRay(origin, direction, self._min_wavelength, self._max_wavelength, self._bins,
                          self.max_distance, self._extinction_prob, self._extinction_min_depth,
                          self._max_depth, self.importance_sampling, self._important_path_weight)

    @property
    def path_vertices(self):
        """
        List of vertices defining the ray's path.

        The list includes the ray's origin and all intersection points along its path.

        :rtype: list[Point3D]

        >>> ray = LoggingRay(Point3D(0, 0, 0), Vector3D(0, 0, 1))
        >>> ray.path_vertices
        [Point3D(0, 0, 0), Point3D(...), ... ]
        """

        cdef:
            list vertices
            Intersection intersection

        if self.log:
            vertices = [self.origin.copy()]
            vertices += [intersection.hit_point.transform(intersection.primitive_to_world) for intersection in self.log]
            return vertices
        else:
            return []
