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

from raysect.optical.spectrum cimport new_spectrum_array

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

cdef class Ray(CoreRay):

    def __init__(self,
                 Point origin = Point([0,0,0]),
                 Vector direction = Vector([0,0,1]),
                 list wavebands = list(),
                 double refraction_wavelength = 550,
                 double max_distance = INFINITY,
                 double max_depth = 15):

        super().__init__(origin, direction, max_distance)

        self.refraction_wavelength = refraction_wavelength
        self.wavebands = wavebands

        self.max_depth = max_depth
        self.depth = 0

    property refraction_wavelength:

        def __get__(self):

            return self._refraction_wavelength

        def __set__(self, double wavelength):

            if wavelength <= 0:

                raise ValueError("Wavelength can not be less than or equal to zero.")

            self._refraction_wavelength = wavelength

    property wavebands:

        def __get__(self):

            return self._wavebands

        def __set__(self, list wavebands not None):

            # objects must be Waveband objects as cython implemented materials
            # require Waveband's cython interface
            for waveband in wavebands:

                if not isinstance(waveband, Waveband):

                    raise TypeError("The Ray spectral waveband definition must be a list of Waveband objects.")

            self._wavebands = wavebands

    def __getitem__(self, int index):

        return self._wavebands[index]

    def __setitem__(self, int index, Waveband waveband not None):

        self._wavebands[index] = waveband

    cpdef ndarray trace(self, World world):

        cdef ndarray spectrum
        cdef Intersection intersection
        cdef list primitives
        cdef Primitive primitive
        cdef Point start_point, end_point
        cdef Material material

        spectrum = new_spectrum_array(self)

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
        ray.max_distance = self.max_distance
        ray._refraction_wavelength = self._refraction_wavelength
        ray._wavebands = self._wavebands
        ray.max_depth = self.max_depth
        ray.depth = self.depth + 1

        return ray

    cdef inline double get_refraction_wavelength(self):

        return self._refraction_wavelength

    cdef inline int get_waveband_count(self):

        return len(self._wavebands)

    cdef inline Waveband get_waveband(self, int index):

        return self._wavebands[index]


def monocromatic_ray(origin, direction, min_wavelength, max_wavelength, max_distance = INFINITY):

    return Ray(origin,
               direction,
               [Waveband(min_wavelength, max_wavelength)],
               0.5 * (min_wavelength + max_wavelength),
               max_distance)


def polycromatic_ray(origin, direction, min_wavelength, max_wavelength, steps, max_distance = INFINITY):

    wavebands = list()

    delta_wavelength = (max_wavelength - min_wavelength) / steps

    for index in range(0, steps):

        wavebands.append(Waveband(min_wavelength + delta_wavelength * index,
                                  min_wavelength + delta_wavelength * (index + 1)))

    return Ray(origin,
               direction,
               wavebands,
               0.5 * (min_wavelength + max_wavelength),
               max_distance)
