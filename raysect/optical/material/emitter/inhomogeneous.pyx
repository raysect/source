# cython: language_level=3

# Copyright (c) 2014-2017, Dr Alex Meakins, Raysect Project
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

from raysect.optical cimport new_point3d
cimport cython


cdef class InhomogeneousVolumeEmitter(NullSurface):

    def __init__(self, double step = 0.01):

        super().__init__()
        self._step = step
        self.importance = 1.0

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, double value):
        if value <= 0:
            raise ValueError("Numerical integration step size can not be less than or equal to zero")
        self._step = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D integration_direction, ray_direction
            double length, t, c
            Spectrum emission, emission_previous, temp
            int index

        # convert start and end points to local space
        start = start_point.transform(world_to_primitive)
        end = end_point.transform(world_to_primitive)

        # obtain local space ray direction and integration length
        integration_direction = start.vector_to(end)
        length = integration_direction.get_length()

        # nothing to contribute?
        if length == 0:
            return spectrum

        integration_direction = integration_direction.normalise()
        ray_direction = integration_direction.neg()

        # create working buffers
        emission = ray.new_spectrum()
        emission_previous = ray.new_spectrum()

        # sample point and sanity check as bounds checking is disabled
        emission_previous = self.emission_function(start, ray_direction, emission_previous, world, ray, primitive, world_to_primitive, primitive_to_world)
        self._check_dimensions(emission_previous, spectrum.bins)

        # numerical integration
        t = self._step
        c = 0.5 * self._step
        while t <= length:

            sample_point = new_point3d(
                start.x + t * integration_direction.x,
                start.y + t * integration_direction.y,
                start.z + t * integration_direction.z
            )

            # sample point and sanity check as bounds checking is disabled
            emission = self.emission_function(sample_point, ray_direction, emission, world, ray, primitive, world_to_primitive, primitive_to_world)
            self._check_dimensions(emission, spectrum.bins)

            # trapezium rule integration
            for index in range(spectrum.bins):
                spectrum.samples_mv[index] += c * (emission.samples_mv[index] + emission_previous.samples_mv[index])

            # swap buffers and clear the active buffer
            temp = emission_previous
            emission_previous = emission
            emission = temp
            emission.clear()

            t += self._step

        # step back to process any length that remains
        t -= self._step

        # sample point and sanity check as bounds checking is disabled
        emission = self.emission_function(end, ray_direction, emission, world, ray, primitive, world_to_primitive, primitive_to_world)
        self._check_dimensions(emission, spectrum.bins)

        # trapezium rule integration of remainder
        c = 0.5 * (length - t)
        for index in range(spectrum.bins):
            spectrum.samples_mv[index] += c * (emission.samples_mv[index] + emission_previous.samples_mv[index])
        return spectrum

    cdef inline int _check_dimensions(self, Spectrum spectrum, int bins) except -1:
        if spectrum.samples.ndim != 1 or spectrum.samples.shape[0] != bins:
            raise ValueError("Spectrum returned by emission function has the wrong number of samples.")

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")
