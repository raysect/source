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
from libc.math cimport floor
cimport cython


cdef class VolumeIntegrator:

    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        raise NotImplementedError("Virtual method integrate() has not been implemented.")


cdef class NumericalIntegrator(VolumeIntegrator):

    def __init__(self, step, min_samples=5):

        self._step = step
        self._min_samples = min_samples

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, double value):
        if value <= 0:
            raise ValueError("Numerical integration step size can not be less than or equal to zero")
        self._step = value

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, int value):
        if value < 2:
            raise ValueError("At least two samples are required to perform the numerical integration.")
        self._min_samples = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D integration_direction, ray_direction
            double length, step, t, c
            Spectrum emission, emission_previous, temp
            int intervals, interval, index

        # convert start and end points to local space
        start = start_point.transform(world_to_primitive)
        end = end_point.transform(world_to_primitive)

        # obtain local space ray direction and integration length
        integration_direction = start.vector_to(end)
        length = integration_direction.get_length()

        # nothing to contribute?
        if length == 0.0:
            return spectrum

        integration_direction = integration_direction.normalise()
        ray_direction = integration_direction.neg()

        # calculate number of complete intervals (samples - 1)
        intervals = max(self._min_samples - 1, <int> floor(length / self._step))

        # adjust (increase) step size to absorb any remainder and maintain equal interval spacing
        step = length / intervals

        # create working buffers
        emission = ray.new_spectrum()
        emission_previous = ray.new_spectrum()

        # sample point and sanity check as bounds checking is disabled
        emission_previous = material.emission_function(start, ray_direction, emission_previous, world, ray, primitive, world_to_primitive, primitive_to_world)
        self._check_dimensions(emission_previous, spectrum.bins)

        # numerical integration
        c = 0.5 * step
        for interval in range(1, intervals):

            # calculate location of new sample point
            t = interval * step
            sample_point = new_point3d(
                start.x + t * integration_direction.x,
                start.y + t * integration_direction.y,
                start.z + t * integration_direction.z
            )

            # sample point and sanity check as bounds checking is disabled
            emission = material.emission_function(sample_point, ray_direction, emission, world, ray, primitive, world_to_primitive, primitive_to_world)
            self._check_dimensions(emission, spectrum.bins)

            # trapezium rule integration
            for index in range(spectrum.bins):
                spectrum.samples_mv[index] += c * (emission.samples_mv[index] + emission_previous.samples_mv[index])

            # swap buffers and clear the active buffer
            temp = emission_previous
            emission_previous = emission
            emission = temp
            emission.clear()

        return spectrum

    cdef inline int _check_dimensions(self, Spectrum spectrum, int bins) except -1:
        if spectrum.samples.ndim != 1 or spectrum.samples.shape[0] != bins:
            raise ValueError("Spectrum returned by emission function has the wrong number of samples.")


cdef class InhomogeneousVolumeEmitter(NullSurface):

    def __init__(self, VolumeIntegrator integrator=None):

        super().__init__()
        self.integrator = integrator or NumericalIntegrator(step=0.01, min_samples=5)
        self.importance = 1.0

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # pass to volume integrator class
        return self.integrator.integrate(spectrum, world, ray, primitive, self, start_point, end_point,
                                         world_to_primitive, primitive_to_world)

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")

