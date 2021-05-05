# cython: language_level=3

# Copyright (c) 2014-2020, Dr Alex Meakins, Raysect Project
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

from raysect.optical cimport Point3D, Vector3D, AffineMatrix3D, Primitive, World, Ray, Spectrum, new_vector3d
from libc.math cimport sin, cos, fabs
cimport cython

DEF EPSILON = 1e-12
DEF RAD2DEG = 57.29577951308232000  # 180 / pi
DEF DEG2RAD = 0.017453292519943295  # pi / 180


cdef class LinearPolariser(NullSurface):

    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis or Vector3D(0, 1, 0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum evaluate_volume(
        self, Spectrum spectrum, World world, Ray ray, Primitive primitive, Point3D start_point,
        Point3D end_point, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            double k
            int bin, component

        # transmission path length
        w_light_path = start_point.vector_to(end_point)

        # obtain light propagation direction and electric field orientation in primitive space
        p_direction = w_light_path.transform(world_to_primitive)
        p_orientation = ray.orientation.transform(world_to_primitive)

        # normalise to reduce risk of numerical issues
        p_direction = p_direction.normalise()
        p_orientation = p_orientation.normalise()
        axis = self.axis.normalise()

        # calculate projection of material axis perpendicular to the light direction
        k = p_direction.dot(axis)

        # if light direction is parallel to the axis, transmission is entirely perpendicular and non absorbing
        if 1 - fabs(k) < EPSILON:
            return spectrum

        # calculate projection of axis orthogonal to the propagation direction
        matrix_orientation = new_vector3d(
            axis.x - k * p_direction.x,
            axis.y - k * p_direction.y,
            axis.z - k * p_direction.z
        )
        matrix_orientation = matrix_orientation.normalise()

        # calculate rotation about direction vector
        angle = p_orientation.angle(matrix_orientation) * DEG2RAD
        if p_direction.dot(p_orientation.cross(matrix_orientation)) < 0:
            angle = -angle

        # apply polarisation
        for bin in range(spectrum.bins):
            self._apply_rotated_polariser(spectrum, bin, angle)

        return spectrum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _apply_rotated_polariser(self, Spectrum spectrum, int bin, double angle) nogil:

        cdef double s0, s1, s2, c, s, k0, k1, k2

        # precalculate common terms
        c = cos(2 * angle)
        s = sin(2 * angle)
        k0 = c * c
        k1 = s * c
        k2 = s * s

        # obtain stokes parameters
        s0 = spectrum.samples_mv[bin, 0]
        s1 = spectrum.samples_mv[bin, 1]
        s2 = spectrum.samples_mv[bin, 2]

        # optimised mueller matrix multiplication
        spectrum.samples_mv[bin, 0] = 0.5 * (s0 + c*s1 + s*s2)
        spectrum.samples_mv[bin, 1] = 0.5 * (c*s0 + k0*s1 + k1*s2)
        spectrum.samples_mv[bin, 2] = 0.5 * (s*s0 + k1*s1 + k2*s2)
        spectrum.samples_mv[bin, 3] = 0.0
