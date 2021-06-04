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


cdef class Retarder(NullSurface):
    """
    An ideal optical retarder.

    The retarder material behaves as an idealised uni-axial crystal. Light
    passing through the retarder experiences a change of phase, determined
    by the angle of propagation vs the retarder axis. The phase shift has
    no wavelength dependence.

    The retarder axis defines the "fast" axis of propagation. Light with an
    electric field aligned with the axis propagates faster than light with
    an electric field perpendicular to the axis (the "slow" axes).

    The phase shift parameter defines the amount of phase shift introduced
    per meter of propagation though the material for a ray propagating
    perpendicular to the retarder axis. The accumulated phase shift reduces
    according to the sin of the angle between the ray direction and the
    optical axis.
    """

    def __init__(self, Vector3D axis=None, double phase_shift):
        """
        :param axis: Optical axis vector.
        :param phase_shift: Phase shift per meter in degrees.
        """

        super().__init__()
        self.axis = axis or Vector3D(0, 1, 0)
        self.phase_shift = phase_shift

    @property
    def phase_shift(self):
        return self._phase_shift

    @phase_shift.setter
    def phase_shift(self, double value):
        # prevent -ve phase shift so fast axis is always the fast axis
        if value < 0:
            raise ValueError('The phase shift cannot be less than zero.')
        self._phase_shift = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum evaluate_volume(
        self, Spectrum spectrum, World world, Ray ray, Primitive primitive, Point3D start_point,
        Point3D end_point, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            double k
            int bin, component

        # # transmission path length
        # w_light_path = start_point.vector_to(end_point)
        # todo: calculate path length

        # # obtain light propagation direction and electric field orientation in primitive space
        # p_direction = w_light_path.transform(world_to_primitive)
        # p_orientation = ray.orientation.transform(world_to_primitive)
        #
        # # normalise to reduce risk of numerical issues
        # p_direction = p_direction.normalise()
        # p_orientation = p_orientation.normalise()
        # axis = self.axis.normalise()
        #
        # # calculate projection of material axis perpendicular to the light direction
        # k = p_direction.dot(axis)

        # if light direction is parallel to the axis, transmission is entirely perpendicular to fast-axis and non retarding
        if 1 - fabs(k) < EPSILON:
            return spectrum

        # # calculate projection of axis orthogonal to the propagation direction
        # matrix_orientation = new_vector3d(
        #     axis.x - k * p_direction.x,
        #     axis.y - k * p_direction.y,
        #     axis.z - k * p_direction.z
        # )
        # matrix_orientation = matrix_orientation.normalise()

        # todo: calculate phase shift (path length * sin theta = sqrt(1 - k**2)


        # # calculate rotation about direction vector
        # angle = p_orientation.angle(matrix_orientation) * DEG2RAD
        # if p_direction.dot(p_orientation.cross(matrix_orientation)) < 0:
        #     angle = -angle
        #
        # # apply retarder
        # for bin in range(spectrum.bins):
        #     self._apply_rotated_polariser(spectrum, bin, angle)

        return spectrum
    #
    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.initializedcheck(False)
    # cdef void _apply_rotated_polariser(self, Spectrum spectrum, int bin, double angle) nogil:
    #
    #     cdef double s0, s1, s2, c, s, k0, k1, k2
    #
    #     # precalculate common terms
    #     c = cos(2 * angle)
    #     s = sin(2 * angle)
    #     k0 = c * c
    #     k1 = s * c
    #     k2 = s * s
    #
    #     # obtain stokes parameters
    #     s0 = spectrum.samples_mv[bin, 0]
    #     s1 = spectrum.samples_mv[bin, 1]
    #     s2 = spectrum.samples_mv[bin, 2]
    #
    #     # optimised mueller matrix multiplication
    #     spectrum.samples_mv[bin, 0] = 0.5 * (s0 + c*s1 + s*s2)
    #     spectrum.samples_mv[bin, 1] = 0.5 * (c*s0 + k0*s1 + k1*s2)
    #     spectrum.samples_mv[bin, 2] = 0.5 * (s*s0 + k1*s1 + k2*s2)
    #     spectrum.samples_mv[bin, 3] = 0.0
