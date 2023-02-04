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

from raysect.optical cimport Point3D, Vector3D, AffineMatrix3D, Primitive, World, Ray, Spectrum, ConstantSF, new_vector3d
from raysect.optical.mueller cimport diattenuator, rotate_angle
from raysect.optical.stokes cimport StokesVector, new_stokesvector
from libc.math cimport sqrt, fabs, sin, cos, pow as cpow
cimport cython

DEF RAD2DEG = 57.29577951308232000  # 180 / pi
DEF DEG2RAD = 0.017453292519943295  # pi / 180


cdef class UniformAttenuator(NullSurface):
    """
    An isotropic and homogeneous attenuating volume.
    """

    def __init__(self, SpectralFunction transmission=None):
        super().__init__()
        self.transmission = transmission or ConstantSF(0.5)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum evaluate_volume(
        self, Spectrum spectrum, World world, Ray ray, Primitive primitive, Point3D start_point,
        Point3D end_point, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            double length
            double[::1] transmission
            int bin, component

        length = start_point.vector_to(end_point).get_length()
        transmission = self.transmission.sample_mv(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
        for bin in range(spectrum.bins):
            for component in range(4):
                spectrum.samples_mv[bin, component] *= cpow(transmission[bin], length)
        return spectrum


cdef class UniformDiattenuator(NullSurface):
    """
    A homogeneous di-attenuating volume.

    A di-attenuating material is a transmitting material where the transmission
    depends on the polarisation state of the incident light. The transmission
    of linear polarised light parallel with the material axis is different to
    the transmission perpendicular to the material axis.
    """

    cdef:
        public Vector3D axis
        public SpectralFunction parallel_transmission, perpendicular_transmission

    def __init__(self, axis=None, SpectralFunction parallel_transmission=None, SpectralFunction perpendicular_transmission=None):
        super().__init__()
        self.axis = axis or Vector3D(0, 1, 0)
        self.parallel_transmission = parallel_transmission or ConstantSF(1.0)
        self.perpendicular_transmission = perpendicular_transmission or ConstantSF(0.0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum evaluate_volume(
        self, Spectrum spectrum, World world, Ray ray, Primitive primitive, Point3D start_point,
        Point3D end_point, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            double length, tx, ty, k
            double[::1] parallel_transmission, perpendicular_transmission
            int bin, component

        # transmission path length
        w_light_path = start_point.vector_to(end_point)
        length = w_light_path.get_length()

        # obtain light propagation direction and electric field orientation in primitive space
        p_direction = w_light_path.transform(world_to_primitive)
        p_orientation = ray.orientation.transform(world_to_primitive)

        # normalise to reduce risk of numerical issues
        p_direction = p_direction.normalise()
        p_orientation = p_orientation.normalise()

        # resample transmission functions
        parallel_transmission = self.parallel_transmission.sample_mv(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
        perpendicular_transmission = self.perpendicular_transmission.sample_mv(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)

        # calculate projection of material axis perpendicular to the light direction
        axis = self.axis.normalise()
        k = p_direction.dot(axis)

        # if light direction is parallel to the axis, transmission is entirely perpendicular
        if k == 1:
            for bin in range(spectrum.bins):
                for component in range(4):
                    spectrum.samples_mv[bin, component] *= cpow(perpendicular_transmission[bin], length)
            return spectrum

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

        # only require magnitude of cosine for transmission calculation
        k = fabs(k)

        # apply di-attenuation
        for bin in range(spectrum.bins):

            # calculate projected transmission and apply
            ty = cpow(perpendicular_transmission[bin], length)
            tx = cpow(parallel_transmission[bin], length)
            tx = (1 - k) * tx + k * ty
            self._apply_rotated_diattenuator(spectrum, bin, tx, ty, angle)

        return spectrum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _apply_rotated_diattenuator(self, Spectrum spectrum, int bin, double tx, double ty, double angle) nogil:

        cdef double s0, s1, s2, s3

        # precalculate common terms
        c = cos(2 * angle)
        s = sin(2 * angle)
        t0 = 0.5 * (tx + ty)
        t1 = 0.5 * (tx - ty)
        t2 = sqrt(tx * ty)
        t3 = t1 * c
        t4 = t1 * s
        t5 = t0 * c * c + t2 * s * s
        t6 = t0 * s * c - t2 * s * c
        t7 = t0 * s * s + t2 * c * c

        # obtain stokes parameters
        s0 = spectrum.samples_mv[bin, 0]
        s1 = spectrum.samples_mv[bin, 1]
        s2 = spectrum.samples_mv[bin, 2]
        s3 = spectrum.samples_mv[bin, 3]

        # optimised mueller matrix multiplication
        spectrum.samples_mv[bin, 0] = t0 * s0 + t3 * s1 + t4 * s2
        spectrum.samples_mv[bin, 1] = t3 * s0 + t5 * s1 + t6 * s2
        spectrum.samples_mv[bin, 2] = t4 * s0 + t6 * s1 + t7 * s2
        spectrum.samples_mv[bin, 3] = t2 * s3
