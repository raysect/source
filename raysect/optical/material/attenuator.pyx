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
from libc.math cimport sqrt, fabs
cimport cython


from libc.math cimport pow as cpow
cimport cython


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
            MuellerMatrix rotate_in, rotate_out, mm
            StokesVector s

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

        # obtain rotation matrices to rotate stokes vector into an out of the mueller matrix frame
        rotate_in = self._rotate_vector(p_orientation, matrix_orientation, p_direction)
        rotate_out = self._rotate_vector(matrix_orientation, p_orientation, p_direction)

        # apply di-attenuation
        for bin in range(spectrum.bins):

            # populate stokes vector
            s = new_stokesvector(
                spectrum.samples_mv[bin, 0],
                spectrum.samples_mv[bin, 1],
                spectrum.samples_mv[bin, 2],
                spectrum.samples_mv[bin, 3]
            )

            # calculate projected transmission
            k = fabs(k)
            ty = cpow(perpendicular_transmission[bin], length)
            tx = cpow(parallel_transmission[bin], length)
            tx = (1 - k) * tx + k * ty

            # generate mueller matrix and apply
            mm = rotate_out.mul(diattenuator(tx, ty).mul(rotate_in))
            s = s.apply(mm)

            # populate bin
            spectrum.samples_mv[bin, 0] = s.i
            spectrum.samples_mv[bin, 1] = s.q
            spectrum.samples_mv[bin, 2] = s.u
            spectrum.samples_mv[bin, 3] = s.v

        return spectrum

    cdef MuellerMatrix _rotate_vector(self, Vector3D source, Vector3D target, Vector3D direction):
        """
        Generates a Mueller matrix that rotates from source to target frames.
        
        The direction is the light propagation direction. Both the source and
        target vectors must be orthogonal to direction. All vectors must be
        unit vectors.
                
        :param source: Source orientation vector. 
        :param target: Target orientation vector. 
        :param direction: Light propagation direction.
        :return: MuellerMatrix
        """

        # calculate rotation around direction vector
        cdef double angle = source.angle(target)
        if direction.dot(source.cross(target)) < 0:
            angle = -angle

        return rotate_angle(angle)