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

# TODO: POLARISATION

from raysect.core.math.random cimport probability
from raysect.optical cimport Point3D, Vector3D, new_vector3d, Normal3D, AffineMatrix3D, World, Primitive, ConstantSF, Ray
from libc.math cimport fabs, sqrt, pow as cpow, atan2, cos, sin
cimport cython

DEF EPSILON = 1e-12
DEF RAD2DEG = 57.29577951308232000  # 180 / pi
DEF DEG2RAD = 0.017453292519943295  # pi / 180


cdef class Sellmeier(NumericallyIntegratedSF):
    """
    Material with refractive index defined by `Sellmeier equation <https://en.wikipedia.org/wiki/Sellmeier_equation>`_

    :param float b1: Sellmeier :math:`B_1` coefficient.
    :param float b2: Sellmeier :math:`B_2` coefficient.
    :param float b3: Sellmeier :math:`B_3` coefficient.
    :param float c1: Sellmeier :math:`C_1` coefficient.
    :param float c2: Sellmeier :math:`C_2` coefficient.
    :param float c3: Sellmeier :math:`B_1` coefficient.
    :param float sample_resolution: The numerical sampling resolution in nanometers.

    .. code-block:: pycon

        >>> from raysect.optical import ConstantSF
        >>> from raysect.optical.material import Dielectric, Sellmeier
        >>>
        >>> diamond_material = Dielectric(Sellmeier(0.3306, 4.3356, 0.0, 0.1750**2, 0.1060**2, 0.0),
                                          ConstantSF(1))
    """

    def __init__(self, double b1, double b2, double b3, double c1, double c2, double c3, double sample_resolution=1):
        super().__init__(sample_resolution)

        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def __getstate__(self):
        """Encodes state for pickling."""

        return (
            self.b1,
            self.b2,
            self.b3,
            self.c1,
            self.c2,
            self.c3,
            super().__getstate__()
        )

    def __setstate__(self, state):
        """Decodes state for pickling."""

        (
            self.b1,
            self.b2,
            self.b3,
            self.c1,
            self.c2,
            self.c3,
            super_state
        ) = state
        super().__setstate__(super_state)

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @cython.cdivision(True)
    cpdef double function(self, double wavelength):
        """
        Returns a sample of the three term Sellmeier equation at the specified
        wavelength.

        :param float wavelength: Wavelength in nm.
        :return: Refractive index sample.
        :rtype: float
        """

        # wavelength in Sellmeier eqn. is specified in micrometers
        cdef double w2 = wavelength * wavelength * 1e-6
        return sqrt(1 + (self.b1 * w2) / (w2 - self.c1)
                      + (self.b2 * w2) / (w2 - self.c2)
                      + (self.b3 * w2) / (w2 - self.c3))


# todo: apply mueller matrix per bin for better spectral accuracy when not using dispersion, like conductor
cdef class Dielectric(Material):
    """
    An ideal dielectric material.

    :param SpectralFunction index: Refractive index as a function of wavelength.
    :param SpectralFunction transmission: Transmission per metre as a function of wavelength.
    :param SpectralFunction external_index: Refractive index of the external material at the interface,
      defaults to a vacuum (n=1).
    :param bool transmission_only: toggles transmission only, no reflection (default=False).

    .. code-block:: pycon

        >>> from raysect.optical import ConstantSF
        >>> from raysect.optical.material import Dielectric, Sellmeier
        >>>
        >>> diamond_material = Dielectric(Sellmeier(0.3306, 4.3356, 0.0, 0.1750**2, 0.1060**2, 0.0),
                                          ConstantSF(1))
    """

    def __init__(self, SpectralFunction index, SpectralFunction transmission, SpectralFunction external_index=None):
        super().__init__()
        self.index = index
        self.transmission = transmission

        if external_index is None:
            self.external_index = ConstantSF(1.0)
        else:
            self.external_index = external_index

        self.importance = 1.0

    @cython.cdivision(True)
    cpdef Spectrum evaluate_surface(
        self, World world, Ray ray, Primitive primitive, Point3D hit_point,
        bint exiting, Point3D inside_point, Point3D outside_point, Normal3D surface_normal,
        AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # Raysect is a reverse raytracer, so the inbound ray is the outbound ray
        # from the perspective of the light propagation. Be aware that the
        # transmitted, reflected and incident rays in this material refer to
        # the raytracing direction, not the light propagation direction. The
        # inbound ray corresponds to either the transmitted or reflected ray of
        # an incident light beam. This material samples one of the possible
        # light paths that would result in the propagation down the incident
        # ray path.
        #
        # Rays launched by this material are aligned with the interface frame,
        # the ray orientation vector lies in the plane of incidence.

        # convert ray direction to local coordinates
        i_direction = ray.direction.transform(world_to_primitive)

        # ensure vectors are normalised for reflection calculation
        i_direction = i_direction.normalise()
        normal = surface_normal.as_vector().normalise()

        # calculate signed cosine of angle between incident and normal
        k = -normal.dot(i_direction)

        # are we entering or leaving material - calculate refractive change
        # note, we do not use the supplied exiting parameter as the normal is
        # not guaranteed to be perpendicular to the surface for some primitives
        # (e.g. mesh with normal interpolation)
        # todo: modify mesh to use facet normal for source point calculation, use interpolated normal for everything
        #  then this calculation can be removed - caveats with interpolated normals should be highlighted in mesh docs
        exiting = k < 0.0

        # sample refractive indices
        internal_index = self.index.average(ray.get_min_wavelength(), ray.get_max_wavelength())
        external_index = self.external_index.average(ray.get_min_wavelength(), ray.get_max_wavelength())

        # map normal, points and indices to lie on the right side of the surface relative to the incident ray
        if exiting:

            # flip normal to point into the primitive
            normal = normal.neg()

            # ray launch points
            r_origin = inside_point
            t_origin = outside_point

            # refractive indices
            ni = internal_index
            nt = external_index

        else:

            # ray launch points
            r_origin = outside_point
            t_origin = inside_point

            # entering material
            ni = external_index
            nt = internal_index

        gamma = ni / nt

        # incident and transmitted cosine magnitudes required for fresnel calculation
        ci = fabs(k)
        ct_sqr = 1.0 - (gamma*gamma) * (1.0 - ci*ci)
        ct = sqrt(ct_sqr)

        # establish polarisation frame for fresnel calculation
        # If the incident ray and normal are collinear, an arbitrary orthogonal
        # vector is generated. In the collinear case this orientation must be
        # replicated for the transmitted and reflected rays or the fresnel
        # calculation will be invalid.
        i_orientation = i_direction.orthogonal(normal)

        # check for total internal reflection
        if ct_sqr <= 0:

            # calculate direction and orientation
            temp = 2 * ci
            r_direction = new_vector3d(
                i_direction.x + temp * normal.x,
                i_direction.y + temp * normal.y,
                i_direction.z + temp * normal.z
            )
            r_orientation = r_direction.orthogonal(normal) if (1.0 - ci) > EPSILON else i_orientation

            # launch reflected ray
            reflected_ray = ray.spawn_daughter(
                r_origin.transform(primitive_to_world),
                r_direction.transform(primitive_to_world),
                r_orientation.transform(primitive_to_world)
            )
            spectrum = reflected_ray.trace(world)

            # apply mueller matrix
            self._apply_mueller_reflection_tir(spectrum, ci, gamma)

        else:

            # calculate fresnel reflection and transmission coefficients
            rp, rs, tp, ts, ta = self._fresnel(ci, ct, ni, nt)
            transmission = 0.5*ta*(ts*ts + tp*tp)

            # select path by roulette using the strength of the coefficients as probabilities
            if probability(transmission):

                # transmitted ray path selected
                temp = gamma * ci - ct
                t_direction = new_vector3d(
                    gamma * i_direction.x + temp * normal.x,
                    gamma * i_direction.y + temp * normal.y,
                    gamma * i_direction.z + temp * normal.z
                )
                t_orientation = t_direction.orthogonal(normal.neg()).neg() if (1.0 - ci) > EPSILON else i_orientation

                # spawn ray on correct side of surface
                transmitted_ray = ray.spawn_daughter(
                    t_origin.transform(primitive_to_world),
                    t_direction.transform(primitive_to_world),
                    t_orientation.transform(primitive_to_world)
                )
                spectrum = transmitted_ray.trace(world)

                # apply normalised mueller matrix
                self._apply_mueller_transmission(spectrum, tp, ts)

            else:

                # reflected ray path selected
                temp = 2 * ci
                r_direction = new_vector3d(
                    i_direction.x + temp * normal.x,
                    i_direction.y + temp * normal.y,
                    i_direction.z + temp * normal.z
                )
                r_orientation = r_direction.orthogonal(normal) if (1.0 - ci) > EPSILON else i_orientation

                # spawn reflected ray
                reflected_ray = ray.spawn_daughter(
                    r_origin.transform(primitive_to_world),
                    r_direction.transform(primitive_to_world),
                    r_orientation.transform(primitive_to_world)
                )
                spectrum = reflected_ray.trace(world)

                # apply normalised mueller matrix
                self._apply_mueller_reflection_non_tir(spectrum, rp, rs)

        # ray stokes orientation
        s_orientation = ray.orientation.transform(world_to_primitive)
        s_orientation = s_orientation.normalise()

        # calculate rotation from fresnel polarisation frame to incident polarisation frame (inbound ray)
        theta = self._polarisation_frame_angle(i_direction, s_orientation, i_orientation)
        self._apply_stokes_rotation(spectrum, theta)
        return spectrum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _apply_mueller_reflection_tir(self, Spectrum spectrum, double ci, double gamma):

        cdef:
            double theta, c, s
            double s0, s1, s2, s3

        # phase shift between perpendicular and parallel reflected beam components
        theta = -2.0 * atan2(ci * sqrt(gamma*gamma * (1.0 - ci*ci) - 1.0), gamma * (1.0 - ci*ci))

        # apply retarder mueller matrix with phase shift
        c = cos(theta)
        s = sin(theta)
        for bin in range(spectrum.bins):

            s0 = spectrum.samples_mv[bin, 0]
            s1 = spectrum.samples_mv[bin, 1]
            s2 = spectrum.samples_mv[bin, 2]
            s3 = spectrum.samples_mv[bin, 3]

            spectrum.samples_mv[bin, 0] = s0
            spectrum.samples_mv[bin, 1] = s1
            spectrum.samples_mv[bin, 2] = c * s2 - s * s3
            spectrum.samples_mv[bin, 3] = s * s2 + c * s3

    @cython.cdivision(True)
    cdef (double, double, double, double, double) _fresnel(self, double ci, double ct, double ni, double nt) nogil:

        cdef double k0, k1, k2, k3, a, b, rp, rs, tp, ts, ta

        # calculation expects magnitude of cosines

        # common coefficients
        k0 = ni * ct
        k1 = nt * ci
        k2 = ni * ci
        k3 = nt * ct

        a = 1.0 / (k1 + k0)
        b = 1.0 / (k2 + k3)

        # reflection coefficients
        rp = a * (k0 - k1)
        rs = b * (k2 - k3)

        # transmission coefficients
        tp = 2.0 * a * k2
        ts = 2.0 * b * k2

        # projected area for transmitted beam
        ta = k3 / k2

        return rp, rs, tp, ts, ta

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _apply_mueller_reflection_non_tir(self, Spectrum spectrum, double p, double s):
        """
        Applies a normalised mueller matrix for reflection.
        
        A reflected or transmitted ray is selected by roulette with the
        probability determined by the transmission strength. To obtain an
        unbiased sample, the spectrum must be normalised by the probability of
        the selected path. This normalisation results in a simpler calculation
        of the mueller matrices due to term cancellation.
        """

        cdef:
            int bin
            double k0, k1, k2
            double s0, s1, s2, s3

        k0 = 1.0 / (p*p + s*s)
        k1 = k0 * (p*p - s*s)
        k2 = -2 * k0 * p * s

        for bin in range(spectrum.bins):

            s0 = spectrum.samples_mv[bin, 0]
            s1 = spectrum.samples_mv[bin, 1]
            s2 = spectrum.samples_mv[bin, 2]
            s3 = spectrum.samples_mv[bin, 3]

            spectrum.samples_mv[bin, 0] = s0 + k1 * s1
            spectrum.samples_mv[bin, 1] = k1 * s0 + s1
            spectrum.samples_mv[bin, 2] = k2 * s2
            spectrum.samples_mv[bin, 3] = k2 * s3

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _apply_mueller_transmission(self, Spectrum spectrum, double p, double s):
        """
        Applies a normalised mueller matrix for reflection.
        
        A reflected or transmitted ray is selected by roulette with the
        probability determined by the transmission strength. To obtain an
        unbiased sample, the spectrum must be normalised by the probability of
        the selected path. This normalisation results in a simpler calculation
        of the mueller matrices due to term cancellation.
        """

        cdef:
            int bin
            double k0, k1, k2
            double s0, s1, s2, s3

        k0 = 1.0 / (p*p + s*s)
        k1 = k0 * (p*p - s*s)
        k2 = 2 * k0 * p * s

        for bin in range(spectrum.bins):

            s0 = spectrum.samples_mv[bin, 0]
            s1 = spectrum.samples_mv[bin, 1]
            s2 = spectrum.samples_mv[bin, 2]
            s3 = spectrum.samples_mv[bin, 3]

            spectrum.samples_mv[bin, 0] = s0 + k1 * s1
            spectrum.samples_mv[bin, 1] = k1 * s0 + s1
            spectrum.samples_mv[bin, 2] = k2 * s2
            spectrum.samples_mv[bin, 3] = k2 * s3

    cdef double _polarisation_frame_angle(self, Vector3D direction, Vector3D ray_orientation, Vector3D interface_orientation):

        # light propagation direction is opposite to ray direction
        propagation = direction.neg()

        # calculate rotation about light propagation direction
        angle = ray_orientation.angle(interface_orientation) * DEG2RAD
        if propagation.dot(ray_orientation.cross(interface_orientation)) < 0:
            angle = -angle
        return angle

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _apply_stokes_rotation(self, Spectrum spectrum, double theta):

        cdef:
            double c, s
            double s0, s1, s2, s3

        c = cos(2*theta)
        s = sin(2*theta)
        for bin in range(spectrum.bins):

            s0 = spectrum.samples_mv[bin, 0]
            s1 = spectrum.samples_mv[bin, 1]
            s2 = spectrum.samples_mv[bin, 2]
            s3 = spectrum.samples_mv[bin, 3]

            spectrum.samples_mv[bin, 0] = s0
            spectrum.samples_mv[bin, 1] = c * s1 - s * s2
            spectrum.samples_mv[bin, 2] = s * s1 + c * s2
            spectrum.samples_mv[bin, 3] = s3

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
