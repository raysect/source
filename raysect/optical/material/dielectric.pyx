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

cimport cython
from numpy import array, float64
from numpy cimport ndarray
from libc.math cimport sqrt, pow as cpow
from raysect.core.math.random cimport probability
from raysect.optical cimport Point3D, Vector3D, new_vector3d, Normal3D, AffineMatrix3D, World, Primitive, ConstantSF, Spectrum, Ray, Intersection


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

    def __init__(self, double b1, double b2, double b3, double c1, double c2, double c3, double sample_resolution=10):
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

    def __init__(self, SpectralFunction index, SpectralFunction transmission, SpectralFunction external_index=None, bint transmission_only=False):
        super().__init__()
        self.index = index
        self.transmission = transmission
        self.transmission_only = transmission_only

        if external_index is None:
            self.external_index = ConstantSF(1.0)
        else:
            self.external_index = external_index

        self.importance = 1.0

    @cython.cdivision(True)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef:
            Vector3D incident, reflected, transmitted
            double internal_index, external_index, n1, n2
            double c1, c2s, gamma, reflectivity, transmission, temp
            Ray reflected_ray, transmitted_ray
            Spectrum spectrum

        # convert ray direction normal to local coordinates
        incident = ray.direction.transform(world_to_primitive)

        # ensure vectors are normalised for reflection calculation
        incident = incident.normalise()
        normal = normal.normalise()

        # calculate cosine of angle between incident and normal
        c1 = -normal.dot(incident)

        # sample refractive indices
        internal_index = self.index.average(ray.get_min_wavelength(), ray.get_max_wavelength())
        external_index = self.external_index.average(ray.get_min_wavelength(), ray.get_max_wavelength())

        # are we entering or leaving material - calculate refractive change
        # note, we do not use the supplied exiting parameter as the normal is
        # not guaranteed to be perpendicular to the surface for meshes
        if c1 < 0.0:

            # leaving material
            n1 = internal_index
            n2 = external_index

        else:

            # entering material
            n1 = external_index
            n2 = internal_index

        gamma = n1 / n2

        # calculate square of cosine of angle between transmitted ray and normal
        c2s = 1 - (gamma * gamma) * (1 - c1 * c1)

        # check for total internal reflection
        if c2s <= 0:

            # skip calculation if transmission only enabled
            if self.transmission_only:
                return ray.new_spectrum()

            # total internal reflection
            temp = 2 * c1
            reflected = new_vector3d(incident.x + temp * normal.x,
                                     incident.y + temp * normal.y,
                                     incident.z + temp * normal.z)

            # convert reflected ray direction to world space
            reflected = reflected.transform(primitive_to_world)

            # spawn reflected ray and trace
            # note, we do not use the supplied exiting parameter as the normal is
            # not guaranteed to be perpendicular to the surface for meshes
            if c1 < 0.0:

                # incident ray is pointing out of surface, reflection is therefore inside
                reflected_ray = ray.spawn_daughter(inside_point.transform(primitive_to_world), reflected)

            else:

                # incident ray is pointing in to surface, reflection is therefore outside
                reflected_ray = ray.spawn_daughter(outside_point.transform(primitive_to_world), reflected)

            return reflected_ray.trace(world)

        else:

            # calculate transmitted ray normal
            # note, we do not use the supplied exiting parameter as the normal is
            # not guaranteed to be perpendicular to the surface for meshes
            if c1 < 0.0:
                temp = gamma * c1 + sqrt(c2s)
            else:
                temp = gamma * c1 - sqrt(c2s)

            transmitted = new_vector3d(gamma * incident.x + temp * normal.x,
                                       gamma * incident.y + temp * normal.y,
                                       gamma * incident.z + temp * normal.z)

            # calculate fresnel reflection and transmission coefficients
            self._fresnel(c1, -normal.dot(transmitted), n1, n2, &reflectivity, &transmission)

            # select path by roulette using the strength of the coefficients as probabilities
            if self.transmission_only or probability(transmission):

                # transmitted ray path selected

                # we have already calculated the transmitted normal
                transmitted = transmitted.transform(primitive_to_world)

                # spawn ray on correct side of surface
                # note, we do not use the supplied exiting parameter as the normal is
                # not guaranteed to be perpendicular to the surface for meshes
                if c1 < 0.0:

                    # incident ray is pointing out of surface
                    outside_point = outside_point.transform(primitive_to_world)
                    transmitted_ray = ray.spawn_daughter(outside_point, transmitted)

                else:

                    # incident ray is pointing in to surface
                    inside_point = inside_point.transform(primitive_to_world)
                    transmitted_ray = ray.spawn_daughter(inside_point, transmitted)

                spectrum = transmitted_ray.trace(world)

            else:

                # reflected ray path selected

                # calculate ray normal
                temp = 2 * c1
                reflected = new_vector3d(incident.x + temp * normal.x,
                                         incident.y + temp * normal.y,
                                         incident.z + temp * normal.z)
                reflected = reflected.transform(primitive_to_world)

                # spawn ray on correct side of surface
                # note, we do not use the supplied exiting parameter as the normal is
                # not guaranteed to be perpendicular to the surface for meshes
                if c1 < 0.0:

                    # incident ray is pointing out of surface
                    inside_point = inside_point.transform(primitive_to_world)
                    reflected_ray = ray.spawn_daughter(inside_point, reflected)

                else:

                    # incident ray is pointing in to surface
                    outside_point = outside_point.transform(primitive_to_world)
                    reflected_ray = ray.spawn_daughter(outside_point, reflected)

                spectrum = reflected_ray.trace(world)

            # note, normalisation not required as path probability equals the reflection/transmission coefficient
            # the two values cancel exactly
            return spectrum

    @cython.cdivision(True)
    cdef void _fresnel(self, double ci, double ct, double n1, double n2, double *reflectivity, double *transmission) nogil:

        reflectivity[0] = 0.5 * (((n1*ci - n2*ct) / (n1*ci + n2*ct))**2 + ((n1*ct - n2*ci) / (n1*ct + n2*ci))**2)
        transmission[0] = 1 - reflectivity[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            double length
            double[::1] transmission
            int index

        length = start_point.vector_to(end_point).get_length()
        transmission = self.transmission.sample_mv(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
        for index in range(spectrum.bins):
            spectrum.samples_mv[index] *= cpow(transmission[index], length)

        return spectrum


