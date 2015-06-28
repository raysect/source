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

cimport cython
from numpy import array, float64
from numpy cimport ndarray
from libc.math cimport sqrt, pow as cpow
from raysect.core.math.affinematrix cimport AffineMatrix
from raysect.core.math.point cimport Point
from raysect.core.math.vector cimport Vector, new_vector
from raysect.core.math.normal cimport Normal
from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.scenegraph.world cimport World
from raysect.optical.spectralfunction cimport ConstantSF
from raysect.optical.spectrum cimport Spectrum
from raysect.optical.ray cimport Ray


cdef class Sellmeier(SpectralFunction):

    def __init__(self, double b1, double b2, double b3, double c1, double c2, double c3, int subsamples=10):

        if subsamples < 1:

            raise ValueError("The number of sub-samples cannot be less than 1.")

        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        self.subsamples = subsamples

        # initialise cache to invalid values
        self.cached_min_wavelength = -1
        self.cached_max_wavelength = -1
        self.cached_index = -1

    @cython.cdivision(True)
    cpdef double sample_single(self, double min_wavelength, double max_wavelength):
        """
        Generates a single sample of the refractive index given by the
        Sellmeier equation.

        The refractive index returned is the average over the wavelength range.
        The number of sub-samples used for the average calculation can be
        configured by modifying the subsamples attribute. If the subsamples
        attribute is set to 1, a single sample is taken from the centre of the
        wavelength range.

        :param min_wavelength: Minimum wavelength in nm.
        :param max_wavelength: Maximum wavelength in nm.
        :return: A refractive index sample.
        """

        cdef:
            double index, delta_wavelength, centre_wavelength, reciprocal
            int i

        if self.cached_min_wavelength == min_wavelength and \
             self.cached_max_wavelength == max_wavelength:

            return self.cached_index

        # sample the refractive index
        index = 0.0
        delta_wavelength = (max_wavelength - min_wavelength) / self.subsamples
        reciprocal =  1.0 / self.subsamples
        for i in range(self.subsamples):

            centre_wavelength = (min_wavelength + (0.5 + i) * delta_wavelength)

            # Sellmeier coefficients are specified for wavelength in micrometers
            index += reciprocal * self._sellmeier(centre_wavelength * 1e-3)

        # update cache
        self.cached_min_wavelength = min_wavelength
        self.cached_max_wavelength = max_wavelength
        self.cached_index = index

        return index

    @cython.cdivision(True)
    cdef inline double _sellmeier(self, double wavelength) nogil:
        """
        Returns a sample of the three term Sellmeier equation at the specified
        wavelength.

        :param wavelength: Wavelength in um.
        :return: Refractive index sample.
        """

        cdef double w2 = wavelength * wavelength
        return sqrt(1 + (self.b1 * w2) / (w2 - self.c1)
                      + (self.b2 * w2) / (w2 - self.c2)
                      + (self.b3 * w2) / (w2 - self.c3))


# TODO: consider carefully the impact of changes made to support mesh normal interpolation
cdef class Dielectric(Material):

    def __init__(self, SpectralFunction index, SpectralFunction transmission, cutoff = 1e-6, SpectralFunction external_index=ConstantSF(1.0)):

        self.index = index
        self.external_index = external_index
        self.transmission = transmission
        self.cutoff = cutoff

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point hit_point,
                                    bint exiting, Point inside_point, Point outside_point,
                                    Normal normal, AffineMatrix to_local, AffineMatrix to_world):

        cdef:
            Vector incident, reflected, transmitted
            double internal_index, external_index, n1, n2
            double c1, c2s, gamma, reflectivity, transmission, temp
            Spectrum spectrum, ray_sample

        # convert ray direction normal to local coordinates
        incident = ray.direction.transform(to_local)

        # ensure vectors are normalised for reflection calculation
        incident = incident.normalise()
        normal = normal.normalise()

        # calculate cosine of angle between incident and normal
        c1 = -normal.dot(incident)

        # sample refractive indices
        internal_index = self.index.sample_single(ray.get_min_wavelength(), ray.get_max_wavelength())
        external_index = self.external_index.sample_single(ray.get_min_wavelength(), ray.get_max_wavelength())

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

        with cython.cdivision:

            gamma = n1 / n2

        # calculate square of cosine of angle between transmitted ray and normal
        c2s = 1 - (gamma * gamma) * (1 - c1 * c1)

        # check for total internal reflection
        if c2s <= 0:

            # total internal reflection
            temp = 2 * c1
            reflected = new_vector(incident.x + temp * normal.x,
                                   incident.y + temp * normal.y,
                                   incident.z + temp * normal.z)

            # convert reflected ray direction to world space
            reflected = reflected.transform(to_world)

            # spawn reflected ray and trace
            # note, we do not use the supplied exiting parameter as the normal is
            # not guaranteed to be perpendicular to the surface for meshes
            if c1 < 0.0:

                # incident ray is pointing out of surface, reflection is therefore inside
                reflected_ray = ray.spawn_daughter(inside_point.transform(to_world), reflected)

            else:

                # incident ray is pointing in to surface, reflection is therefore outside
                reflected_ray = ray.spawn_daughter(outside_point.transform(to_world), reflected)

            return reflected_ray.trace(world)

        else:

            # calculate reflected and transmitted ray normals
            temp = 2 * c1
            reflected = new_vector(incident.x + temp * normal.x,
                                   incident.y + temp * normal.y,
                                   incident.z + temp * normal.z)

            # note, we do not use the supplied exiting parameter as the normal is
            # not guaranteed to be perpendicular to the surface for meshes
            if c1 < 0.0:

                temp = gamma * c1 + sqrt(c2s)
                transmitted = new_vector(gamma * incident.x + temp * normal.x,
                                         gamma * incident.y + temp * normal.y,
                                         gamma * incident.z + temp * normal.z)

            else:

                temp = gamma * c1 - sqrt(c2s)
                transmitted = new_vector(gamma * incident.x + temp * normal.x,
                                         gamma * incident.y + temp * normal.y,
                                         gamma * incident.z + temp * normal.z)

            # calculate fresnel reflection and transmission coefficients
            self._fresnel(c1, -normal.dot(transmitted), n1, n2, &reflectivity, &transmission)

            # convert reflected and transmitted rays to world space
            reflected = reflected.transform(to_world)
            transmitted = transmitted.transform(to_world)

            # convert origin points to world space
            inside_point = inside_point.transform(to_world)
            outside_point = outside_point.transform(to_world)

            # spawn reflected and transmitted rays
            if exiting:

                # incident ray is pointing out of surface
                reflected_ray = ray.spawn_daughter(inside_point, reflected)
                transmitted_ray = ray.spawn_daughter(outside_point, transmitted)

            else:

                # incident ray is pointing in to surface
                reflected_ray = ray.spawn_daughter(outside_point, reflected)
                transmitted_ray = ray.spawn_daughter(inside_point, transmitted)

            # trace rays and return results
            if reflectivity > self.cutoff:

                spectrum = reflected_ray.trace(world)
                spectrum.mul_scalar(reflectivity)

            else:

                spectrum = ray.new_spectrum()

            if transmission > self.cutoff:

                ray_sample = transmitted_ray.trace(world)
                ray_sample.mul_scalar(transmission)
                spectrum.add_array(ray_sample.samples)

            return spectrum

    @cython.cdivision(True)
    cdef inline void _fresnel(self, double ci, double ct, double n1, double n2, double *reflectivity, double *transmission) nogil:

        reflectivity[0] = 0.5 * (((n1*ci - n2*ct) / (n1*ci + n2*ct))**2 + ((n1*ct - n2*ci) / (n1*ct + n2*ci))**2)
        transmission[0] = 1 - reflectivity[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point start_point, Point end_point,
                                   AffineMatrix to_local, AffineMatrix to_world):

        cdef:
            double length
            ndarray transmission
            double[::1] s_view, t_view
            int index

        length = start_point.vector_to(end_point).get_length()

        transmission = self.transmission.sample_multiple(spectrum.min_wavelength,
                                                         spectrum.max_wavelength,
                                                         spectrum.num_samples)

        s_view = spectrum.samples
        t_view = transmission

        for index in range(spectrum.num_samples):

            s_view[index] *= cpow(t_view[index], length)

        return spectrum


