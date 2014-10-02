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
from numpy cimport ndarray
from libc.math cimport round
from raysect.core.math.affinematrix cimport AffineMatrix
from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.scenegraph.world cimport World
from raysect.optical.ray cimport Ray
from raysect.core.math.normal cimport Normal
from raysect.core.math.point cimport new_point
from raysect.optical.spectrum cimport new_spectrum
from raysect.optical.spectralfunction cimport ConstantSF
from raysect.optical.colour import d65_white

cdef class VolumeEmitterHomogeneous(NullSurface):

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point start_point, Point end_point,
                                   AffineMatrix to_local, AffineMatrix to_world):

        cdef:
            Point start, end
            Vector direction
            double length
            Spectrum emission
            double[::1] e_view, s_view
            int index

        # convert start and end points to local space
        start = start_point.transform(to_local)
        end = end_point.transform(to_local)

        # obtain local space ray direction and integration length
        direction = start.vector_to(end)
        length = direction.get_length()

        if length == 0:

            # nothing to contribute
            return spectrum

        direction = direction.normalise()

        # obtain emission density from emission function (W/m^3/str)
        emission = new_spectrum(spectrum.min_wavelength,
                                spectrum.max_wavelength,
                                spectrum.samples)

        emission = self.emission_function(direction, emission)

        # sanity check as bounds checking is disabled
        if (emission.samples.ndim != 1 or spectrum.samples.ndim != 1
            or emission.samples.shape[0] != spectrum.samples.shape[0]):

            raise ValueError("Spectrum returned by emission function has the wrong number of bins.")

        # memoryviews used for fast element access
        e_view = emission.samples
        s_view = spectrum.samples

        # integrate emission density along ray path
        for index in range(spectrum.samples.shape[0]):

            s_view[index] += e_view[index] * length

        return spectrum

    cpdef Spectrum emission_function(self, Vector direction, Spectrum spectrum):

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")


cdef class VolumeEmitterInhomogeneous(NullSurface):

    def __init__(self, double step = 0.01):

        self._step = step

    property step:

        def __get__(self):

            return self._step

        def __set__(self, double step):

            if step <= 0:

                raise ValueError("Numerical integration step size can not be less than or equal to zero")

            self._step = step

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point start_point, Point end_point,
                                   AffineMatrix to_local, AffineMatrix to_world):

        cdef:
            Point start, end
            Vector direction
            double length, t, c
            Spectrum emission, emission_previous
            double[::1] e1_view, e2_view, s_view
            int index

        # convert start and end points to local space
        start = start_point.transform(to_local)
        end = end_point.transform(to_local)

        # obtain local space ray direction and integration length
        direction = start.vector_to(end)
        length = direction.get_length()

        if length == 0:

            # nothing to contribute
            return spectrum

        direction = direction.normalise()

        emission_previous = new_spectrum(spectrum.min_wavelength,
                                        spectrum.max_wavelength,
                                        spectrum.samples)

        emission_previous = self.emission_function(start, direction, emission_previous)

        # sanity check as bounds checking is disabled
        if (emission_previous.samples.ndim != 1 or spectrum.samples.ndim != 1
            or emission_previous.samples.shape[0] != spectrum.samples.shape[0]):

            raise ValueError("Spectrum returned by emission function has the wrong number of samples.")

        # assign memoryview for fast element access to output spectrum
        s_view = spectrum.samples

        # numerical integration
        t = self._step
        c = 0.5 * self._step
        while t <= length:

            sample_point = new_point(start.x + t * direction.x,
                                     start.y + t * direction.y,
                                     start.z + t * direction.z)

            emission = new_spectrum(spectrum.min_wavelength,
                                    spectrum.max_wavelength,
                                    spectrum.samples)

            emission = self.emission_function(sample_point, direction, emission)

            # sanity check as bounds checking is disabled
            if (emission.samples.ndim != 1 or spectrum.samples.ndim != 1
                or emission.samples.shape[0] != spectrum.samples.shape[0]):

                raise ValueError("Spectrum returned by emission function has the wrong number of samples.")

            # memoryviews used for fast element access
            e1_view = emission.samples
            e2_view = emission_previous.samples

            # trapezium rule integration
            for index in range(spectrum.samples.shape[0]):

                s_view[index] += c * (e1_view[index] + e2_view[index])

            emission_previous = emission
            t += self._step

        # step back to process any length that remains
        t -= self._step

        emission = new_spectrum(spectrum.min_wavelength,
                                spectrum.max_wavelength,
                                spectrum.samples)

        emission = self.emission_function(end, direction, emission)

        # sanity check as bounds checking is disabled
        if (emission.samples.ndim != 1 or spectrum.samples.ndim != 1
            or emission.samples.shape[0] != spectrum.samples.shape[0]):

            raise ValueError("Spectrum returned by emission function has the wrong number of samples.")

        # memoryviews used for fast element access
        e1_view = emission.samples
        e2_view = emission_previous.samples

        # trapezium rule integration of remainder
        c = 0.5 * (length - t)
        for index in range(spectrum.samples.shape[0]):

            s_view[index] += c * (e1_view[index] + e2_view[index])

        return spectrum

    cpdef Spectrum emission_function(self, Point point, Vector direction, Spectrum spectrum):

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")


cdef class UniformSurfaceEmitter(NullVolume):

    def __init__(self, SpectralFunction emission_spectrum, double scale = 1.0):
        """
        Uniform and isotropic surface emitter

        emission is spectral radiance: W/m2/str/nm"""

        self.emission_spectrum = emission_spectrum
        self.scale = scale

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point hit_point,
                                bint exiting, Point inside_point, Point outside_point,
                                Normal normal, AffineMatrix to_local, AffineMatrix to_world):

        cdef:
            Spectrum spectrum
            ndarray emission
            double[::1] s_view, e_view
            int index

        spectrum = ray.new_spectrum()

        emission = self.emission_spectrum.sample_multiple(spectrum.min_wavelength,
                                                          spectrum.max_wavelength,
                                                          spectrum.num_samples)

        # obtain memoryviews
        s_view = spectrum.samples
        e_view = emission

        for index in range(spectrum.num_samples):

            s_view[index] += e_view[index] * self.scale

        return spectrum


cdef class UniformVolumeEmitter(VolumeEmitterHomogeneous):

    def __init__(self, SpectralFunction emission_spectrum, double scale=1.0):
        """
        Uniform, homogeneous and isotropic volume emitter

        emission is spectral volume radiance: W/m3/str/nm ie spectral radiance per meter"""

        self.emission_spectrum = emission_spectrum
        self.scale = scale

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum emission_function(self, Vector direction, Spectrum spectrum):

        cdef:
            ndarray emission
            double[::1] s_view, e_view
            int index

        emission = self.emission_spectrum.sample_multiple(spectrum.min_wavelength,
                                                          spectrum.max_wavelength,
                                                          spectrum.num_samples)

        # obtain memoryviews
        s_view = spectrum.samples
        e_view = emission

        for index in range(spectrum.num_samples):

            s_view[index] += e_view[index] * self.scale

        return spectrum


cdef class Checkerboard(NullVolume):

    def __init__(self, double width=1.0, SpectralFunction emission_spectrum1=d65_white, SpectralFunction emission_spectrum2=d65_white, double scale1=0.25, double scale2=0.5):
        """
        Isotropic checkerboard surface emitter

        emission1 and emission2 is spectral radiance: W/m2/str/nm
        scale in meters
        """

        self._width = width
        self._rwidth = 1.0 / width
        self.emission_spectrum1 = emission_spectrum1
        self.emission_spectrum2 = emission_spectrum2
        self.scale1 = scale1
        self.scale2 = scale2

    property width:

        def __get__(self):

            return self._width

        @cython.cdivision(True)
        def __set__(self, double v):

            self._width = v
            self._rwidth = 1.0 / v

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point hit_point,
                                bint exiting, Point inside_point, Point outside_point,
                                Normal normal, AffineMatrix to_local, AffineMatrix to_world):

        cdef:
            Spectrum spectrum
            ndarray emission
            double[::1] s_view, e_view
            bint v
            int index
            double scale

        v = False

        # generate check pattern
        v = self._flip(v, hit_point.x)
        v = self._flip(v, hit_point.y)
        v = self._flip(v, hit_point.z)

        # select emission
        spectrum = ray.new_spectrum()
        s_view = spectrum.samples

        if v:

            emission = self.emission_spectrum1.sample_multiple(spectrum.min_wavelength,
                                                               spectrum.max_wavelength,
                                                               spectrum.num_samples)
            e_view = emission
            scale = self.scale1

        else:

            emission = self.emission_spectrum2.sample_multiple(spectrum.min_wavelength,
                                                               spectrum.max_wavelength,
                                                               spectrum.num_samples)
            e_view = emission
            scale = self.scale2

        for index in range(spectrum.num_samples):

            s_view[index] = e_view[index] * scale

        return spectrum

    @cython.cdivision(True)
    cdef inline bint _flip(self, bint v, double p):

        # round to avoid numerical precision issues (rounds to nearest nanometer)
        p = round(p * 1e9) / 1e9

        # generates check pattern from [0, inf]
        if abs(self._rwidth * p) % 2 >= 1.0:

            v = not v

        # invert pattern for negative
        if p < 0:

            v = not v

        return v