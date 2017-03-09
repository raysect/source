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

from raysect.optical.colour import d65_white

from numpy cimport ndarray
from libc.math cimport round
from raysect.optical cimport new_point3d, Normal3D, new_spectrum
cimport cython


cdef class UniformSurfaceEmitter(NullVolume):

    def __init__(self, SpectralFunction emission_spectrum, double scale = 1.0):
        """
        Uniform and isotropic surface emitter

        emission is spectral radiance: W/m2/str/nm"""

        super().__init__()
        self.emission_spectrum = emission_spectrum
        self.scale = scale
        self.importance = 1.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Spectrum spectrum
            ndarray emission
            double[::1] s_view, e_view
            int index

        spectrum = ray.new_spectrum()
        emission = self.emission_spectrum.sample(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)

        # obtain memoryviews
        s_view = spectrum.samples
        e_view = emission

        for index in range(spectrum.bins):
            s_view[index] = e_view[index] * self.scale

        return spectrum

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # no volume contribution
        return spectrum


cdef class VolumeEmitterHomogeneous(NullSurface):

    def __init__(self):
        super().__init__()
        self.importance = 1.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D direction
            double length
            Spectrum emission
            double[::1] e_view, s_view
            int index

        # convert start and end points to local space
        start = start_point.transform(world_to_primitive)
        end = end_point.transform(world_to_primitive)

        # obtain local space ray direction (travels end->start) and integration length
        direction = end.vector_to(start)
        length = direction.get_length()

        # nothing to contribute?
        if length == 0:
            return spectrum

        direction = direction.normalise()

        # obtain emission density from emission function (W/m^3/str)
        emission = new_spectrum(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)

        # emission function specifies direction from ray origin to hit-point
        emission = self.emission_function(direction, emission, world, ray, primitive, world_to_primitive, primitive_to_world)

        # sanity check as bounds checking is disabled
        if emission.samples.ndim != 1 or spectrum.samples.ndim != 1 or emission.samples.shape[0] != spectrum.samples.shape[0]:
            raise ValueError("Spectrum returned by emission function has the wrong number of bins.")

        # memoryviews used for fast element access
        e_view = emission.samples
        s_view = spectrum.samples

        # integrate emission density along ray path
        for index in range(spectrum.bins):
            s_view[index] += e_view[index] * length

        return spectrum

    cpdef Spectrum emission_function(self, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")


cdef class VolumeEmitterInhomogeneous(NullSurface):

    def __init__(self, double step = 0.01):

        super().__init__()
        self._step = step
        self.importance = 1.0

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
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D integration_direction, ray_direction
            double length, t, c
            Spectrum emission, emission_previous
            double[::1] e1_view, e2_view, s_view
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
        ray_direction = -integration_direction

        emission_previous = new_spectrum(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
        emission_previous = self.emission_function(start, ray_direction, emission_previous, world, ray, primitive, world_to_primitive, primitive_to_world)

        # sanity check as bounds checking is disabled
        if emission_previous.samples.ndim != 1 or spectrum.samples.ndim != 1 or emission_previous.samples.shape[0] != spectrum.samples.shape[0]:
            raise ValueError("Spectrum returned by emission function has the wrong number of samples.")

        # assign memoryview for fast element access to output spectrum
        s_view = spectrum.samples

        # numerical integration
        t = self._step
        c = 0.5 * self._step
        while t <= length:

            sample_point = new_point3d(
                start.x + t * integration_direction.x,
                start.y + t * integration_direction.y,
                start.z + t * integration_direction.z
            )

            emission = new_spectrum(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
            emission = self.emission_function(sample_point, ray_direction, emission, world, ray, primitive, world_to_primitive, primitive_to_world)

            # sanity check as bounds checking is disabled
            if emission.samples.ndim != 1 or spectrum.samples.ndim != 1 or emission.samples.shape[0] != spectrum.samples.shape[0]:
                raise ValueError("Spectrum returned by emission function has the wrong number of samples.")

            # memoryviews used for fast element access
            e1_view = emission.samples
            e2_view = emission_previous.samples

            # trapezium rule integration
            for index in range(spectrum.bins):
                s_view[index] += c * (e1_view[index] + e2_view[index])

            emission_previous = emission
            t += self._step

        # step back to process any length that remains
        t -= self._step

        emission = new_spectrum(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
        emission = self.emission_function(end, ray_direction, emission, world, ray, primitive, world_to_primitive, primitive_to_world)

        # sanity check as bounds checking is disabled
        if emission.samples.ndim != 1 or spectrum.samples.ndim != 1 or emission.samples.shape[0] != spectrum.samples.shape[0]:
            raise ValueError("Spectrum returned by emission function has the wrong number of samples.")

        # memoryviews used for fast element access
        e1_view = emission.samples
        e2_view = emission_previous.samples

        # trapezium rule integration of remainder
        c = 0.5 * (length - t)
        for index in range(spectrum.bins):
            s_view[index] += c * (e1_view[index] + e2_view[index])

        return spectrum

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")


cdef class UnityVolumeEmitter(VolumeEmitterHomogeneous):

    def __init__(self):
        """
        Uniform, isotropic volume emitter with emission 1W/str/m^3/ x nm, where x is the spectrum's wavelength interval.

        This material is useful for general purpose debugging and evaluating the coupling coefficients between cameras
        and emitting volumes.
        """

        super().__init__()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum emission_function(self, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):
        spectrum.samples[:] = 1.0
        return spectrum


cdef class UniformVolumeEmitter(VolumeEmitterHomogeneous):

    def __init__(self, SpectralFunction emission_spectrum, double scale=1.0):
        """
        Uniform, homogeneous and isotropic volume emitter

        emission is spectral volume radiance: W/m^3/str/nm ie spectral radiance per meter"""

        super().__init__()
        self.emission_spectrum = emission_spectrum
        self.scale = scale

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum emission_function(self, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            ndarray emission
            double[::1] s_view, e_view
            int index

        emission = self.emission_spectrum.sample(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)

        # obtain memoryviews
        s_view = spectrum.samples
        e_view = emission

        for index in range(spectrum.bins):
            s_view[index] += e_view[index] * self.scale

        return spectrum


cdef class Checkerboard(NullVolume):

    def __init__(self, double width=1.0, SpectralFunction emission_spectrum1=d65_white, SpectralFunction emission_spectrum2=d65_white, double scale1=0.25, double scale2=0.5):
        """
        Isotropic checkerboard surface emitter

        emission1 and emission2 is spectral radiance: W/m2/str/nm
        scale in meters
        """

        super().__init__()
        self._width = width
        self._rwidth = 1.0 / width
        self.emission_spectrum1 = emission_spectrum1
        self.emission_spectrum2 = emission_spectrum2
        self.scale1 = scale1
        self.scale2 = scale2
        self.importance = 1.0

    property width:

        def __get__(self):
            return self._width

        @cython.cdivision(True)
        def __set__(self, double v):
            self._width = v
            self._rwidth = 1.0 / v

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                bint exiting, Point3D inside_point, Point3D outside_point,
                                Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

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
            emission = self.emission_spectrum1.sample(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
            e_view = emission
            scale = self.scale1
        else:
            emission = self.emission_spectrum2.sample(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
            e_view = emission
            scale = self.scale2

        for index in range(spectrum.bins):
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
