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
from raysect.core.math.point cimport new_point

cdef class VolumeEmitterHomogeneous(NullSurface):

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point start_point, Point end_point,
                                   AffineMatrix to_local, AffineMatrix to_world):

        cdef:
            Point start, end
            Vector direction
            double length
            Spectrum emission

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
        emission = self.emission_function(direction, ray.wavebands)

        # integrate emission density along ray path
        emission = emission.mul(length)

        return spectrum.add(emission)

    cpdef Spectrum emission_function(self, Vector direction, tuple wavebands):

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")


cdef class VolumeEmitterInhomogeneous(NullSurface):

    def __init__(self, double step = 0.01):

        self.step = step

    property step:

        def __get__(self):

            return self._step

        def __set__(self, double step):

            if step <= 0:

                raise ValueError("Numerical integration step size can not be less than or equal to zero")

            self._step = step

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point start_point, Point end_point,
                                   AffineMatrix to_local, AffineMatrix to_world):

        cdef:
            Point start, end
            Vector direction
            double length, t, c
            Spectrum emission, emission_previous

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

        # numerical integration
        emission_previous = self.emission_function(start, direction, spectrum.wavebands)
        t = self._step
        c = 0.5 * self._step
        while(t <= length):

            sample_point = new_point(start.x + t * direction.x,
                                     start.y + t * direction.y,
                                     start.z + t * direction.z)

            emission = self.emission_function(sample_point, direction, spectrum.wavebands)

            # trapizium rule integration
            spectrum = spectrum.add(emission.add(emission_previous).mul(c))

            emission_previous = emission
            t += self._step

        # trapizium rule integration of remainder
        t -= self._step
        emission = self.emission_function(end, direction, spectrum.wavebands)
        spectrum = spectrum.add(emission.add(emission_previous).mul(0.5 * (length - t)))

        return spectrum

    cpdef Spectrum emission_function(self, Point point, Vector direction, tuple wavebands):

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")


