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

cdef class SurfaceEmitter(NullVolume):

    #TODO: implement
    pass


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
            Spectrum emission_density
            double[::1] spectrum_view, emission_view

        # convert entry and exit to local space
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
        emission_density = self.emission_function(direction, ray.wavebands)

        # sanity check as bound checking has been disabled
        if (spectrum.bins.ndim != 1 or emission_density.bins.ndim != 1
           or emission_density.bins.shape[0] != spectrum.bins.shape[0]):

            raise ValueError("The spectrum returned by evaluate_volume has the wrong number of bins.")

        # integrate emission density along ray path
        spectrum_view = spectrum.bins
        emission_view = emission_density.bins

        for index in range(spectrum.bins.shape[0]):

            spectrum_view[index] += emission_view[index] * length

        return spectrum

    cpdef Spectrum emission_function(self, Vector direction, tuple wavebands):

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")


cdef class VolumeEmitterInhomogeneous(NullSurface):

    # TODO: Implement
    pass



