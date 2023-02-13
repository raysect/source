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

from libc.math cimport fabs
from raysect.optical cimport World, Ray, Primitive, Point3D, AffineMatrix3D, Vector3D, Normal3D, Intersection
cimport cython


cdef class AnisotropicSurfaceEmitter(NullVolume):
    """
    Base class for anisotropic surface emitters.

    Simplifies the development of anisotropic light sources. The emitter
    spectrum can be varied based on the angle between the normal and
    incident observation direction, and the side of the surface.
    """

    def __init__(self):

        super().__init__()
        self.importance = 1.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef:
            Vector3D incident
            Spectrum spectrum
            double cosine, angle
            bint back_face

        # convert ray direction normal to local coordinates
        incident = ray.direction.transform(world_to_primitive)

        # ensure vectors are normalised for reflection calculation
        incident = incident.normalise()
        normal = normal.normalise()

        # calculate polar angle
        cosine = normal.dot(incident)
        back_face = cosine < 0
        cosine = fabs(cosine)

        # obtain emission spectrum
        spectrum = self.emission_function(ray.new_spectrum(), cosine, back_face)

        # check spectrum object
        if spectrum.samples.ndim != 1 or spectrum.samples.shape[0] != ray.get_bins():
            raise ValueError("Spectrum returned by emission function has the wrong number of samples.")

        return spectrum

    cpdef Spectrum emission_function(self, Spectrum spectrum, double cosine, bint back_face):
        """
        Returns the emission along the observation direction.

        This is a virtual method and must be implemented by sub-classing. The emission is modulated
        by the angle between the observation direction and the surface normal. To this end, the
        cosine of the angle between the normal and observation directions is supplied. This value
        is always in the range [0, 1], no matter which side the ray intersects.

        If the emission must vary according to the side of the surface the ray intersects, this may
        be determined by inspecting the back_face argument. This will be True if the inside (or back)
        surface is struck by the ray, otherwise it is False.

        :param Spectrum spectrum: The Spectrum object in which to place the observed emission.
        :param float cosine: The cosine of the angle between the normal and the observation
        direction.
        :param bool back_face: True if the back face of the surface (inside), False otherwise.
        :return: The spectrum object.
        """

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")


