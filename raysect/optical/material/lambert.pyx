# cython: language_level=3

# Copyright (c) 2015, Dr Alex Meakins, Raysect Project
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

from raysect.core.math.affinematrix cimport AffineMatrix3D
from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.scenegraph.world cimport World
from raysect.optical.ray cimport Ray
from raysect.core.math.point cimport Point3D
from raysect.core.math.vector cimport Vector3D
from raysect.optical.spectrum cimport Spectrum
from raysect.core.math.normal cimport Normal3D
from raysect.optical.spectralfunction cimport SpectralFunction, ConstantSF
from raysect.core.math.random cimport vector_hemisphere_cosine
from raysect.core.math.cython cimport transform
from numpy cimport ndarray

cdef class Lambert(NullVolume):

    def __init__(self, SpectralFunction reflectivity=None):

        if reflectivity is None:
            reflectivity = ConstantSF(0.5)

        self.reflectivity = reflectivity

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_local, AffineMatrix3D local_to_world):

        cdef:
            Ray reflected
            Vector3D v_normal, v_tangent, direction
            AffineMatrix3D surface_to_local
            Spectrum spectrum
            ndarray reflectivity

        # generate transform from surface to local space
        surface_to_local = transform.surface_to_local(normal.as_vector(), normal.orthogonal())

        # obtain new world space ray vector from cosine-weighted hemisphere
        direction = vector_hemisphere_cosine()
        direction = direction.transform(surface_to_local)

        # generate and trace ray
        if exiting:
            reflected = ray.spawn_daughter(inside_point.transform(local_to_world), direction.neg().transform(local_to_world))
        else:
            reflected = ray.spawn_daughter(outside_point.transform(local_to_world), direction.transform(local_to_world))

        spectrum = reflected.trace(world)

        # obtain samples of reflectivity
        reflectivity = self.reflectivity.sample_multiple(spectrum.min_wavelength,
                                                         spectrum.max_wavelength,
                                                         spectrum.num_samples)
        spectrum.mul_array(reflectivity)

        return spectrum







