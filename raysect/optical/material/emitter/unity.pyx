# cython: language_level=3

# Copyright (c) 2014-2025, Dr Alex Meakins, Raysect Project
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

from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, Normal3D, AffineMatrix3D, Intersection

cimport cython


cdef class UnitySurfaceEmitter(NullVolume):
    """
    Uniform and isotropic surface emitter with emission 1W/str/m^2/ x nm,
    where x is the spectrum's wavelength interval.

    This material is useful for general purpose debugging and testing energy
    conservation.

        >>> from raysect.primitive import Sphere
        >>> from raysect.optical import World
        >>> from raysect.optical.material import UnitySurfaceEmitter
        >>>
        >>> # set-up scenegraph
        >>> world = World()
        >>> emitter = Sphere(radius=0.01, parent=world, material=UnitySurfaceEmitter())
    """

    def __init__(self):
        super().__init__()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef Spectrum spectrum

        spectrum = ray.new_spectrum()
        spectrum.samples_mv[:] = 1.0
        return spectrum


cdef class UnityVolumeEmitter(HomogeneousVolumeEmitter):
    """
    Uniform, isotropic volume emitter with emission 1W/str/m^3/ x nm,
    where x is the spectrum's wavelength interval.

    This material is useful for general purpose debugging and evaluating the coupling
    coefficients between cameras and emitting volumes.

        >>> from raysect.primitive import Sphere
        >>> from raysect.optical import World
        >>> from raysect.optical.material import UnityVolumeEmitter
        >>>
        >>> # set-up scenegraph
        >>> world = World()
        >>> emitter = Sphere(radius=0.01, parent=world, material=UnityVolumeEmitter())
    """

    def __init__(self):
        super().__init__()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum emission_function(self, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):
        spectrum.samples_mv[:] = 1.0
        return spectrum
