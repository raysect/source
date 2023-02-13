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

from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D, Normal3D, Intersection
cimport cython


cdef class UniformSurfaceEmitter(NullVolume):
    """
    Uniform and isotropic surface emitter.

    Uniform emission will be given by the emission_spectrum multiplied by the
    emission scale.

    :param SpectralFunction emission_spectrum: The surface's emission function.
    :param float scale: Scale of the emission function (default = 1 W/m^2/str/nm).

    .. code-block:: pycon

        >>> from raysect.primitive import Sphere
        >>> from raysect.optical import World, ConstantSF
        >>> from raysect.optical.material import UniformSurfaceEmitter
        >>>
        >>> # set-up scenegraph
        >>> world = World()
        >>> emitter = Sphere(radius=0.01, parent=world)
        >>> emitter.material=UniformSurfaceEmitter(ConstantSF(1.0))
    """

    def __init__(self, SpectralFunction emission_spectrum, double scale = 1.0):
        super().__init__()
        self.emission_spectrum = emission_spectrum
        self.scale = scale
        self.importance = 1.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef:
            Spectrum spectrum
            double[::1] emission
            int index

        spectrum = ray.new_spectrum()
        emission = self.emission_spectrum.sample_mv(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
        for index in range(spectrum.bins):
            spectrum.samples_mv[index] = emission[index] * self.scale
        return spectrum

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # no volume contribution
        return spectrum


cdef class UniformVolumeEmitter(HomogeneousVolumeEmitter):
    """
    Uniform, homogeneous and isotropic volume emitter.

    Uniform emission will be given by the emission_spectrum multiplied by the
    emission scale in radiance.

    :param SpectralFunction emission_spectrum: The volume's emission function.
    :param float scale: Scale of the emission function (default = 1 W/m^3/str/nm).

    .. code-block:: pycon

        >>> from raysect.primitive import Sphere
        >>> from raysect.optical import World, ConstantSF
        >>> from raysect.optical.material import UniformVolumeEmitter
        >>>
        >>> # set-up scenegraph
        >>> world = World()
        >>> emitter = Sphere(radius=0.01, parent=world)
        >>> emitter.material=UniformVolumeEmitter(ConstantSF(1.0))
    """

    def __init__(self, SpectralFunction emission_spectrum, double scale=1.0):
        super().__init__()
        self.emission_spectrum = emission_spectrum
        self.scale = scale

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum emission_function(self, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            double[::1] emission
            int index

        emission = self.emission_spectrum.sample_mv(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
        for index in range(spectrum.bins):
            spectrum.samples_mv[index] += emission[index] * self.scale

        return spectrum

