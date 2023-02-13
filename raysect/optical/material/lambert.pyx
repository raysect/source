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

from raysect.core.math.sampler cimport HemisphereCosineSampler
from raysect.optical cimport Point3D, Vector3D, AffineMatrix3D, Primitive, World, Ray, Spectrum, SpectralFunction, ConstantSF, Intersection
from raysect.optical.material cimport ContinuousBSDF
from numpy cimport ndarray


cdef HemisphereCosineSampler hemisphere_sampler = HemisphereCosineSampler()


cdef class Lambert(ContinuousBSDF):
    """
    An ideal Lambertian surface material.

    A Lambertian is a perfectly diffuse surface that scatters light equally in
    all directions. It is a good approximation to many real world surfaces.

    :param SpectralFunction reflectivity: Reflectance function which defines the
      fraction of light scattered at each wavelength.

    .. code-block:: pycon

        >>> from raysect.primitive import Sphere
        >>> from raysect.optical import World, ConstantSF
        >>> from raysect.optical.material import Lambert
        >>>
        >>> # set-up scenegraph
        >>> world = World()
        >>> sphere = Sphere(radius=0.01, parent=world)
        >>> sphere.material=Lambert(0.25)  # 25% diffuse reflectivity
    """

    cdef SpectralFunction reflectivity

    def __init__(self, SpectralFunction reflectivity=None):

        super().__init__()
        if reflectivity is None:
            reflectivity = ConstantSF(0.5)
        self.reflectivity = reflectivity

    cpdef double pdf(self, Vector3D s_incoming, Vector3D s_outgoing, bint back_face):
        return hemisphere_sampler.pdf(s_outgoing)

    cpdef Vector3D sample(self, Vector3D s_incoming, bint back_face):
        return hemisphere_sampler.sample()

    cpdef Spectrum evaluate_shading(self, World world, Ray ray, Vector3D s_incoming, Vector3D s_outgoing,
                                    Point3D w_reflection_origin, Point3D w_transmission_origin, bint back_face,
                                    AffineMatrix3D world_to_surface, AffineMatrix3D surface_to_world,
                                    Intersection intersection):

        cdef:
            Spectrum spectrum
            Ray reflected
            double[::1] reflectivity
            double pdf

        # outgoing ray is sampling incident light so s_outgoing = incident

        # lambert material does not transmit
        pdf = hemisphere_sampler.pdf(s_outgoing)
        if pdf == 0.0:
            return ray.new_spectrum()

        # generate and trace ray
        reflected = ray.spawn_daughter(w_reflection_origin, s_outgoing.transform(surface_to_world))
        spectrum = reflected.trace(world)

        # obtain samples of reflectivity
        reflectivity = self.reflectivity.sample_mv(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)

        # combine and normalise
        spectrum.mul_array(reflectivity)
        spectrum.mul_scalar(pdf)
        return spectrum

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # no volume contribution
        return spectrum



