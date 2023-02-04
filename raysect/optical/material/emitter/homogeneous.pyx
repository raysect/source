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

from raysect.optical cimport Point3D
cimport cython


cdef class HomogeneousVolumeEmitter(NullSurface):
    """
    Base class for homogeneous volume emitters.

    Total power output of the light from each point is constant,
    but not necessarily isotropic.

    The deriving class must implement the emission_function() method.
    """

    def __init__(self):
        super().__init__()
        self.importance = 1.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D direction
            double length
            Spectrum emission
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
        emission = ray.new_spectrum()

        # emission function specifies direction from ray origin to hit-point
        emission = self.emission_function(direction, emission, world, ray, primitive, world_to_primitive, primitive_to_world)

        # sanity check as bounds checking is disabled
        if emission.samples.ndim != 1 or spectrum.samples.ndim != 1 or emission.samples.shape[0] != spectrum.samples.shape[0]:
            raise ValueError("Spectrum returned by emission function has the wrong number of bins.")

        # integrate emission density along ray path
        for index in range(spectrum.bins):
            spectrum.samples_mv[index] += emission.samples_mv[index] * length

        return spectrum

    cpdef Spectrum emission_function(self, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):
        """
        The emission function for the material.

        This is a virtual method and must be implemented in a sub class.

        :param Vector3D direction: The emission direction vector in local coordinates.
        :param Spectrum spectrum: Spectrum measured so far along ray path. Add your emission
          to this spectrum, don't override it.
        :param World world: The world scene-graph.
        :param Ray ray: The ray being traced.
        :param Primitive primitive: The geometric primitive to which this material belongs
          (i.e. a cylinder or a mesh).
        :param AffineMatrix3D world_to_primitive: Affine matrix defining the coordinate
          transform from world space to the primitive's local space.
        :param AffineMatrix3D primitive_to_world: Affine matrix defining the coordinate
          transform from the primitive's local space to world space.
        """

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")
