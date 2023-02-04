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

import numpy as np
from raysect.core.scenegraph.signal import MATERIAL

from raysect.core cimport BoundingBox3D, BoundingSphere3D, AffineMatrix3D, _NodeBase, ChangeSignal
from raysect.core.acceleration cimport BoundPrimitive
from raysect.core.math.random cimport uniform, vector_sphere, vector_cone_uniform
from raysect.core.math.cython cimport find_index, rotate_basis
from libc.math cimport M_PI as PI, asin, sqrt
cimport cython


class ImportanceError(Exception):
    pass


cdef class ImportanceManager:
    """
    Specialist class for managing sampling of important primitives.
    """

    def __init__(self, primitives):

        # The sum of importance weights on all important primitives in this scene-graph.
        self._total_importance = 0

        # A list of tuples defining the bounding spheres of all important primitives in the scene-graph.
        # Each tuple has the structure (bounding_sphere, primitive_importance).
        self._spheres = []

        if len(primitives) == 0:
            self._cdf = None
            self._cdf_mv = None
            return

        self._process_primitives(primitives)

        if self._total_importance == 0:
            # no important primitives were found
            self._cdf = None
            self._cdf_mv = None
            return

        # Populate numpy array storing the normalised cumulative importance weights of all important primitives.
        # Used for selecting a random primitive proportional to their respective weights.
        self._calculate_cdf()

    def __getstate__(self):
        state = self._cdf, self._total_importance, self._spheres

    def __setstate__(self, state):
        self._cdf, self._total_importance, self._spheres = state
        self._cdf_mv = self._cdf

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    cdef object _process_primitives(self, list primitives):
        """
        Process all the important primitives in the scene-graph.

        For any primitives with importance > 0, create a bounding sphere
        for the primitive and add its importance weighting to the
        cumulative importance value.

        :param list primitives: List of primitives in this scene-graph.
        """

        for primitive in primitives:
            if primitive.material.importance > 0:

                sphere = primitive.bounding_sphere()
                importance = primitive.material.importance

                self._total_importance += importance
                self._spheres.append((sphere, importance))

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef object _calculate_cdf(self):
        """
        Calculate the cumulative distribution function for import primitives (CDF).

        Stores an array with length equal to the number of important primitives. At
        each point in the array the normalised cumulative importance weighting is stored.
        """

        self._cdf = np.zeros(len(self._spheres), dtype=np.float64)
        for index, sphere_data in enumerate(self._spheres):
            _, importance = sphere_data
            if index == 0:
                self._cdf[index] = importance
            else:
                self._cdf[index] = self._cdf[index-1] + importance
        self._cdf /= self._total_importance

        # create memoryview for fast access
        self._cdf_mv = self._cdf

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef tuple _pick_sphere(self):
        """
        Find the important primitive bounding sphere corresponding to a uniform random number.
        """

        cdef:
            int index
            double probability
            BoundingBox3D box
            BoundPrimitive bound_primitive

        if self._cdf is None:
            return None

        # due to the CDF not starting at zero, using find_index means that the result is offset by 1 index point.
        index = find_index(self._cdf_mv, uniform()) + 1
        return self._spheres[index]

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef Vector3D sample(self, Point3D origin):
        """
        Sample a random important primitive weighted by their importance weight.

        :param Point3D origin: The point from which to sample.
        :return: The vector along which to sample.
        :rtype: Vector3D
        """

        # calculate projection of sphere (a disk) as seen from origin point and
        # generate a random direction towards that projection

        cdef:
            BoundingSphere3D sphere
            double importance, distance, angular_radius
            Vector3D direction, sample
            AffineMatrix3D rotation

        # TODO: move the projection code to a projection method on BoundingSphere3D

        if self._cdf is None:
            raise ImportanceError("Attempted to sample important direction when no important primitives have been"
                                  "specified.")

        sphere, importance = self._pick_sphere()

        direction = origin.vector_to(sphere.centre)
        distance = direction.get_length()

        # is point inside sphere?
        if distance == 0 or distance < sphere.radius:
            # the point lies inside the sphere, sample random direction from full sphere
            return vector_sphere()

        # calculate the angular radius and solid angle projection of the sphere
        angular_radius = asin(sphere.radius / distance)

        # sample a vector from a cone of half angle equal to the angular radius
        sample = vector_cone_uniform(angular_radius * 180 / PI)

        # rotate cone to lie along vector from observation point to sphere centre
        direction = direction.normalise()
        rotation = rotate_basis(direction, direction.orthogonal())
        return sample.transform(rotation)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double pdf(self, Point3D origin, Vector3D direction):
        """
        Calculates the value of the PDF for the specified sample point and direction.

        :param Point3D origin: The point from which to sample.
        :param Vector3D direction: The sample direction.
        :rtype: float
        """

        cdef:
            BoundingSphere3D sphere
            double importance, distance, solid_angle, angular_radius_cos, t
            double pdf_all, pdf_sphere, selection_weight
            Vector3D cone_axis
            AffineMatrix3D rotation

        pdf_all = 0
        for sphere, importance in self._spheres:

            cone_axis = origin.vector_to(sphere.centre)
            distance = cone_axis.get_length()

            # is point inside sphere?
            if distance == 0 or distance < sphere.radius:

                # the point lies inside the sphere, the projection is a full sphere
                solid_angle = 4 * PI

            else:

                # calculate cosine of angular radius of cone
                t = sphere.radius / distance
                angular_radius_cos = sqrt(1 - t * t)

                # does the direction lie inside the cone of projection
                cone_axis = cone_axis.normalise()
                if direction.dot(cone_axis) < angular_radius_cos:
                    # no contribution, outside code of projection
                    continue

                # calculate solid angle
                solid_angle = 2 * PI * (1 - angular_radius_cos)

            # calculate probability
            pdf_sphere = 1 / solid_angle
            selection_weight = importance / self._total_importance

            # add contribution to pdf
            pdf_all += selection_weight * pdf_sphere

        return pdf_all

    cpdef bint has_primitives(self):
        """
        Returns true if any primitives in this scene-graph have an importance weighting.

        :rtype: bool
        """
        return self._total_importance > 0


cdef class World(CoreWorld):
    """
    The root node of the optical scene-graph.

    Inherits a lot of functionality and attributes from the core World object.

    The world node tracks all primitives and observers in the world. It maintains acceleration structures to speed up
    the ray-tracing calculations. The particular acceleration algorithm used is selectable. The default acceleration
    structure is a kd-tree.

    :param name: A string defining the node name.
    """

    def __init__(self, str name=None):
        super().__init__(name)
        self._importance = None

    cpdef build_importance(self, bint force=False):
        """
        This method manually triggers a rebuild of the importance manager object.

        If the importance manager object is already in a consistent state this method
        will do nothing unless the force keyword option is set to True.

        :param bint force: If set to True, forces rebuilding of acceleration structure.
        """

        if self._importance is None or force:
            self._importance = ImportanceManager(self.primitives)

    def _change(self, _NodeBase node, ChangeSignal change not None):
        """
        Notifies the World of a change to the scene-graph.

        This method must be called if a change occurs that may have invalidated
        any acceleration structures held by the World, and also the important primitives
        list maintained be the importance manager.

        The node on which the change occurs and a ChangeSignal must be
        provided. The ChangeSignal must specify the nature of the change.

        The optical World object only recognises the MATERIAL signal. When a
        MATERIAL signal is recieved, the ImportanceManager is rebuilt to reflect
        changes to the important primitive list and their respective weights.
        """

        if change is MATERIAL:
            self._importance = None

        super()._change(node, change)

    cpdef Vector3D important_direction_sample(self, Point3D origin):
        """
        Get a sample direction of an important primitive.

        :param Point3D origin: The point from which to sample.
        :return: The vector along which to sample.
        :rtype: Vector3D
        """

        self.build_importance()
        return self._importance.sample(origin)

    cpdef double important_direction_pdf(self, Point3D origin, Vector3D direction):
        """
        Calculates the value of the PDF for the specified sample point and direction.

        :param Point3D origin: The point from which to sample.
        :param Vector3D direction: The sample direction.
        :rtype: float
        """

        self.build_importance()
        return self._importance.pdf(origin, direction)

    cpdef bint has_important_primitives(self):
        """
        Returns true if any primitives in this scene-graph have an importance weighting.

        :rtype: bool
        """

        self.build_importance()
        return self._importance.has_primitives()
