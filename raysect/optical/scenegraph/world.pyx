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

from numpy import zeros
from raysect.core.scenegraph.signal import MATERIAL

from raysect.core cimport BoundingBox3D, AffineMatrix3D, _NodeBase, ChangeSignal
from raysect.core.acceleration cimport BoundPrimitive
from raysect.core.math.random cimport uniform, vector_sphere, vector_cone
from raysect.core.math.cython cimport find_index, rotate_basis
from libc.math cimport M_PI as PI, asin, sqrt
cimport cython


class ImportanceError(Exception):
    pass


# TODO: docstrings
cdef class ImportanceManager:

    def __init__(self, primitives):

        self.total_importance = 0
        self.spheres = []

        if len(primitives) == 0:
            self.cdf = None
            return

        self._process_primitives(primitives)

        if self.total_importance == 0:
            # no important primitives were found
            self.cdf = None
            return

        self._calculate_cdf()

    cdef object _process_primitives(self, list primitives):

        for primitive in primitives:
            if primitive.material.importance > 0:

                # generate bounding box
                box = primitive.bounding_box()

                # obtain bounding sphere and importance
                centre = box.centre
                radius = box.enclosing_sphere()
                importance = primitive.material.importance

                self.total_importance += importance
                self.spheres.append((centre, radius, importance))

    cdef object _calculate_cdf(self):

        self.cdf = zeros(len(self.spheres))
        for index, sphere_data in enumerate(self.spheres):
            _, _, importance = sphere_data
            if index == 0:
                self.cdf[index] = importance
            else:
                self.cdf[index] = self.cdf[index-1] + importance
        self.cdf /= self.total_importance

    cdef inline tuple _pick_sphere(self):

        cdef:
            int index
            double probability
            BoundingBox3D box
            BoundPrimitive bound_primitive

        if self.cdf is None:
            return None

        # due to the CDF not starting at zero, using find_index means that the result is offset by 1 index point.
        index = find_index(self.cdf, uniform()) + 1
        return self.spheres[index]

    @cython.cdivision(True)
    cpdef Vector3D sample(self, Point3D origin):

        # calculate projection of sphere (a disk) as seen from origin point and
        # generate a random direction towards that projection

        cdef:
            Point3D centre
            double radius, importance, distance, angular_radius
            Vector3D direction, sample
            AffineMatrix3D rotation

        if self.cdf is None:
            raise ImportanceError("Attempted to sample important direction when no important primitives have been"
                                  "specified.")

        centre, radius, importance = self._pick_sphere()

        direction = origin.vector_to(centre)
        distance = direction.get_length()

        # is point inside sphere?
        if distance == 0 or distance < radius:
            # the point lies inside the sphere, sample random direction from full sphere
            return vector_sphere()

        # calculate the angular radius and solid angle projection of the sphere
        angular_radius = asin(radius / distance)

        # sample a vector from a cone of half angle equal to the angular radius
        sample = vector_cone(angular_radius * 180 / PI)

        # rotate cone to lie along vector from observation point to sphere centre
        direction = direction.normalise()
        rotation = rotate_basis(direction, direction.orthogonal())
        return sample.transform(rotation)

    @cython.cdivision(True)
    cpdef double pdf(self, Point3D origin, Vector3D direction):
        """
        Calculates the value of the PDF in the specified direction.
        """

        cdef:
            double radius, importance, distance, solid_angle, angular_radius_cos, t
            double pdf_all, pdf_sphere, selection_weight
            Point3D centre
            Vector3D cone_axis
            AffineMatrix3D rotation

        pdf_all = 0
        for centre, radius, importance in self.spheres:

            cone_axis = origin.vector_to(centre)
            distance = cone_axis.get_length()

            # is point inside sphere?
            if distance == 0 or distance < radius:

                # the point lies inside the sphere, the projection is a full sphere
                solid_angle = 4 * PI

            else:

                # calculate cosine of angular radius of cone
                t = radius / distance
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
            selection_weight = importance / self.total_importance

            # add contribution to pdf
            pdf_all += selection_weight * pdf_sphere

        return pdf_all

    cpdef bint has_primitives(self):
        return self.total_importance > 0


# # TODO: update docstrings
cdef class World(CoreWorld):
    """
    The root node of the optical scene-graph.

    The world node tracks all primitives and observers in the world. It maintains acceleration structures to speed up
    the ray-tracing calculations. The particular acceleration algorithm used is selectable. The default acceleration
    structure is a kd-tree.

    :param name: A string defining the node name.
    """

    def __init__(self, str name=None):
        super().__init__(name)
        self._importance = None

    cpdef build_importance(self, bint force=False):
        # """
        # This method manually triggers a rebuild of the Acceleration object.
        #
        # If the Acceleration object is already in a consistent state this method
        # will do nothing unless the force keyword option is set to True.
        #
        # The Acceleration object is used to accelerate hit() and contains()
        # calculations, typically using a spatial sub-division method. If changes are
        # made to the scene-graph structure, transforms or to a primitive's
        # geometry the acceleration structures may no longer represent the
        # geometry of the scene and hence must be rebuilt. This process is
        # usually performed automatically as part of the first call to hit() or
        # contains() following a change in the scene-graph. As calculating these
        # structures can take some time, this method provides the option of
        # triggering a rebuild outside of hit() and contains() in case the user wants
        # to be able to perform a benchmark without including the overhead of the
        # Acceleration object rebuild.
        #
        # :param bint force: If set to True, forces rebuilding of acceleration structure.
        # """

        if self._importance is None or force:
            self._importance = ImportanceManager(self.primitives)

    def _change(self, _NodeBase node, ChangeSignal change not None):
        # """
        # Notifies the World of a change to the scene-graph.
        #
        # This method must be called is a change occurs that may have invalidated
        # any acceleration structures held by the World.
        #
        # The node on which the change occurs and a ChangeSignal must be
        # provided. The ChangeSignal must specify the nature of the change to the
        # scene-graph.
        #
        # The core World object only recognises the GEOMETRY signal. When a
        # GEOMETRY signal is received, the world will be instructed to rebuild
        # it's spatial acceleration structures on the next call to any method
        # that interacts with the scene-graph geometry.
        # """

        if change is MATERIAL:
            self._importance = None

        super()._change(node, change)

    cpdef Vector3D important_direction_sample(self, Point3D origin):
        self.build_importance()
        return self._importance.sample(origin)

    cpdef double important_direction_pdf(self, Point3D origin, Vector3D direction):
        self.build_importance()
        return self._importance.pdf(origin, direction)

    cpdef bint has_important_primitives(self):
        self.build_importance()
        return self._importance.has_primitives()
