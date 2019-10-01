# cython: language_level=3

# Copyright (c) 2014-2017, Dr Alex Meakins, Raysect Project
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
cimport numpy as np
from libc.math cimport M_PI
from numpy cimport float32_t, int32_t

from raysect.core.math.random cimport uniform, point_triangle, probability
from raysect.core.math.sampler cimport HemisphereCosineSampler, TargettedHemisphereSampler
from raysect.core.math.cython cimport find_index
from raysect.optical cimport Ray, Primitive, Point3D, new_point3d, new_vector3d, Vector3D, AffineMatrix3D, new_affinematrix3d
from raysect.optical.observer.base cimport Observer0D
from raysect.optical.observer.pipeline.spectral import SpectralPowerPipeline0D
from raysect.primitive.mesh.mesh cimport Mesh
cimport cython

# convenience defines
DEF X = 0
DEF Y = 1
DEF Z = 2

DEF V1 = 0
DEF V2 = 1
DEF V3 = 2

DEF R_2_PI = 0.15915494309189535  # 1 / (2 * pi)


cdef class MeshPixel(Observer0D):
    """
    Uses a supplied mesh surface as a pixel.

    .. Warning::
       Users must be careful when using this camera to not double count radiance. For example,
       if you have a concave mesh its possible for two surfaces to see the same emission. In cases
       like this, the mesh should have an absorbing surface to prevent double counting.

    This observer samples over the surface defined by a triangular mesh. At each point on the surface
    the incoming radiance over a hemisphere is sampled.

    A mesh surface offset can be set to ensure sample don't collide with a coincident primitive. When set,
    the surface offset specifies the distance along the surface normal that the ray launch origin is shifted.

    :param Mesh mesh: The mesh instance to use for observations.
    :param float surface_offset: The offset from the mesh surface (default=0).
    :param list pipelines: The list of pipelines that will process the spectrum measured
      by this pixel (default=SpectralPowerPipeline0D()).
    :param kwargs: **kwargs from Observer0D and _ObserverBase

    .. code-block:: pycon

        >>> from raysect.primitive import Mesh
        >>> from raysect.optical import World
        >>> from raysect.optical.material import AbsorbingSurface
        >>> from raysect.optical.observer import MeshPixel, PowerPipeline0D
        >>>
        >>> world = World()
        >>>
        >>> mesh = Mesh.from_file("my_mesh.rsm", material=AbsorbingSurface(), parent=world)
        >>>
        >>> power = PowerPipeline0D(accumulate=False)
        >>> observer = MeshPixel(mesh, pipelines=[power], parent=world,
        >>>                      min_wavelength=400, max_wavelength=750,
        >>>                      spectral_bins=1, pixel_samples=10000, surface_offset=1E-6)
        >>> observer.observe()
    """

    cdef:
        double _surface_offset, _solid_angle, _collection_area, _targetted_path_prob
        readonly Mesh mesh
        float32_t[:, ::1] _vertices_mv
        float32_t[:, ::1] _face_normals_mv
        int32_t[:, ::1] _triangles_mv
        np.ndarray _cdf
        double [::1] _cdf_mv
        tuple _targets
        TargettedHemisphereSampler _targetted_sampler
        HemisphereCosineSampler _vector_sampler

    def __init__(self, Mesh mesh not None, surface_offset=None, pipelines=None, parent=None, transform=None, name=None,
                 render_engine=None, pixel_samples=None, samples_per_task=None, spectral_rays=None, spectral_bins=None,
                 min_wavelength=None, max_wavelength=None, ray_extinction_prob=None, ray_extinction_min_depth=None,
                 ray_max_depth=None, ray_importance_sampling=None, ray_important_path_weight=None,
                 targets=None, targetted_path_prob=None, quiet=False):

        pipelines = pipelines or [SpectralPowerPipeline0D()]

        super().__init__(pipelines, parent=parent, transform=transform, name=name, render_engine=render_engine,
                         pixel_samples=pixel_samples, samples_per_task=samples_per_task, spectral_rays=spectral_rays,
                         spectral_bins=spectral_bins, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                         ray_extinction_prob=ray_extinction_prob, ray_extinction_min_depth=ray_extinction_min_depth,
                         ray_max_depth=ray_max_depth, ray_importance_sampling=ray_importance_sampling,
                         ray_important_path_weight=ray_important_path_weight, quiet=quiet)

        surface_offset = surface_offset or 0.0
        if surface_offset < 0:
            raise ValueError("Surface offset must be greater than or equal to zero.")

        self._surface_offset = surface_offset

        self.mesh = mesh
        self._vertices_mv = mesh.data.vertices_mv
        self._face_normals_mv = mesh.data.face_normals_mv
        self._triangles_mv = mesh.data.triangles_mv

        self._vector_sampler = HemisphereCosineSampler()
        self._solid_angle = 2 * M_PI
        self._calculate_areas()

        self.targets = targets or tuple()
        self.targetted_path_prob = targetted_path_prob or 0.7

    def __getstate__(self):

        state = (
            self._surface_offset,
            self._solid_angle,
            self._collection_area,
            self.mesh,
            self._cdf,
            self._vector_sampler,
            super().__getstate__()
        )

    def __setstate__(self, state):

        (
            self._surface_offset,
            self._solid_angle,
            self._collection_area,
            self.mesh,
            self._cdf,
            self._vector_sampler,
            super_state
        ) = state

        super().__setstate__(super_state)

        # recreate memoryviews
        self._cdf_mv = self._cdf
        self._vertices_mv = self.mesh.data.vertices_mv
        self._face_normals_mv = self.mesh.data.face_normals_mv
        self._triangles_mv = self.mesh.data.triangles_mv

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef object _calculate_areas(self):
        cdef:
            int32_t i, v1i, v2i, v3i

        # initialise area attributes
        self._collection_area = 0.0
        self._cdf = np.zeros(self._triangles_mv.shape[0])
        self._cdf_mv = self._cdf
        
        # calculate cumulative and total area simultaneously 
        for i in range(self._triangles_mv.shape[0]):

            # obtain vertex indices
            v1i = self._triangles_mv[i, V1]
            v2i = self._triangles_mv[i, V2]
            v3i = self._triangles_mv[i, V3]

            # obtain area and accumulate
            triangle_area = self._triangle_area(
                new_point3d(self._vertices_mv[v1i, X], self._vertices_mv[v1i, Y], self._vertices_mv[v1i, Z]),
                new_point3d(self._vertices_mv[v2i, X], self._vertices_mv[v2i, Y], self._vertices_mv[v2i, Z]),
                new_point3d(self._vertices_mv[v3i, X], self._vertices_mv[v3i, Y], self._vertices_mv[v3i, Z])
            )
            self._collection_area += triangle_area
            self._cdf_mv[i] = self._collection_area
        
        # normalise cumulative area to make cdf
        self._cdf /= self._collection_area

    cdef double _triangle_area(self, Point3D v1, Point3D v2, Point3D v3):
        cdef Vector3D e1 = v1.vector_to(v2)
        cdef Vector3D e2 = v1.vector_to(v3)
        return 0.5 * e1.cross(e2).get_length()

    @property
    def collection_area(self):
        """
        The pixel's collection area in m^2.

        :rtype: float
        """
        return self._collection_area

    @property
    def solid_angle(self):
        """
        The pixel's solid angle in steradians str.

        :rtype: float
        """
        return self._solid_angle

    @property
    def sensitivity(self):
        """
        The pixel's sensitivity measured in units of per area per solid angle (m^-2 str^-1).

        :rtype: float
        """
        return self._pixel_sensitivity()

    @property
    def targets(self):
        """
        The list of primitives this mesh will target for sampling.

        :rtype: tuple
        """
        return self._targets

    @targets.setter
    def targets(self, value):

        # No targets?
        if value is None:
            self._targets = ()
            return

        value = tuple(value)

        # List must contain only primitives
        for target in value:
            if not isinstance(target, Primitive):
                raise TypeError("Target list must contain only primitives.")

        self._targets = value

    @property
    def targetted_path_prob(self):
        """
        The probability that an individual sample will be fired at a target instead of a sample from the whole
        observing hemisphere.

        .. Warning..
           If the target probability is set to 1, rays will only be fired directly towards the targets. The user must
           ensure there are no sources of radiance outside of the targeted directions, otherwise they will not be
           sampled and the result will be biased.

        :rtype: float
        """
        return self._targetted_path_prob

    @targetted_path_prob.setter
    def targetted_path_prob(self, double value):

        if value < 0 or value > 1:
            raise ValueError("Targeted path probability must lie in the range [0, 1].")
        self._targetted_path_prob = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef list _generate_rays(self, Ray template, int ray_count):

        cdef:
            list rays, origins, directions
            Point3D origin, surface_origin, sphere_centre
            Vector3D normal, direction
            int n
            double weight
            AffineMatrix3D surface_to_local, local_to_surface
            int32_t triangle, v1i, v2i, v3i

        # test target primitives are in the same scene-graph as the observer
        for target in self._targets:
            if target.root is not self.root:
                raise ValueError("Target primitive is not in the same scenegraph as the TargetedPixel observer.")

        rays = []
        for n in range(ray_count):

            # pick triangle
            triangle = self._pick_triangle()

            # unpack vertex indices
            v1i = self._triangles_mv[triangle, V1]
            v2i = self._triangles_mv[triangle, V2]
            v3i = self._triangles_mv[triangle, V3]

            # obtain face normal
            normal = new_vector3d(
                self._face_normals_mv[triangle, X],
                self._face_normals_mv[triangle, Y],
                self._face_normals_mv[triangle, Z]
            )

            # sample triangle surface to get origin point in local space
            origin = point_triangle(
                new_point3d(self._vertices_mv[v1i, X], self._vertices_mv[v1i, Y], self._vertices_mv[v1i, Z]),
                new_point3d(self._vertices_mv[v2i, X], self._vertices_mv[v2i, Y], self._vertices_mv[v2i, Z]),
                new_point3d(self._vertices_mv[v3i, X], self._vertices_mv[v3i, Y], self._vertices_mv[v3i, Z])
            )

            # generate the transform from the triangle surface normal to local (z-axis aligned)
            surface_to_local = self._surface_to_local(normal)
            local_to_surface = surface_to_local.inverse()

            # shift origin point forward along normal by distance in surface_offset
            origin = origin.add(normal.mul(self._surface_offset))
            surface_origin = origin.transform(local_to_surface)

            if len(self._targets) > 0:

                # generate bounding spheres and convert to local coordinate system
                spheres = []
                for target in self._targets:
                    sphere = target.bounding_sphere()
                    sphere_centre = sphere.centre.transform(self.to_local()).transform(surface_to_local)
                    spheres.append((sphere_centre, sphere.radius, 1.0))

                # instance targetted pixel sampler
                targetted_sampler = TargettedHemisphereSampler(spheres)

                if probability(self._targetted_path_prob):
                    # obtain targetted vector sample
                    direction = targetted_sampler.sample(surface_origin)

                else:
                    # obtain cosine weighted hemisphere sample
                    direction = self._vector_sampler.sample()

                # calculate combined pdf and ray weight
                pdf = self._targetted_path_prob * targetted_sampler.pdf(origin, direction) + \
                      (1-self._targetted_path_prob) * self._vector_sampler.pdf(direction)

                # weight = 1 / (2 * pi) * cos(theta) * 1/pdf
                weight = R_2_PI * direction.z / pdf

            else:
                # obtain cosine weighted hemisphere sample
                direction = self._vector_sampler.sample()

                # cosine weighted distribution
                # projected area cosine is implicit in distribution
                # weight = (1 / 2*pi) * (pi / cos(theta)) * cos(theta) = 0.5
                weight = 0.5

            # rotate direction sample so z aligned with face normal
            direction = direction.transform(surface_to_local)

            rays.append((template.copy(origin, direction), weight))

        return rays

    @cython.initializedcheck(False)
    cdef int32_t _pick_triangle(self):
        """
        Pick a triangle such that sample points are uniform across the surface area.
        """

        # due to the CDF not starting at zero, using find_index means that the result is offset by 1 index point.
        return find_index(self._cdf_mv, uniform()) + 1

    cdef AffineMatrix3D _surface_to_local(self, Vector3D normal):
        """
        Calculates the surface to local space transform.

        :param Vector3D normal: Local space surface normal.
        :return: AffineMatrix3D surface to local transform.
        """

        cdef:
            Vector3D tangent, bitangent
            AffineMatrix3D surface_to_primitive

        tangent = normal.orthogonal()
        bitangent = normal.cross(tangent)

        surface_to_primitive = new_affinematrix3d(
            tangent.x, bitangent.x, normal.x, 0.0,
            tangent.y, bitangent.y, normal.y, 0.0,
            tangent.z, bitangent.z, normal.z, 0.0,
            0.0, 0.0, 0.0, 1.0
        )

        return surface_to_primitive

    cpdef double _pixel_sensitivity(self):
        return self._solid_angle * self._collection_area
