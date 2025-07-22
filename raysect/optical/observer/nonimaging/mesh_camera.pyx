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
cimport numpy as np
from libc.math cimport M_PI
from numpy cimport float32_t, int32_t

from raysect.core.math.random cimport point_triangle
from raysect.core.math.sampler cimport HemisphereCosineSampler
from raysect.optical cimport Ray, Point3D, new_point3d, new_vector3d, Vector3D, AffineMatrix3D, new_affinematrix3d
from raysect.optical.observer.base cimport Observer1D
from raysect.optical.observer.pipeline.mono import PowerPipeline1D
from raysect.optical.observer.sampler1d import FullFrameSampler1D
from raysect.primitive.mesh.mesh cimport Mesh
cimport cython

# convenience defines
DEF X = 0
DEF Y = 1
DEF Z = 2

DEF V1 = 0
DEF V2 = 1
DEF V3 = 2


cdef class MeshCamera(Observer1D):
    """
    Uses a supplied mesh surface as a linear camera.

    .. Warning::
       Users must be careful when using this camera to not double count radiance. For example,
       if you have a concave mesh its possible for two surfaces to see the same emission. In cases
       like this, the mesh should have an absorbing surface to prevent double counting.

    This observer samples over each triangle or a triangular mesh. At each point on the surface
    the incoming radiance over a hemisphere is sampled. The pixel id corresponds to the triangle
    id in the mesh.

    A mesh surface offset can be set to ensure samples don't collide with a coincident primitive.
    When set, the surface offset specifies the distance along the surface normal that the ray
    launch origin is shifted.

    :param Mesh mesh: The Mesh object to use as the sampling surface.
    :param float surface_offset: The offset from the mesh surface (default=0).
    :param list pipelines: The list of pipelines that will process the spectrum measured
      by this observer (default=PowerPipeline1D()).
    :param kwargs: **kwargs from Observer1D and _ObserverBase

    .. code-block:: pycon

        >>> from raysect.primitive import Mesh
        >>> from raysect.optical import World
        >>> from raysect.optical.material import AbsorbingSurface
        >>> from raysect.optical.observer import MeshCamera, PowerPipeline1D, MonoAdaptiveSampler1D
        >>>
        >>> world = World()
        >>>
        >>> mesh = Mesh.from_file("my_mesh.rsm", material=AbsorbingSurface(), parent=world)
        >>>
        >>> power = PowerPipeline1D()
        >>> sampler = MonoAdaptiveSampler1D(power, fraction=0.2, ratio=25.0, min_samples=1000, cutoff=0.1)
        >>> camera = MeshCamera(mesh,
                                surface_offset=1e-6,  # launch rays 1mm off surface to avoid intersection with absorbing mesh
                                pipelines=[power],
                                frame_sampler=sampler,
                                parent=world,
                                spectral_bins=1,
                                min_wavelength=400,
                                max_wavelength=740,
                                pixel_samples=250)
        >>> camera.observe()
    """

    def __init__(self, Mesh mesh not None, surface_offset=None, frame_sampler=None, pipelines=None, parent=None,
                 transform=None, name=None, render_engine=None, pixel_samples=None,
                 spectral_rays=None, spectral_bins=None, min_wavelength=None, max_wavelength=None,
                 ray_extinction_prob=None, ray_extinction_min_depth=None, ray_max_depth=None,
                 ray_importance_sampling=None, ray_important_path_weight=None, quiet=False):

        pipelines = pipelines or [PowerPipeline1D()]
        frame_sampler = frame_sampler or FullFrameSampler1D()

        pixels = mesh.data.triangles_mv.shape[0]
        super().__init__(pixels, frame_sampler, pipelines, parent=parent, transform=transform, name=name,
                         render_engine=render_engine, pixel_samples=pixel_samples,
                         spectral_rays=spectral_rays, spectral_bins=spectral_bins,
                         min_wavelength=min_wavelength, max_wavelength=max_wavelength,
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

    def __getstate__(self):

        state = (
            self._surface_offset,
            self._solid_angle,
            self._collection_area,
            self.mesh,
            self._areas,
            self._vector_sampler,
            super().__getstate__()
        )

    def __setstate__(self, state):

        (
            self._surface_offset,
            self._solid_angle,
            self._collection_area,
            self.mesh,
            self._areas,
            self._vector_sampler,
            super_state
        ) = state

        super().__setstate__(super_state)

        # recreate memoryviews
        self._areas_mv = self._areas
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
        self._areas = np.zeros(self._triangles_mv.shape[0])
        self._areas_mv = self._areas

        # calculate cumulative and total area simultaneously
        for i in range(self._triangles_mv.shape[0]):

            # obtain vertex indices
            v1i = self._triangles_mv[i, V1]
            v2i = self._triangles_mv[i, V2]
            v3i = self._triangles_mv[i, V3]

            # obtain area and accumulate
            self._areas_mv[i] = self._triangle_area(
                new_point3d(self._vertices_mv[v1i, X], self._vertices_mv[v1i, Y], self._vertices_mv[v1i, Z]),
                new_point3d(self._vertices_mv[v2i, X], self._vertices_mv[v2i, Y], self._vertices_mv[v2i, Z]),
                new_point3d(self._vertices_mv[v3i, X], self._vertices_mv[v3i, Y], self._vertices_mv[v3i, Z])
            )

    cdef double _triangle_area(self, Point3D v1, Point3D v2, Point3D v3):
        cdef Vector3D e1 = v1.vector_to(v2)
        cdef Vector3D e2 = v1.vector_to(v3)
        return 0.5 * e1.cross(e2).get_length()

    @property
    def collection_areas(self):
        return self._areas.copy()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double collection_area(self, int pixel):
        """
        The mesh camera's collection area in m^2.

        :rtype: float
        """
        if pixel < 0 or pixel >= self._areas_mv.shape[0]:
            raise ValueError('Pixel must lie in the range [0, number of triangles).')
        return self._areas_mv[pixel]

    @property
    def solid_angles(self):
        return np.ones(self._areas.shape[0]) * self._solid_angle

    cpdef double solid_angle(self, int pixel):
        """
        The solid angle observed at each mesh triangle in steradians str.

        :rtype: float
        """
        if pixel < 0 or pixel >= self._areas_mv.shape[0]:
            raise ValueError('Pixel must lie in the range [0, number of triangles).')
        return self._solid_angle

    @property
    def sensitivitys(self):
        return self._areas.copy() * self._solid_angle

    cpdef double sensitivity(self, int pixel):
        """
        The mesh camera's sensitivity measured in units of per area per solid angle (m^-2 str^-1).

        :rtype: float
        """
        if pixel < 0 or pixel >= self._areas_mv.shape[0]:
            raise ValueError('Pixel must lie in the range [0, number of triangles).')
        return self._pixel_sensitivity(pixel)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef list _generate_rays(self, int pixel, Ray template, int ray_count):

        cdef:
            list rays, origins, directions
            Point3D v1, v2, v3, origin
            Vector3D normal, direction
            int n
            double weight
            AffineMatrix3D surface_to_local
            int32_t triangle, v1i, v2i, v3i

        directions = self._vector_sampler.samples(ray_count)

        # unpack vertex indices
        v1i = self._triangles_mv[pixel, V1]
        v2i = self._triangles_mv[pixel, V2]
        v3i = self._triangles_mv[pixel, V3]

        # generate vertex points
        v1 = new_point3d(self._vertices_mv[v1i, X], self._vertices_mv[v1i, Y], self._vertices_mv[v1i, Z])
        v2 = new_point3d(self._vertices_mv[v2i, X], self._vertices_mv[v2i, Y], self._vertices_mv[v2i, Z])
        v3 = new_point3d(self._vertices_mv[v3i, X], self._vertices_mv[v3i, Y], self._vertices_mv[v3i, Z])

        # obtain face normal
        normal = new_vector3d(
            self._face_normals_mv[pixel, X],
            self._face_normals_mv[pixel, Y],
            self._face_normals_mv[pixel, Z]
        )

        # generate the transform from the triangle surface normal to local (z-axis aligned)
        surface_to_local = self._surface_to_local(normal)

        rays = []
        for n in range(ray_count):

            # sample triangle surface to get origin point in local space
            origin = point_triangle(v1, v2, v3)

            # shift origin point forward along normal by distance in surface_offset
            origin = origin.add(normal.mul(self._surface_offset))

            # rotate direction sample so z aligned with face normal
            direction = (<Vector3D> directions[n]).transform(surface_to_local)

            # cosine weighted distribution
            # projected area cosine is implicit in distribution
            # weight = (1 / 2*pi) * (pi / cos(theta)) * cos(theta) = 0.5
            rays.append((template.copy(origin, direction), 0.5))

        return rays

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

    cpdef double _pixel_sensitivity(self, int pixel):
        return self._solid_angle * self._areas_mv[pixel]
