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

import numpy as np
cimport numpy as np
from raysect.core.math.function.float.function2d cimport Function2D
from raysect.core.math.point cimport new_point2d
cimport cython


cdef class Discrete2DMesh(Function2D):
    """
    Discrete interpolator for data on a 2d ungridded tri-poly mesh.

    The mesh is specified as a set of 2D vertices supplied as an Nx2 numpy
    array or a suitably sized sequence that can be converted to a numpy array.

    The mesh triangles are defined with a Mx3 array where the three values are
    indices into the vertex array that specify the triangle vertices. The
    mesh must not contain overlapping triangles. Supplying a mesh with
    overlapping triangles will result in undefined behaviour.

    A data array of length M, containing a value for each triangle, holds the
    data to be interpolated across the mesh.

    By default, requesting a point outside the bounds of the mesh will cause
    a ValueError exception to be raised. If this is not desired the limit
    attribute (default True) can be set to False. When set to False, a default
    value will be returned for any point lying outside the mesh. The value
    return can be specified by setting the default_value attribute (default is
    0.0).

    To optimise the lookup of triangles, the interpolator builds an
    acceleration structure (a KD-Tree) from the specified mesh data. Depending
    on the size of the mesh, this can be quite slow to construct. If the user
    wishes to interpolate a number of different data sets across the same mesh
    - for example: temperature and density data that are both defined on the
    same mesh - then the user can use the instance() method on an existing
    interpolator to create a new interpolator. The new interpolator will shares
    a copy of the internal acceleration data. The triangle_data, limit and
    default_value can be customised for the new instance. See instance(). This
    will avoid the cost in memory and time of rebuilding an identical
    acceleration structure.

    :param ndarray vertex_coords: An array of vertex coordinates (x, y) with shape Nx2.
    :param ndarray triangles: An array of vertex indices defining the mesh triangles, with shape Mx3.
    :param ndarray triangle_data: An array containing data for each triangle of shape Mx1.
    :param bool limit: Raise an exception outside mesh limits - True (default) or False.
    :param float default_value: The value to return outside the mesh limits if limit is set to False.
    """

    def __init__(self, object vertex_coords not None, object triangles not None, object triangle_data not None, bint limit=True, double default_value=0.0):

        # use numpy arrays to store data internally
        vertex_coords = np.array(vertex_coords, dtype=np.float64)
        triangles = np.array(triangles, dtype=np.int32)
        triangle_data = np.array(triangle_data, dtype=np.float64)

        # validate triangle_data
        if triangle_data.ndim != 1 or triangle_data.shape[0] != triangles.shape[0]:
            raise ValueError("triangle_data dimensions ({}) are incompatible with the number of triangles ({}).".format(triangle_data.shape[0], triangles.shape[0]))

        # build kdtree
        self._kdtree = MeshKDTree2D(vertex_coords, triangles)

        # populate internal attributes
        self._triangle_data = triangle_data
        self._triangle_data_mv = triangle_data
        self._default_value = default_value
        self._limit = limit

    def __getstate__(self):
        return self._triangle_data, self._kdtree, self._limit, self._default_value

    def __setstate__(self, state):
        self._triangle_data, self._kdtree, self._limit, self._default_value = state
        self._triangle_data_mv = self._triangle_data

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @classmethod
    def instance(cls, Discrete2DMesh instance not None, object triangle_data=None, object limit=None, object default_value=None):
        """
        Creates a new interpolator instance from an existing interpolator instance.

        The new interpolator instance will share the same internal acceleration
        data as the original interpolator. The triangle_data, limit and default_value
        settings of the new instance can be redefined by setting the appropriate
        attributes. If any of the attributes are set to None (default) then the
        value from the original interpolator will be copied.

        This method should be used if the user has multiple sets of triangle_data
        that lie on the same mesh geometry. Using this methods avoids the
        repeated rebuilding of the mesh acceleration structures by sharing the
        geometry data between multiple interpolator objects.

        :param Discrete2DMesh instance: Discrete2DMesh object.
        :param ndarray triangle_data: An array containing data for each triangle of shape Mx1 (default None).
        :param bool limit: Raise an exception outside mesh limits - True (default) or False (default None).
        :param float default_value: The value to return outside the mesh limits if limit is set to False (default None).
        :return: An Discrete2DMesh object.
        :rtype: Discrete2DMesh
        """

        cdef Discrete2DMesh m

        # copy source data
        m = Discrete2DMesh.__new__(Discrete2DMesh)
        m._kdtree = instance._kdtree

        # do we have replacement triangle data?
        if triangle_data is None:
            m._triangle_data = instance._triangle_data
        else:
            m._triangle_data = np.array(triangle_data, dtype=np.float64)
            if m._triangle_data.ndim != 1 or m._triangle_data.shape[0] != instance._triangle_data.shape[0]:
                raise ValueError("triangle_data dimensions ({}) are incompatible with the number of triangles ({}).".format(m._triangle_data.shape[0], instance._triangle_data.shape[0]))

        # create memoryview
        m._triangle_data_mv = m._triangle_data

        # do we have a replacement limit check setting?
        if limit is None:
            m._limit = instance._limit
        else:
            m._limit = limit

        # do we have a replacement default value?
        if default_value is None:
            m._default_value = instance._default_value
        else:
            m._default_value = default_value

        return m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double x, double y) except? -1e999:

        cdef:
            np.int32_t triangle_id

        if self._kdtree.is_contained(new_point2d(x, y)):
            triangle_id = self._kdtree.triangle_id
            return self._triangle_data_mv[triangle_id]

        if not self._limit:
            return self._default_value

        raise ValueError("Requested value outside mesh bounds.")
