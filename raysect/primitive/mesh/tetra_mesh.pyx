# cython: language_level=3

# Copyright (c) 2014-2021, Dr Alex Meakins, Raysect Project
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

import io
import struct

import numpy as np
cimport numpy as np
from libc.math cimport fabs
from raysect.core.math.spatial.kdtree3d cimport Item3D
from raysect.core.boundingbox cimport BoundingBox3D, new_boundingbox3d
from raysect.core.math cimport AffineMatrix3D, Point3D, new_point3d, Vector3D
from raysect.core.math.cython cimport barycentric_inside_tetrahedra, barycentric_coords_tetra
from cpython.bytes cimport PyBytes_AsString
cimport cython

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-6

# convenience defines
DEF V1 = 0
DEF V2 = 1
DEF V3 = 2
DEF V4 = 3

DEF X = 0
DEF Y = 1
DEF Z = 2

# raysect mesh format constants
DEF RSM_VERSION_MAJOR = 1
DEF RSM_VERSION_MINOR = 1


cdef class TetraMesh(KDTree3DCore):
    """
    Holds the 3D tetrahedral mesh data and acceleration structures.
    
    This arrangement simplifies tetrahedral mesh instancing and the load/dump methods.

    The mesh vertices are supplied as an Nx3 list/array of floating point
    values. For each Vertex, x, y and z coordinates must be supplied. e.g.

        vertices = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], ...]

    The tetrahedral array is an Mx4. Tetrahedra are defined by indexing i.e:

        tetrahedra = [[v1, v2, v3, v4], ...]

    where v1, v2, v3, v4 are the vertex array indices specifying the tetrahedral vertices.

    :param object vertices: A list/array or tetrahedral vertices with shape Nx3,
      where N is the number of vertices.
    :param object tetrahedra: A list/array of tetrahedra with shape Mx4,
      where M is the number of tetrahedra in the mesh. For each tetrahedra there
      must be four integers identifying the tetrahedral vertices in the vertices array.
    :param bool tolerant: Toggles filtering out of degenerate tetrahedra
      (default=True).

    :ivar ndarray vertices: tetrahedral vertices with shape Nx3, where N is the number of vertices.
    :ivar ndarray tetrahedra: tetrahedra with shape Mx4, where M is the number
      of tetrahedra in the mesh.
    """

    def __init__(self, object vertices not None, object tetrahedra not None, bint tolerant=True):

        vertices = np.array(vertices, dtype=np.double)
        tetrahedra = np.array(tetrahedra, dtype=np.int32)

        # check dimensions are correct
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("The vertex array must have dimensions Nx3.")

        if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
            raise ValueError("The tetrahedra array must have dimensions Mx4.")

        # check tetrahedra contains only valid indices
        invalid = (tetrahedra[:, 0:4] < 0) | (tetrahedra[:, 0:4] >= vertices.shape[0])
        if invalid.any():
            raise ValueError("The tetrahedra array references non-existent vertices.")

        # assign to internal attributes
        self._vertices = vertices
        self._tetrahedra = tetrahedra

        # assign to memory views
        self.vertices_mv = vertices
        self.tetrahedra_mv = tetrahedra

        # initialise hit state attributes
        self.tetrahedra_id = -1
        self.i1 = -1
        self.i2 = -1
        self.i3 = -1
        self.i4 = -1
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.delta = 0.0

        # filter out degenerate tetrahedra if we are being tolerant
        if tolerant:
            self._filter_tetrahedra()

        # kd-Tree init
        items = []
        for tetrahedra in range(self._tetrahedra.shape[0]):
            items.append(Item3D(tetrahedra, self._generate_bounding_box(tetrahedra)))
        super().__init__(items, max_depth=0, min_items=1, hit_cost=50.0, empty_bonus=0.2)

        # TODO: (possible enhancement) check if tetrahedra are overlapping?
        # (any non-owned vertex lying inside another tetrahedra)

        # init cache
        self._cache_available = False
        self._cached_x = 0.0
        self._cached_y = 0.0
        self._cached_z = 0.0
        self._cached_result = False

    def __getstate__(self):
        state = io.BytesIO()
        self.save(state)
        return state.getvalue()

    def __setstate__(self, state):
        self.load(io.BytesIO(state))

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @property
    def vertices(self):
        return self._vertices.copy()

    @property
    def tetrahedra(self):
        return self._tetrahedra.copy()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Point3D vertex(self, int index):
        """
        Returns the specified vertex.

        :param int index: The vertex index.
        :return: A Point3D object.
        :rtype: Point3D
        """

        if index < 0 or index >= self.vertices_mv.shape[0]:
            raise ValueError('Vertex index is out of range: [0, {}].'.format(self.vertices_mv.shape[0]))

        return new_point3d(
            self.vertices_mv[index, X],
            self.vertices_mv[index, Y],
            self.vertices_mv[index, Z]
        )

    cpdef ndarray tetrahedron(self, int index):
        """
        Returns the specified tetrahedron.

        The returned data will be a 4 element numpy array which are the tetrahedral vertex indices.

        :param int index: The tetrahedral index.
        :return: A numpy array.
        :rtype: ndarray
        """

        if index < 0 or index >= self.vertices_mv.shape[0]:
            raise ValueError('Tetrahedral index is out of range: [0, {}].'.format(self.tetrahedra_mv.shape[0]))

        return self._tetrahedra[index, :].copy()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double volume(self, int index):
        """
        Calculate a volume of the specified tetrahedron

        :param index: The tetrahedral index.
        :return: A volume of specified tetrahedron
        :rtype: double
        """
        if index < 0 or index >= self.tetrahedra_mv.shape[0]:
            raise ValueError('Tetrahedral index is out of range: [0, {}].'.format(self.tetrahedra_mv.shape[0]))

        return self._volume(index)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double volume_total(self):
        """
        Calculate a total volume of all tetrahedra

        :return: total volume of all tetrahedra
        :rtype: double
        """
        cdef:
            np.int32_t i
            double volume = 0.0

        for i in range(self.tetrahedra_mv.shape[0]):
            volume += self._volume(i)

        return volume

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double _volume(self, int index):
        """
        Fast calculation of volume of a tetrahedron
        """
        cdef:
            np.int32_t i1, i2, i3, i4
            Point3D p1, p2, p3, p4
            Vector3D v1, v2, v3, V4
            double area, height

        i1 = self.tetrahedra_mv[index, V1]
        i2 = self.tetrahedra_mv[index, V2]
        i3 = self.tetrahedra_mv[index, V3]
        i4 = self.tetrahedra_mv[index, V4]

        p1 = new_point3d(self.vertices_mv[i1, X], self.vertices_mv[i1, Y], self.vertices_mv[i1, Z])
        p2 = new_point3d(self.vertices_mv[i2, X], self.vertices_mv[i2, Y], self.vertices_mv[i2, Z])
        p3 = new_point3d(self.vertices_mv[i3, X], self.vertices_mv[i3, Y], self.vertices_mv[i3, Z])
        p4 = new_point3d(self.vertices_mv[i4, X], self.vertices_mv[i4, Y], self.vertices_mv[i4, Z])

        v1 = p1.vector_to(p2)
        v2 = p1.vector_to(p3)
        v3 = p1.vector_to(p4)

        # area of the base
        v4 = v1.cross(v2)
        area = v4.get_length() * 0.5

        # height from the base to the apex
        height = fabs(v4.normalise().dot(v3))

        return (area * height) / 3.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef object _filter_tetrahedra(self):

        cdef:
            np.int32_t i, valid

        # scan tetrahedra and make valid tetrahedra contiguous
        valid = 0
        for i in range(self.tetrahedra_mv.shape[0]):

            if self._volume(i) == 0.0:

                # tetrahedron is degenerate, skip
                continue

            # shift tetrahedra
            self.tetrahedra_mv[valid, :] = self.tetrahedra_mv[i, :]
            valid += 1

        # reslice array to contain only valid tetrahedra
        self.tetrahedra_mv = self.tetrahedra_mv[:valid, :]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef BoundingBox3D _generate_bounding_box(self, np.int32_t tetrahedra):
        """
        Generates a bounding box for the specified tetrahedron.

        A small degree of padding is added to the bounding box to provide the
        conservative bounds required by the watertight mesh algorithm.

        :param tetrahedra: tetrahedral array index.
        :return: A BoundingBox3D object.
        :rtype: BoundingBox3D
        """

        cdef:
            np.int32_t i1, i2, i3, i4
            BoundingBox3D bbox

        i1 = self.tetrahedra_mv[tetrahedra, V1]
        i2 = self.tetrahedra_mv[tetrahedra, V2]
        i3 = self.tetrahedra_mv[tetrahedra, V3]
        i4 = self.tetrahedra_mv[tetrahedra, V4]

        bbox = new_boundingbox3d(
            new_point3d(
                min(self.vertices_mv[i1, X], self.vertices_mv[i2, X], self.vertices_mv[i3, X], self.vertices_mv[i4, X]),
                min(self.vertices_mv[i1, Y], self.vertices_mv[i2, Y], self.vertices_mv[i3, Y], self.vertices_mv[i4, Y]),
                min(self.vertices_mv[i1, Z], self.vertices_mv[i2, Z], self.vertices_mv[i3, Z], self.vertices_mv[i4, Z]),
            ),
            new_point3d(
                max(self.vertices_mv[i1, X], self.vertices_mv[i2, X], self.vertices_mv[i3, X], self.vertices_mv[i4, X]),
                max(self.vertices_mv[i1, Y], self.vertices_mv[i2, Y], self.vertices_mv[i3, Y], self.vertices_mv[i4, Y]),
                max(self.vertices_mv[i1, Z], self.vertices_mv[i2, Z], self.vertices_mv[i3, Z], self.vertices_mv[i4, Z]),
            ),
        )

        # The bounding box and tetrahedral vertices may not align following coordinate
        # transforms in the water tight mesh algorithm, therefore a small bit of padding
        # is added to avoid numerical representation issues.
        bbox.pad(max(BOX_PADDING, bbox.largest_extent() * BOX_PADDING))

        return bbox

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef bint _is_contained_leaf(self, np.int32_t id, Point3D point):

        cdef:
            np.int32_t index, tetrahedra_id, i1, i2, i3, i4
            double alpha, beta, gamma, delta

        # identify the first tetrahedra that contains the point, if any
        for index in range(self._nodes[id].count):

            # obtain vertex indices
            tetrahedra_id = self._nodes[id].items[index]
            i1 = self.tetrahedra_mv[tetrahedra_id, V1]
            i2 = self.tetrahedra_mv[tetrahedra_id, V2]
            i3 = self.tetrahedra_mv[tetrahedra_id, V3]
            i4 = self.tetrahedra_mv[tetrahedra_id, V4]

            barycentric_coords_tetra(self.vertices_mv[i1, X], self.vertices_mv[i1, Y], self.vertices_mv[i1, Z],
                                     self.vertices_mv[i2, X], self.vertices_mv[i2, Y], self.vertices_mv[i2, Z],
                                     self.vertices_mv[i3, X], self.vertices_mv[i3, Y], self.vertices_mv[i3, Z],
                                     self.vertices_mv[i4, X], self.vertices_mv[i4, Y], self.vertices_mv[i4, Z],
                                     point.x, point.y, point.z, &alpha, &beta, &gamma, &delta)

            if barycentric_inside_tetrahedra(alpha, beta, gamma, delta):

                # store id of tetrahedral hit
                self.tetrahedra_id = tetrahedra_id

                # store vertex indices and barycentric coords
                self.i1 = i1
                self.i2 = i2
                self.i3 = i3
                self.i4 = i4
                self.alpha = alpha
                self.beta = beta
                self.gamma = gamma
                self.delta = delta

                return True

        return False

    cpdef bint is_contained(self, Point3D point):
        """
        Traverses the kd-Tree to identify if the point is contained by an item.
        :param Point3D point: A Point3D object.
        :return: True if the point lies inside an item, false otherwise.
        :rtype: bool
        """

        cdef bint result

        if self._cache_available and point.x == self._cached_x and point.y == self._cached_y and point.z == self._cached_z:
            return self._cached_result

        result = self._is_contained(point)

        # add cache
        self._cache_available = True
        self._cached_x = point.x
        self._cached_y = point.y
        self._cached_z = point.z
        self._cached_result = result

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef BoundingBox3D bounding_box(self, AffineMatrix3D to_world):
        """
        Returns a bounding box that encloses the mesh.

        The box is padded by a small margin to reduce the risk of numerical
        accuracy problems between the mesh and box representations following
        coordinate transforms.

        :param to_world: Local to world space transform matrix.
        :return: A BoundingBox3D object.
        """

        cdef:
            np.int32_t i
            BoundingBox3D bbox
            Point3D vertex

        # TODO: padding should really be a function of mesh extent
        # convert vertices to world space and grow a bounding box around them
        bbox = BoundingBox3D()
        for i in range(self.vertices_mv.shape[0]):
            vertex = new_point3d(self.vertices_mv[i, X], self.vertices_mv[i, Y], self.vertices_mv[i, Z])
            bbox.extend(vertex.transform(to_world), BOX_PADDING)

        return bbox

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def save(self, object file):
        """
        Save the mesh's kd-Tree representation to a binary Raysect mesh file (.rsm).

        :param object file: File stream or string file name to save state.
        """

        cdef:
            np.int32_t i, j
            double[:, ::1] vertices
            np.int32_t[:, ::1] tetrahedra

        close = False

        # treat as a filename if a stream is not supplied
        if not isinstance(file, io.IOBase):
            file = open(file, mode="wb")
            close = True

        # hold local references to avoid repeated memory view object checks
        vertices = self.vertices_mv
        tetrahedra = self.tetrahedra_mv

        # write header
        file.write(b"RSM")
        file.write(struct.pack("<B", RSM_VERSION_MAJOR))
        file.write(struct.pack("<B", RSM_VERSION_MINOR))

        # mesh setting flags
        file.write(struct.pack("<?", True))    # kdtree in file (hardcoded for now, will be an option)

        # item counts
        file.write(struct.pack("<i", vertices.shape[0]))
        file.write(struct.pack("<i", tetrahedra.shape[0]))

        # write vertices
        for i in range(vertices.shape[0]):
            for j in range(3):
                file.write(struct.pack("<d", vertices[i, j]))

        # tetrahedra
        for i in range(tetrahedra.shape[0]):
            for j in range(4):
                file.write(struct.pack("<i", tetrahedra[i, j]))

        # write kd-tree
        super().save(file)

        # if we opened a file, we should close it
        if close:
            file.close()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def load(self, object file):
        """
        Load a mesh with its kd-Tree representation from Raysect mesh binary file (.rsm).

        :param object file: File stream or string file name to save state.
        """

        cdef:
            np.int32_t i, j

        close = False

        # treat as a filename if a stream is not supplied
        if not isinstance(file, io.IOBase):
            file = open(file, mode="rb")
            close = True

        # read and check header
        identifier = file.read(3)
        major_version = self._read_uint8(file)
        minor_version = self._read_uint8(file)

        # validate
        if identifier != b"RSM":
            raise ValueError("Specified file is not a Raysect mesh file.")

        if major_version != RSM_VERSION_MAJOR or minor_version != RSM_VERSION_MINOR:
            raise ValueError("Unsupported Raysect mesh version.")

        # mesh setting flags
        _ = self._read_bool(file)    # kdtree option, ignore for now (to be implemented)

        # item counts
        num_vertices = self._read_int32(file)
        num_tetrahedra = self._read_int32(file)

        # read vertices
        self._vertices = np.zeros((num_vertices, 3), dtype=np.double)
        self.vertices_mv = self._vertices
        for i in range(num_vertices):
            for j in range(3):
                self.vertices_mv[i, j] = self._read_double(file)

        # read tetrahedra
        self._tetrahedra = np.zeros((num_tetrahedra, 4), dtype=np.int32)
        self.tetrahedra_mv = self._tetrahedra
        for i in range(num_tetrahedra):
            for j in range(4):
                self.tetrahedra_mv[i, j] = self._read_int32(file)

        # read kdtree
        super().load(file)

        # initialise hit state attributes
        self.tetrahedra_id = -1
        self.i1 = -1
        self.i2 = -1
        self.i3 = -1
        self.i4 = -1
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.delta = 0.0

        # initialise cache values
        self._cache_available = False
        self._cached_x = 0.0
        self._cached_y = 0.0
        self._cached_z = 0.0
        self._cached_result = False

        # if we opened a file, we should close it
        if close:
            file.close()

    @classmethod
    def from_file(cls, file):
        """
        Load a mesh with its kd-Tree representation from Raysect mesh binary file (.rsm).

        :param object file: File stream or string file name to save state.
        """

        m = TetraMesh.__new__(TetraMesh)
        m.load(file)
        return m

    cdef uint8_t _read_uint8(self, object file):
        return (<uint8_t *> PyBytes_AsString(file.read(sizeof(uint8_t))))[0]

    cdef bint _read_bool(self, object file):
        return self._read_uint8(file) != 0
