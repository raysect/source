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

"""
The following emitters and integrators are used in Regular Grid Volumes.
They allow fast integration along the ray's trajectory as they use pre-calculated
values of spectral emissivity on a regular grid.
Note that these emitters support other integrators as well, however high performance
with other integrators is not guaranteed.

Performance tips:

  * If dispesive rendering is off (`camera.spectral_rays = 1`) and spectral properties of
    rays do not change during rendering, consider calling:

    .. code-block:: pycon
        >>> emitter.build_cache(camera.min_wavelength, camera.max_wavelength,
                                camera.spectral_bins)

    before the first call of `camera.observe()`. This will save a lot of memory in case of
    multi-process rendering, as well as some time between the calls of `camera.observe()`.

  * In case of insufficient memory, one can initialise the emitter with a dummy emission
    array and then populate the cache with a pre-calculated `csr_matrix`.

    .. code-block:: pycon
        >>> grid_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
        >>> wavelengths = np.ones(1)
        >>> emission = csc_matrix((grid_size, 1))
        >>> emitter = CartesianRegularEmitter(grid_shape, grid_steps, emission,
                                              wavelengths)
        >>> emitter.cache_override(cache, camera.min_wavelength, camera.max_wavelength)

    Note that `cache.shape` must be equal to `(grid_size, camera.spectral_bins)`.
    This solution will work only if dispersive rendering is off (`camera.spectral_rays = 1`)
    and spectral properties of rays do not change during rendering.

  * If the emission spectrum has some regions where the material does not emit, the emission should be
    explicitly zeroed at the borders of these regions. Otherwise, the spectrum will be interpolated
    between these regions during the sampling and the cache will be filled with negligable in terms of
    physics but non-zero values. This will not only increase memory consumption but also the computing time.
"""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material cimport VolumeIntegrator, InhomogeneousVolumeEmitter
from libc.math cimport sqrt, atan2, M_PI as pi
from .regular_grid_utility cimport integrate_contineous, integrate_delta_function
cimport numpy as np
cimport cython


cdef class RegularGridIntegrator(VolumeIntegrator):
    """
    Basic class for regular grid integrators.

    :param float step: Integration step (in meters), defaults to `step=0.001`.
    :param int min_samples: The minimum number of samples to use over integration range,
        defaults to `min_samples=2`.

    :ivar float step: Integration step.
    :ivar int min_samples: The minimum number of samples to use over integration range.
    """

    def __init__(self, double step=0.001, int min_samples=2):
        self.step = step
        self.min_samples = min_samples

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        if value <= 0:
            raise ValueError("Numerical integration step size can not be less than or equal to zero.")
        self._step = value

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, value):
        if value < 2:
            raise ValueError("At least two samples are required to perform the numerical integration.")
        self._min_samples = value


cdef class CylindricalRegularIntegrator(RegularGridIntegrator):
    """
    Integrates the spectral emissivity defined on a regular grid
    in cylindrical coordinate system: :math:`(R, \phi, Z)` along the ray's trajectory.
    This integrator must be used with the `CylindricalRegularEmitter` material class. 
    It is assumed that the emitter is periodic in :math:`\phi` direction with a period
    equal to `material.period`.
    This integrator does not perform interpolation, so the spectral emissivity at
    any spatial point along the ray's trajectory is equal to that of the grid cell
    where this point is located.
    """

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D direction
            int it, ir, iphi, iz, ir_current, iphi_current, iz_current, n
            double length, t, dt, x, y, z, r, phi, ray_path
            CylindricalRegularEmitter emitter

        if not isinstance(material, CylindricalRegularEmitter):
            raise TypeError('Only CylindricalRegularEmitter material is supported by CylindricalRegularIntegrator.')

        emitter = material
        # Building the cache if required
        emitter.cache_build(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_bins())

        if emitter.cache_empty():  # emitter does not emit at this wavelength range
            return spectrum

        # Determining direction of integration and effective integration step
        start = start_point.transform(world_to_primitive)  # start point in local coordinates
        end = end_point.transform(world_to_primitive)  # end point in local coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.get_length()  # integration length
        if length < 0.1 * self._step:  # return if ray's path is too short
            return spectrum

        direction = direction.normalise()  # normalized direction
        n = max(self._min_samples, <int>(length / self._step))  # number of points along ray's trajectory
        dt = length / n  # integration step

        # Starting integration
        ir_current = 0
        iphi_current = 0
        iz_current = 0
        ray_path = 0
        for it in range(n):
            t = (it + 0.5) * dt
            x = start.x + direction.x * t  # x coordinates of the points
            y = start.y + direction.y * t  # y coordinates of the points
            z = start.z + direction.z * t  # z coordinates of the points
            iz = <int>(z / emitter.dz)  # Z-indices of grid cells, in which the points are located
            r = sqrt(x * x + y * y)  # R coordinates of the points
            ir = <int>((r - emitter.rmin) / emitter.dr)  # R-indices of grid cells, in which the points are located
            if emitter.nphi == 1:  # axisymmetric case
                iphi = 0
            else:
                phi = (180. / pi) * atan2(y, x)  # phi coordinates of the points (in the range [-180, 180))
                phi = (phi + 360) % emitter.period  # moving into the [0, period) sector (periodic emitter)
                iphi = <int>(phi / emitter.dphi)  # phi-indices of grid cells, in which the points are located
            if ir != ir_current or iz != iz_current or iphi != iphi_current:  # we moved to the next cell
                emitter.add_emission_to_mv(spectrum.samples_mv, ir_current, iphi_current, iz_current, ray_path)
                ir_current = ir
                iphi_current = iphi
                iz_current = iz
                ray_path = 0  # zeroing ray's path along the cell
            ray_path += dt
        emitter.add_emission_to_mv(spectrum.samples_mv, ir_current, iphi_current, iz_current, ray_path)

        return spectrum


cdef class CartesianRegularIntegrator(RegularGridIntegrator):
    """
    Integrates the spectral emissivity defined on a regular grid
    in Cartesian coordinate system: :math:`(X, Y, Z)` along the ray's trajectory.
    This integrator must be used with the `CartesianRegularEmitter` material class. 
    This integrator does not perform interpolation, so the spectral emissivity at
    any spatial point along the ray's trajectory is equal to that of the grid cell
    where this point is located.
    """

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            Point3D start, end
            Vector3D direction
            int it, ix, iy, iz, ix_current, iy_current, iz_current, n
            double length, t, dt, x, y, z, ray_path
            CartesianRegularEmitter emitter

        if not isinstance(material, CartesianRegularEmitter):
            raise TypeError('Only CartesianRegularEmitter is supported by CartesianRegularIntegrator')

        emitter = material
        # Building the cache if required
        emitter.cache_build(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_bins())

        if emitter.cache_empty():  # material does not emit at this wavelength range
            return spectrum

        # Determining direction of integration and effective integration step
        start = start_point.transform(world_to_primitive)  # start point in local coordinates
        end = end_point.transform(world_to_primitive)  # end point in local coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.get_length()  # integration length
        if length < 0.1 * self._step:  # return if ray's path is too short
            return spectrum
        direction = direction.normalise()  # normalized direction
        n = max(self._min_samples, <int>(length / self._step))  # number of points along ray's trajectory
        dt = length / n  # integration step

        # Starting integrations
        ix_current = 0
        iy_current = 0
        iz_current = 0
        ray_path = 0
        for it in range(n):
            t = (it + 0.5) * dt
            x = start.x + direction.x * t  # x coordinates of the points
            y = start.y + direction.y * t  # y coordinates of the points
            z = start.z + direction.z * t  # z coordinates of the points
            ix = <int>(x / emitter.dx)  # X-indices of grid cells, in which the points are located
            iy = <int>(y / emitter.dy)  # Y-indices of grid cells, in which the points are located
            iz = <int>(z / emitter.dz)  # Z-indices of grid cells, in which the points are located
            if ix != ix_current or iy != iy_current or iz != iz_current:  # we moved to the next cell
                emitter.add_emission_to_mv(spectrum.samples_mv, ix_current, iy_current, iz_current, ray_path)
                ix_current = ix
                iy_current = iy
                iz_current = iz
                ray_path = 0  # zeroing ray's path along the cell
            ray_path += dt
        emitter.add_emission_to_mv(spectrum.samples_mv, ix_current, iy_current, iz_current, ray_path)

        return spectrum


cdef class RegularGridEmitter(InhomogeneousVolumeEmitter):
    """
    Basic class for the emitters defined on a regular 3D grid.
    The emission anywhere outside the specified grid is zero.

    :param tuple grid_shape: The number of grid cells along each direction.
    :param tuple grid_steps: The sizes of grid cells along each direction.
    :param object ~.emission: The 2D or 4D array or scipy sparse matrix containing the
        emission defined on a regular 3D grid in :math:`W/(str\,m^3\,nm)` (contineous
        spectrum) or in :math:`W/(str\,m^3)` (discrete spectrum).
        Spectral emission can be provided either for selected cells of the regular
        grid (2D array or sparse matrix) or for all grid cells (4D array).
        Note that if provided as a 2D array (or sparse matrix), the spatial index `(i, j, k)`
        must be flattened in a row-major order:
        `iflat = grid_shape[1] * grid_shape[2] * i + grid_shape[2] * j + k`.
        Regardless of the form in which the emission is provided, the last axis is the
        spectral one.  The emission will be stored as a сompressed sparse column matrix
        (`scipy.sparse.csc_matrix`). To reduce memory consumption, provide it as a `csc_matrix`.
    :param ndarray wavelengths: The 1D array of wavelengths corresponding to the last axis of
        provided emission array. The size of this array must be equal to `emission.shape[-1]`.
        Initialisation will be faster if this array contains monotonically increasing values.
    :param bool contineous: Defines whether the emission is porvided as a contineous spectrum
        (in :math:`W/(str\,m^3\,nm)`) or as a discrete spectrum (in :math:`W/(str\,m^3)`).
        Defaults to `contineous=True`.
    :param bool extrapolate: If True, the emission outside the provided spectral range
        will be equal to the emission at the borders of this range (nearest-neighbour
        extrapolation), otherwise it will be zero. Defaults to `extrapolate=True`.
        This parameter is ignored if `contineous=False`.
    :param bool cache_32bit: If True, the cached data will be stored in float32 instead of
        float64, thus saving the memory.
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator,
        defaults to `integrator=NumericalIntegrator()`.

    :ivar tuple grid_shape: The shape of regular grid.
    :ivar tuple grid_steps: The sizes of grid cells along each direction.
    :ivar csc_matrix ~.emission: The emission defined on a regular grid stored as a a сompressed
        sparse column matrix (`scipy.sparse.csc_matrix`).
    :ivar ndarray wavelengths: The sorted wavelengths corresponding to the emission array.
    :ivar bool contineous: Defines whether the emission is porvided as a contineous spectrum
        (in :math:`W/(str\,m^3\,nm)`) or as a discrete spectrum (in :math:`W/(str\,m^3)`).
    :ivar bool cache_32bit: Defines whether the cached data is stored in float32 (True) or
        float64 (False).
    :ivar bool extrapolate: Defines whether the emission spectrum is extrapolated outside the
        provided wavelength range (`True`) or not (`False`). Ignored if `contineous=False`.
    :ivar int nvoxel: Total number of grid cells in the spatial grid.
    """

    def __init__(self, tuple grid_shape, tuple grid_steps, object emission, np.ndarray wavelengths,
                 bint contineous=True, bint extrapolate=True, bint cache_32bit=False, VolumeIntegrator integrator=None):

        cdef:
            np.ndarray indx_sorted
            double step
            int i

        for step in grid_steps:
            if step <= 0:
                raise ValueError('Grid steps must be > 0.')
        self._grid_steps = grid_steps

        for i in grid_shape:
            if i <= 0:
                raise ValueError('Grid sizes must be > 0.')

        self.nvoxel = grid_shape[0] * grid_shape[1] * grid_shape[2]  # nvoxel is int64

        self._grid_shape = grid_shape

        if emission.ndim == 2:
            if emission.shape[0] != self.nvoxel:
                raise ValueError("The number of rows in 'emission' array does not match the grid size.")
            self.emission = csc_matrix(emission)  # this does not create a copy if emission is already a csc_matrix

        elif emission.ndim == 4:
            if emission.shape[0] != self._grid_shape[0] or emission.shape[1] != self._grid_shape[1] or emission.shape[2] != self._grid_shape[2]:
                raise ValueError("The shape of 'emission' array does not match the grid shape.")
            self.emission = csc_matrix(emission.reshape(self.nvoxel, emission.shape[3]))

        else:
            raise ValueError("Argument 'emission' must be a 4D or 2D array.")

        if wavelengths.size != self.emission.shape[1]:
            raise ValueError("The size of 'wavelengths' array does not match 'emission.shape[-1]'.")
        if np.any(wavelengths < 0):
            raise ValueError("Wavelengths must be >= 0.")

        if np.any(np.diff(wavelengths) < 0):  # sorting the arrays if required
            indx_sorted = np.argsort(wavelengths)
            self.wavelengths = wavelengths[indx_sorted].astype(np.float64)
            self.emission = self.emission[:, indx_sorted]
        else:
            self.wavelengths = wavelengths.astype(np.float64)

        self._wavelengths_mv = self.wavelengths

        self.contineous = contineous
        self._extrapolate = extrapolate if self.contineous else False
        self._cache_32bit = True if emission.dtype == np.float32 else cache_32bit

        self._cache_init()

        super().__init__(integrator)

    @property
    def extrapolate(self):
        return self._extrapolate

    @extrapolate.setter
    def extrapolate(self, bint value):
        self._extrapolate = value if self.contineous else False

    @property
    def cache_32bit(self):
        return self._cache_32bit

    @cache_32bit.setter
    def cache_32bit(self, bint value):
        if self._cache_32bit != value:
            self._cache_init()
        self._cache_32bit = value

    @cython.nonecheck(False)
    cdef long get_voxel_index(self, int i, int j, int k) nogil:
        """
        Returns a flattened voxel index for provided i, j, k values.
        """

        if i < 0 or i >= self._grid_shape[0] or j < 0 or j >= self._grid_shape[1] or k < 0 or k >= self._grid_shape[2]:
            return -1  # out of grid

        return <long>(i) * self._grid_shape[1] * self._grid_shape[2] + <long>(j) * self._grid_shape[2] + <long>(k)

    cpdef long voxel_index(self, int i, int j, int k):
        """
        Returns a flattened voxel index for provided i, j, k values.
        """

        return self.get_voxel_index(i, j, k)

    cdef void _cache_init(self):
        """
        Initialises the cache.
        """

        # initialise cache with invalid values
        self._cache = None
        self.cache_data_32_mv = None
        self.cache_indptr_32_mv = None
        self.cache_indices_32_mv = None
        self.cache_data_64_mv = None
        self.cache_indptr_64_mv = None
        self.cache_indices_64_mv = None
        self.cache_32bit_indices = True  # unlike _cache_32bit, this parameter is not controlled by user
        self._cache_data_size = -1
        self._cache_min_wvl = -1
        self._cache_max_wvl = -1
        self._cache_num_samp = -1

    cpdef bint cache_valid(self, double min_wavelength, double max_wavelength, int bins):
        """
        Returns true if a suitable cached data are available.

        :param float min_wavelength: The minimum wavelength in nanometers.
        :param float max_wavelength: The maximum wavelength in nanometers.
        :param int bins: The number of spectral bins.
        """

        return (
            self._cache_min_wvl == min_wavelength and
            self._cache_max_wvl == max_wavelength and
            self._cache_num_samp == bins
        )

    cpdef bint cache_empty(self):
        """
        Returns true if the cached data does not contain non-zero elements or the cache is not
        built.
        """

        return self._cache_data_size <= 0

    cpdef void cache_override(self, object cache, double min_wavelength, double max_wavelength):
        """
        Overrides the cache with the provided compressed sparse row matrix.

        :param csr_matrix cache: The cache pre-calculated for the spectral properties of rays.
        :param float min_wavelength: The minimum wavelength in nanometers.
        :param float max_wavelength: The maximum wavelength in nanometers.

        Use this in case of insufficient memory.
        .. code-block:: pycon
          >>> grid_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
          >>> wavelengths = np.ones(1)
          >>> emission = csc_matrix((grid_size, 1))
          >>> emitter = RegularGridEmitter(grid_shape, grid_steps, emission, wavelengths)
          >>> emitter.cache_override(cache, camera.min_wavelength, camera.max_wavelength)

        Note that `cache.shape` must be equal to `(nvoxel, camera.spectral_bins)`.
        Overriden cache remains valid until ray spectral properties change.
        Cache overriding is useless in case of dispesive rendering (`camera.spectral_rays != 1`).
        """

        if cache.shape[0] != self.nvoxel:
            raise ValueError('Provided cache matrix does not match the grid size.')

        self._cache_init()  # deleting current cache

        self._cache = csr_matrix(cache)  # this does not create a copy if the cache is a csr_matrix

        self.cache_32bit_indices = (self._cache.indptr.dtype == np.int32)

        if self._cache.data.dtype == np.float32:
            self._cache_32bit = True
        elif self._cache.data.dtype == np.float64:
            self._cache_32bit = False
        else:
            # using current value of self._cache_32bit
            self._cache.data = self._cache.data.astype(np.float32) if self._cache_32bit else self._cache.data.astype(np.float64)

        # initialising memoryviews
        if self._cache_32bit:
            self.cache_data_32_mv = self._cache.data
        else:
            self.cache_data_64_mv = self._cache.data

        if self.cache_32bit_indices:
            self.cache_indptr_32_mv = self._cache.indptr
            self.cache_indices_32_mv = self._cache.indices
        else:
            self.cache_indptr_64_mv = self._cache.indptr
            self.cache_indices_64_mv = self._cache.indices

        self._cache_data_size = self._cache.data.size
        self._cache_min_wvl = min_wavelength
        self._cache_max_wvl = max_wavelength
        self._cache_num_samp = self._cache.shape[1]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef void cache_build(self, double min_wavelength, double max_wavelength, int bins, bint forced=False):
        """
        Builds a new cache if the old one does not match the wavelength range.

        :param float min_wavelength: The minimum wavelength in nanometers.
        :param float max_wavelength: The maximum wavelength in nanometers.
        :param int bins: The number of spectral bins.
        :param bool forces: Rebuild the cache even if the old cache matches the wavelength
            range, defaults to `forced=False`
        """

        cdef:
            object bin_integral
            np.ndarray data, row_ind, col_inds
            double delta, lower, upper
            int i

        if (not forced) and self.cache_valid(min_wavelength, max_wavelength, bins):
            return

        self._cache_init()  # deleting current cache

        # reducing memory usage if possible
        dtype = np.float32 if self._cache_32bit else np.float64
        dtype_int = np.int32 if self.nvoxel < np.iinfo('int32').max else np.int64

        data = np.array([], dtype=dtype)
        row_ind = np.array([], dtype=dtype_int)
        col_ind = np.array([], dtype=dtype_int)
        delta = (max_wavelength - min_wavelength) / bins
        lower = min_wavelength
        for i in range(bins):
            upper = min_wavelength + (i + 1) * delta
            bin_integral = self.integrate(lower, upper)
            data = np.concatenate((data, (bin_integral.data / delta).astype(dtype)))
            col_ind = np.concatenate((col_ind, i * np.ones(bin_integral.data.size, dtype=dtype_int)))
            row_ind = np.concatenate((row_ind, bin_integral.indices))
            lower = upper

        self._cache = csr_matrix((data, (row_ind, col_ind)), shape=(self.nvoxel, bins), dtype=dtype)

        self.cache_32bit_indices = (self._cache.indptr.dtype == np.int32)  # depends on the data size, may not be equal to dtype_int

        if self._cache_32bit:
            self.cache_data_32_mv = self._cache.data
        else:
            self.cache_data_64_mv = self._cache.data

        if self.cache_32bit_indices:
            self.cache_indptr_32_mv = self._cache.indptr
            self.cache_indices_32_mv = self._cache.indices
        else:
            self.cache_indptr_64_mv = self._cache.indptr
            self.cache_indices_64_mv = self._cache.indices

        self._cache_data_size = self._cache.data.size
        self._cache_min_wvl = min_wavelength
        self._cache_max_wvl = max_wavelength
        self._cache_num_samp = bins

    cpdef object integrate(self, double min_wavelength, double max_wavelength):
        """
        Integrate the emission in the specified wavelength range and returns the result in the
        form of one-column `csc_matrix`.

        :param float min_wavelength: The minimum wavelength in nanometers.
        :param float max_wavelength: The maximum wavelength in nanometers.

        :return: Integrated emission in :math:`W/(str\,m^3)`.
        """

        if self.contineous:
            return integrate_contineous(self._wavelengths_mv, self.emission, min_wavelength, max_wavelength, self._extrapolate)
        else:
            return integrate_delta_function(self._wavelengths_mv, self.emission, min_wavelength, max_wavelength)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cdef void add_emission_to_mv(self, double[::1] samples_mv, int i, int j, int k, double ray_path) nogil:
        """
        Adds to the provided memoryview the spectral emission of the specified spatial cell,
        multiplied by the provided distance, travelled by ray through the cell.

        This function does not check the cache spectral range.

        :param double[::1] samples_mv: Memoryview of the array with ray.bins elements.
        :param int i: 1st index of the grid.
        :param int j: 2nd index of the grid.
        :param int k: 3rd index of the grid.
        :param double ray_path: Distance, travelled by ray through the cell.
        """

        cdef:
            int ivoxel32, i32
            long ivoxel64, i64

        if samples_mv.shape[0] != self._cache_num_samp:
            raise ValueError('Provided array does not match the cache size.')

        if i < 0 or i >= self._grid_shape[0] or j < 0 or j >= self._grid_shape[1] or k < 0 or k >= self._grid_shape[2]:
            # out of spatial grid
            return

        if self.cache_32bit_indices:

            ivoxel32 = i * self._grid_shape[1] * self._grid_shape[2] + j * self._grid_shape[2] + k

            if self._cache_32bit:
                for i32 in range(self.cache_indptr_32_mv[ivoxel32], self.cache_indptr_32_mv[ivoxel32 + 1]):
                    samples_mv[self.cache_indices_32_mv[i32]] += ray_path * self.cache_data_32_mv[i32]
            else:
                for i32 in range(self.cache_indptr_32_mv[ivoxel32], self.cache_indptr_32_mv[ivoxel32 + 1]):
                    samples_mv[self.cache_indices_32_mv[i32]] += ray_path * self.cache_data_64_mv[i32]

        else:

            ivoxel64 = <long>(i) * self._grid_shape[1] * self._grid_shape[2] + <long>(j) * self._grid_shape[2] + <long>(k)

            if self._cache_32bit:
                for i64 in range(self.cache_indptr_64_mv[ivoxel64], self.cache_indptr_64_mv[ivoxel64 + 1]):
                    samples_mv[self.cache_indices_64_mv[i64]] += ray_path * self.cache_data_32_mv[i64]
            else:
                for i64 in range(self.cache_indptr_64_mv[ivoxel64], self.cache_indptr_64_mv[ivoxel64 + 1]):
                    samples_mv[self.cache_indices_64_mv[i64]] += ray_path * self.cache_data_64_mv[i64]

    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef void add_emission_to_array(self, np.ndarray samples, int i, int j, int k, double ray_path):
        """
        Adds to the provided memoryview the spectral emission of the specified spatial cell,
        multiplied by the provided distance, travelled by ray through the cell.

        This function does not check the cache spectral range.

        :param double[::1] samples_mv: Memoryview of the array with ray.bins elements.
        :param int i: 1st index of the grid.
        :param int j: 2nd index of the grid.
        :param int k: 3rd index of the grid.
        :param double ray_path: Distance, travelled by ray through the cell.
        """

        cdef:
            double[::1] samples_mv

        samples_mv = samples
        self.add_emission_to_mv(samples_mv, i, j, k, ray_path)


cdef class CylindricalRegularEmitter(RegularGridEmitter):

    """
    Spectral emitter defined on a regular 3D grid in cylindrical
    coordinates: :math:`(R, \phi, Z)`. This emitter is periodic in :math:`\phi` direction.
    The emission enywhere outsode the provided spatial grid is zero.

    :param tuple grid_shape: The number of grid cells along each direction.
    :param tuple grid_steps: The sizes of grid cells along each direction.
    :param object ~.emission: The 2D or 4D array or scipy sparse matrix containing the
        emission defined on a regular :math:`(R, \phi, Z)` grid in :math:`W/(str\,m^3\,nm)`
        (contineous spectrum) or in :math:`W/(str\,m^3)` (discrete spectrum).
        Spectral emission can be provided either for selected cells of the regular
        grid (2D array or sparse matrix) or for all grid cells (4D array).
        Note that if provided as a 2D array (or sparse matrix), the spatial index `(ir, iphi, iz)`
        must be flattened in a row-major order:
        `iflat = grid_shape[1] * grid_shape[2] * ir + grid_shape[2] * iphi + iz`.
        Regardless of the form in which the emission is provided, the last axis is the
        spectral one.  The emission will be stored as a сompressed sparse column matrix
        (`scipy.sparse.csc_matrix`). To reduce memory consumption, provide it as a `csc_matrix`.
    :param ndarray wavelengths: The 1D array of wavelengths corresponding to the last axis of
        provided emission array. The size of this array must be equal to `emission.shape[-1]`.
        Initialisation will be faster if this array contains monotonically increasing values.
    :param bool contineous: Defines whether the emission is porvided as a contineous spectrum
        (in :math:`W/(str\,m^3\,nm)`) or as a discrete spectrum (in :math:`W/(str\,m^3)`).
        Defaults to `contineous=True`.
    :param bool extrapolate: If True, the emission outside the provided spectral range
        will be equal to the emission at the borders of this range (nearest-neighbour
        extrapolation), otherwise it will be zero. Defaults to `extrapolate=True`.
        This parameter is ignored if `contineous=False`.
    :param bool cache_32bit: If True, the cached data will be stored in float32 instead of
        float64, thus saving the memory.
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator, defaults to
        `CylindricalRegularIntegrator(step=0.25 * min(grid_steps[0], grid_steps[2]))`.
    :param float rmin: Lower bound of grid in `R` direction (in meters), defaults to `rmin=0`.
    
    :ivar tuple grid_shape: The shape of regular grid.
    :ivar tuple grid_steps: The sizes of grid cells along each direction.
    :ivar csc_matrix ~.emission: The emission defined on a regular grid stored as a a сompressed
        sparse column matrix (`scipy.sparse.csc_matrix`).
    :ivar ndarray wavelengths: The sorted wavelengths corresponding to the emission array.
    :ivar int nvoxel: Total number of grid cells in the spatial grid.
    :ivar bool contineous: Defines whether the emission is porvided as a contineous spectrum
        (in :math:`W/(str\,m^3\,nm)`) or as a discrete spectrum (in :math:`W/(str\,m^3)`).
    :ivar bool extrapolate: Defines whether the emission spectrum is extrapolated outside the
        provided wavelength range (`True`) or not (`False`).
    :ivar bool cache_32bit: Defines whether the cached data is stored in float32 (True) or
        float64 (False).
    :ivar float period: The period in :math:`\phi` direction (equals to
        `grid_shape[1] * grid_steps[1]`.
    :ivar float rmin: Lower bound of grid in `R` direction.
    :ivar float dr: The size of grid cell in `R` direction (equals to `grid_steps[0]`).
    :ivar float dphi: The size of grid cell in :math:`\phi` direction (equals to `grid_steps[1]`).
    :ivar float dz: The size of grid cell in `Z` direction (equals to `grid_steps[2]`). 

    Continoues spectrum example:   

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import World, translate, rotate
        >>> from raysect.primitive import Cylinder, Subtract
        >>> from raysect.optical.material import CylindricalRegularEmitter
        >>> ### Contineous case ###
        >>> # grid parameters
        >>> rmin = 0
        >>> rmax = 2.
        >>> zmin = -0.25
        >>> zmax = 0.25
        >>> r, dr = np.linspace(rmin, rmax, 201, retstep=True)
        >>> r = r[:-1] + 0.5 * dr  # moving to the grid cell centers
        >>> grid_shape = (200, 1, 1)
        >>> grid_steps = (dr, 360., zmax - zmin)
        >>> integration_step = 0.05
        >>> # spectral emission profile
        >>> min_wavelength = 375.
        >>> max_wavelength = 740.
        >>> wavelengths, delta_wavelength = np.linspace(min_wavelength, max_wavelength, 50,
                                                        retstep=True)
        >>> wvl_centre = 0.5 * (max_wavelength + min_wavelength)
        >>> wvl_range = min_wavelength - max_wavelength
        >>> shift = 2 * (wavelengths - wvl_centre) / wvl_range + 5.
        >>> emission = np.cos(shift[None, None, None, :] * radius[:, None, None, None])**4
        >>> # scene
        >>> world = World()
        >>> material = CylindricalRegularEmitter(grid_shape, grid_steps, emission,
                                                 wavelengths, rmin=rmin)
        >>> bounding_box = Subtract(Cylinder(rmax, zmax - zmin), Cylinder(rmin, zmax - zmin),
                                    material=material, parent=world)  # bounding primitive
        >>> bounding_box.transform = translate(-rmax, -rmax + 1., zmin) * rotate(30, 0, 0)
        ...
        >>> # if ray spectral properties do not change during the rendering,
        >>> # build the cache before the first camera.observe() call to reduce memory consumptions
        >>> material.cache_build(camera.min_wavelength, camera.max_wavelength,
                                 camera.spectral_bins)

    Discrete spectrum example:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import World, translate
        >>> from raysect.optical.observer import SpectralRadiancePipeline2D
        >>> from raysect.primitive import Cylinder, Subtract
        >>> from raysect.optical.material import CylindricalRegularEmitter
        >>> # Assume that the files 'Be_4574A.npy' and 'Be_527A.npy' contain the emissions
        >>> # (in W / m^3) of Be I (3d1 1D2 -> 2p1 1P1) and Be II (4s1 2S0.5 -> 3p1 2P2.5)
        >>> # spectral lines defined on a regular cylindrical grid: 3.5 m < R < 9 m,
        >>> # 0 < phi < 20 deg, -5 m < Z < 5 m, and periodic in phi direction.
        >>> emission_4574 = np.load('Be_4574A.npy')
        >>> emission_5272 = np.load('Be_5272A.npy')
        >>> wavelengths = np.array([457.4, 527.2])
        >>> # Grid properties
        >>> rmin = 3.5
        >>> rmax = 9.
        >>> phi_period = 20.
        >>> zmin = -5.
        >>> zmax = 5.
        >>> grid_shape = emission_4574.shape
        >>> grid_steps = ((rmax - rmin) / grid_shape[0],
                          phi_period / grid_shape[1],
                          (zmax - zmin) / grid_shape[2])
        >>> emission = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], 2))
        >>> emission[:, :, :, 0] = emission_4574 / (4. * np.pi)  # to W/(m^3 str)
        >>> emission[:, :, :, 1] = emission_5272 / (4. * np.pi)
        >>> # Creating the scene
        >>> world = World()
        >>> pipeline = SpectralRadiancePipeline2D()
        >>> material = CylindricalRegularEmitter(grid_shape, grid_steps, memission,
                                                 wavelengths, rmin=rmin, contineous=False)
        >>> bounding_box = Subtract(Cylinder(rmax, zmax - zmin),
                                    Cylinder(rmin, zmax - zmin),
                                    material=material, parent=world)  # bounding primitive
        >>> bounding_box.transform = translate(0, 0, zmin)
        ...
        >>> camera.spectral_bins = 15
        >>> camera.min_wavelength = 457.
        >>> camera.max_wavelength = 528.
        >>> delta_wavelength = (camera.max_wavelength - camera.min_wavelength)/camera.spectral_bins
        >>> # if ray spectral properties do not change during the rendering,
        >>> # build the cache before the first camera.observe() call to reduce memory consumptions
        >>> material.cache_build(camera.min_wavelength, camera.max_wavelength,
                                 camera.spectral_bins)
        ...
        >>> # If reflections do not change the wavelength, the results for each spectral line
        >>> # can be obtained in W/(m^2 str) in the following way.
        >>> radiance_4574 = pipeline.frame.mean[:, :, 0] * delta_wavelength
        >>> radiance_5272 = pipeline.frame.mean[:, :, -1] * delta_wavelength
    """

    def __init__(self, tuple grid_shape, tuple grid_steps, object emission, np.ndarray wavelengths,
                 bint contineous=True, bint extrapolate=True, bint cache_32bit=False, VolumeIntegrator integrator=None, double rmin=0):

        if rmin < 0:
            raise ValueError("Attribute 'rmin' must be >= 0.")

        integrator = integrator or CylindricalRegularIntegrator(0.25 * min(grid_steps[0], grid_steps[2]))
        super().__init__(grid_shape, grid_steps, emission, wavelengths, contineous=contineous, extrapolate=extrapolate,
                         cache_32bit=cache_32bit, integrator=integrator)

        cdef:
            double period

        self.rmin = rmin

        self.dr = self._grid_steps[0]
        self.dphi = self._grid_steps[1]
        self.dz = self._grid_steps[2]

        self.nr = self._grid_shape[0]
        self.nphi = self._grid_shape[1]
        self.nz = self._grid_shape[2]

        period = self.nphi * self.dphi
        if 360. % period > 1.e-3:
            raise ValueError("The period %.3f (grid_shape[1] * grid_steps[1]) is not a multiple of 360." % period)
        self.period = period

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            int ir, iphi, iz
            double r, phi, delta_wavelength

        # Building the cache if required
        self.cache_build(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_bins())

        if self.cache_empty():  # emitter does not emit at this wavelength range
            return spectrum

        # Obtaining the index of the grid cell, where the point is located
        iz = <int>(point.z / self.dz)  # Z-index of grid cell, in which the point is located
        r = sqrt(point.x * point.x + point.y * point.y)  # R coordinates of the points
        ir = <int>((r - self.rmin) / self.dr)  # R-index of grid cell, in which the points is located
        if self.nphi == 1:  # axisymmetric case
            iphi = 0
        else:
            phi = (180. / pi) * atan2(point.y, point.x)  # phi coordinates of the points (in the range [-180, 180))
            phi = (phi + 360) % self.period  # moving into the [0, period) sector (periodic emitter)
            iphi = <int>(phi / self.dphi)  # phi-index of grid cell, in which the point is located
        self.add_emission_to_mv(spectrum.samples_mv, ir, iphi, iz, 1.0)

        return spectrum


cdef class CartesianRegularEmitter(RegularGridEmitter):

    """
    Spectral emitter defined on a regular 3D grid in Cartesian
    coordinates. The emission enywhere outsode the provided spatial grid is zero.

    :param tuple grid_shape: The number of grid cells along each direction.
    :param tuple grid_steps: The sizes of grid cells along each direction.
    :param object ~.emission: The 2D or 4D array or scipy sparse matrix containing the
        emission defined on a regular :math:`(X, Y, Z)` grid in :math:`W/(str\,m^3\,nm)`
        (contineous spectrum) or in :math:`W/(str\,m^3)` (discrete spectrum).
        Spectral emission can be provided either for selected cells of the regular
        grid (2D array or sparse matrix) or for all grid cells (4D array).
        Note that if provided as a 2D array (or sparse matrix), the spatial index `(ix, iy, iz)`
        must be flattened in a row-major order:
        `iflat = grid_shape[1] * grid_shape[2] * ix + grid_shape[2] * iy + iz`.
        Regardless of the form in which the emission is provided, the last axis is the
        spectral one.  The emission will be stored as a сompressed sparse column matrix
        (`scipy.sparse.csc_matrix`). To reduce memory consumption, provide it as a `csc_matrix`.
    :param ndarray wavelengths: The 1D array of wavelengths corresponding to the last axis of
        provided emission array. The size of this array must be equal to `emission.shape[-1]`.
        Initialisation will be faster if this array contains monotonically increasing values.
    :param bool contineous: Defines whether the emission is porvided as a contineous spectrum
        (in :math:`W/(str\,m^3\,nm)`) or as a discrete spectrum (in :math:`W/(str\,m^3)`).
        Defaults to `contineous=True`.
    :param bool extrapolate: If True, the emission outside the provided spectral range
        will be equal to the emission at the borders of this range (nearest-neighbour
        extrapolation), otherwise it will be zero. Defaults to `extrapolate=True`.
        This parameter is ignored if `contineous=False`.
    :param bool cache_32bit: If True, the cached data will be stored in float32 instead of
        float64, thus saving the memory.
    :param raysect.optical.material.VolumeIntegrator integrator: Volume integrator, defaults to
        `CartesianRegularIntegrator(step=0.25 * min(grid_steps))`.

    :ivar tuple grid_shape: The shape of regular grid.
    :ivar tuple grid_steps: The sizes of grid cells along each direction.
    :ivar csc_matrix ~.emission: The emission defined on a regular grid stored as a a сompressed
        sparse column matrix (`scipy.sparse.csc_matrix`).
    :ivar ndarray wavelengths: The sorted wavelengths corresponding to the emission array.
    :ivar int nvoxel: Total number of grid cells in the spatial grid.
    :ivar bool contineous: Defines whether the emission is porvided as a contineous spectrum
        (in :math:`W/(str\,m^3\,nm)`) or as a discrete spectrum (in :math:`W/(str\,m^3)`).
    :ivar bool extrapolate: Defines whether the emission spectrum is extrapolated outside the
        provided wavelength range (`True`) or not (`False`).
    :ivar bool cache_32bit: Defines whether the cached data is stored in float32 (True) or
        float64 (False).
    :ivar float dx: The size of grid cell in `X` direction (equals to `grid_steps[0]`).
    :ivar float dy: The size of grid cell in `Y` direction (equals to `grid_steps[1]`).
    :ivar float dz: The size of grid cell in `Z` direction (equals to `grid_steps[2]`).

    Continoues spectrum example:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import World, translate, rotate
        >>> from raysect.primitive import Cylinder, Subtract
        >>> from raysect.optical.material import CartesianRegularEmitter
        >>> # grid parameters
        >>> xmin = ymin = -1.
        >>> xmax = ymax = 1.
        >>> zmin = -0.25
        >>> zmax = 0.25
        >>> grid_shape = (100, 100, 1)
        >>> x, dx = np.linspace(xmin, xmax, grid_shape[0] + 1, retstep=True)
        >>> y, dy = np.linspace(ymin, ymax, grid_shape[1] + 1, retstep=True)
        >>> grid_steps = (dx, dy, zmax - zmin)
        >>> x = x[:-1] + 0.5 * dx  # moving to the grid cell centers
        >>> y = y[:-1] + 0.5 * dy
        >>> # spectral emission profile
        >>> min_wavelength = 375.
        >>> max_wavelength = 740.
        >>> wavelengths, delta_wavelength = np.linspace(min_wavelength, max_wavelength, 50,
                                                        retstep=True)
        >>> wvl_centre = 0.5 * (max_wavelength + min_wavelength)
        >>> wvl_range = min_wavelength - max_wavelength
        >>> shift = 2 * (wavelengths - wvl_centre) / wvl_range + 5.
        >>> radius = np.sqrt((x * x)[:, None] + (y * y)[None, :])
        >>> emission = np.cos(shift[None, None, None, :] * radius[:, :, None, None])**4
        >>> # scene
        >>> world = World()
        >>> material = CartesianRegularEmitter(grid_shape, grid_steps, emission, wavelengths)
        >>> bounding_box = Box(lower=Point3D(0, 0, 0),
                               upper=Point3D(xmax - xmin, ymax - ymin, zmax - zmin),
                               material=material, parent=world)
        >>> material.transform = (translate(0, 1., 0) * rotate(30, 0, 0) *
                                  translate(xmin, ymin, zmin))
        ...
        >>> # if ray spectral properties do not change during the rendering,
        >>> # build the cache before the first camera.observe() call to reduce memory consumptions
        >>> material.cache_build(camera.min_wavelength, camera.max_wavelength,
                                 camera.spectral_bins)

    Discrete spectrum example:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import World, translate, Point3D
        >>> from raysect.primitive import Box
        >>> from raysect.optical.observer import SpectralRadiancePipeline2D
        >>> from raysect.optical.material import CartesianRegularEmitter
        >>> # Assume that the files 'Be_4574A.npy' and 'Be_527A.npy' contain the emissions
        >>> # (in W / m^3) of Be I (3d1 1D2 -> 2p1 1P1) and Be II (4s1 2S0.5 -> 3p1 2P2.5)
        >>> # spectral lines defined on a regular Cartesian grid: -3 m < X < 3 m,
        >>> # -3 m < Y < 3 m and -6 m < Z < 6 m.
        >>> emission_4574 = np.load('Be_4574A.npy')
        >>> emission_5272 = np.load('Be_5272A.npy')
        >>> # Grid properties
        >>> xmin = ymin = -3.
        >>> xmax = ymax = 3.
        >>> zmin = -6.
        >>> zmax = 6.
        >>> grid_shape = emission_4574.shape
        >>> grid_steps = ((xmax - xmin) / grid_shape[0],
                          (ymax - ymin) / grid_shape[1],
                          (zmax - zmin) / grid_shape[2])
        >>> emission = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], 2))
        >>> emission[:, :, :, 0] = emission_4574 / (4. * np.pi)  # to W/(m^3 str)
        >>> emission[:, :, :, 1] = emission_5272 / (4. * np.pi)
        >>> # Creating the scene
        >>> world = World()
        >>> pipeline = SpectralRadiancePipeline2D()
        >>> material = CartesianRegularEmitter(grid_shape, grid_steps, emission, wavelengths,
                                               contineous=False)
        >>> bounding_box = Box(lower=Point3D(0, 0, 0),
                               upper=Point3D(xmax - xmin, ymax - ymin, zmax - zmin),
                               material=material, parent=world,
                               transform=translate(xmin, ymin, zmin))
        ...
        >>> camera.spectral_bins = 15
        >>> camera.min_wavelength = 457
        >>> camera.max_wavelength = 528
        >>> delta_wavelength = (camera.max_wavelength - camera.min_wavelength)/camera.spectral_bins
        >>> # if ray spectral properties do not change during the rendering,
        >>> # build the cache before the first camera.observe() call to reduce memory consumptions
        >>> material.cache_build(camera.min_wavelength, camera.max_wavelength,
                                 camera.spectral_bins)
        ...
        >>> # If reflections do not change the wavelength, the results for each spectral line
        >>> # can be obtained in W/(m^2 sr) in the following way.
        >>> radiance_4574 = pipeline.frame.mean[:, :, 0] * delta_wavelength
        >>> radiance_5272 = pipeline.frame.mean[:, :, -1] * delta_wavelength
    """

    def __init__(self, tuple grid_shape, tuple grid_steps, object emission, np.ndarray wavelengths,
                 bint contineous=True, bint extrapolate=True, bint cache_32bit=False, VolumeIntegrator integrator=None):

        integrator = integrator or CartesianRegularIntegrator(0.25 * min(grid_steps))
        super().__init__(grid_shape, grid_steps, emission, wavelengths, contineous=contineous, extrapolate=extrapolate,
                         cache_32bit=cache_32bit, integrator=integrator)

        self.dx = self._grid_steps[0]
        self.dy = self._grid_steps[1]
        self.dz = self._grid_steps[2]

        self.nx = self._grid_shape[0]
        self.ny = self._grid_shape[1]
        self.nz = self._grid_shape[2]

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        cdef:
            int ix, iy, iz

        # Building the cache if required
        self.cache_build(ray.get_min_wavelength(), ray.get_max_wavelength(), ray.get_bins())

        if self.cache_empty():  # emitter does not emit at this wavelength range
            return spectrum

        ix = <int>(point.x / self.dx)  # X-index of grid cell, in which the point is located
        iy = <int>(point.y / self.dy)  # Y-index of grid cell, in which the point is located
        iz = <int>(point.z / self.dz)  # Z-index of grid cell, in which the point is located
        self.add_emission_to_mv(spectrum.samples_mv, ix, iy, iz, 1.0)

        return spectrum
