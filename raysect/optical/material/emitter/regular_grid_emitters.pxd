# cython: language_level=3

# Copyright (c) 2014-2018, Dr Alex Meakins, Raysect Project
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

from numpy cimport ndarray
from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator, InhomogeneousVolumeEmitter
cimport cython


cdef class RegularGridIntegrator(VolumeIntegrator):
    
    cdef:
        double _step
        int _min_samples

cdef class CylindricalRegularIntegrator(RegularGridIntegrator):

    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world)

cdef class CartesianRegularIntegrator(RegularGridIntegrator):

    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world)


cdef class RegularGridEmitter(InhomogeneousVolumeEmitter):

    cdef:
        readonly long nvoxel
        readonly bint contineous
        readonly bint cache_32bit_indices
        readonly ndarray wavelengths
        readonly object emission

        int[3] _grid_shape
        double[3] _grid_steps
        bint _extrapolate
        bint _cache_32bit
        double _cache_min_wvl
        double _cache_max_wvl
        int _cache_num_samp
        long _cache_data_size
        object _cache
        double[::1] _wavelengths_mv

        const float[::1] cache_data_32_mv
        const double[::1] cache_data_64_mv
        const int[::1] cache_indptr_32_mv
        const int[::1] cache_indices_32_mv
        const long[::1] cache_indptr_64_mv
        const long[::1] cache_indices_64_mv

    cdef long get_voxel_index(self, int i, int j, int k) nogil

    cpdef long voxel_index(self, int i, int j, int k)

    cdef void _cache_init(self)

    cpdef bint cache_valid(self, double min_wavelength, double max_wavelength, int bins)

    cpdef bint cache_empty(self)

    cpdef void cache_override(self, object cache, double min_wavelength, double max_wavelength)

    cpdef void cache_build(self, double min_wavelength, double max_wavelength, int bins, bint forced=*)

    cpdef object integrate(self, double min_wavelength, double max_wavelength)

    cdef void add_emission_to_mv(self, double[::1] samples_mv, int i, int j, int k, double ray_path) nogil

    cpdef void add_emission_to_array(self, ndarray samples, int i, int j, int k, double ray_path)


cdef class CylindricalRegularEmitter(RegularGridEmitter):

    cdef:
        readonly double rmin
        readonly double dr
        readonly double dphi
        readonly double dz
        readonly double period
        readonly int nr
        readonly int nphi
        readonly int nz

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world)


cdef class CartesianRegularEmitter(RegularGridEmitter):

    cdef:
        readonly double dx
        readonly double dy
        readonly double dz
        readonly int nx
        readonly int ny
        readonly int nz

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world)
    
