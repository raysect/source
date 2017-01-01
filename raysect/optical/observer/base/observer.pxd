# cython: language_level=3

# Copyright (c) 2014-2016, Dr Alex Meakins, Raysect Project
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

from raysect.optical cimport Ray
from raysect.optical cimport Observer
from raysect.optical.observer.base.sampler cimport _FrameSamplerBase


cdef class _ObserverBase(Observer):

    cdef:
        public _FrameSamplerBase _frame_sampler
        public object render_engine
        public tuple _pipelines
        int _pixel_samples
        double _min_wavelength, _max_wavelength
        int _ray_extinction_min_depth, _ray_max_depth
        int _spectral_samples, _spectral_rays
        double _ray_extinction_prob
        public bint ray_importance_sampling
        double _ray_important_path_weight
        public tuple _pixels
        int _stats_ray_count
        int _stats_total_rays
        double _stats_start_time
        double _stats_progress_timer
        int _stats_total_tasks
        int _stats_completed_tasks

    cdef inline list _slice_spectrum(self)

    cdef inline list _generate_templates(self, list slices)

    cpdef object _render_pixel(self, tuple pixel_id, int slice_id, Ray template)

    cpdef object _update_state(self, tuple packed_result, int slice_id)

    cpdef object _initialise_statistics(self, list tasks)

    cpdef object _update_statistics(self, int sample_ray_count)

    cpdef object _finalise_statistics(self)

    cpdef list _generate_rays(self, tuple pixel_id, Ray template, int ray_count)

    cpdef double _pixel_etendue(self, tuple pixel_id)


cdef class Observer2D(_ObserverBase):
    pass
