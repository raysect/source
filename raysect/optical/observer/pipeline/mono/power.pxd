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

cimport numpy as np
from raysect.optical.spectralfunction cimport SpectralFunction
from raysect.optical.observer.base cimport PixelProcessor, Pipeline0D, Pipeline1D, Pipeline2D
from raysect.core.math cimport StatsBin, StatsArray1D, StatsArray2D
from raysect.optical.observer.base.sampler cimport FrameSampler2D


cdef class PowerPipeline0D(Pipeline0D):

    cdef:
        public str name
        public SpectralFunction filter
        public bint accumulate
        readonly StatsBin value
        StatsArray1D _working_buffer
        list _resampled_filter
        bint _quiet


cdef class PowerPipeline1D(Pipeline1D):

    cdef:
        public str name
        public SpectralFunction filter
        public bint accumulate
        readonly StatsArray1D frame
        double[::1] _working_mean, _working_variance
        char[::1] _working_touched
        list _resampled_filter
        int _pixels
        int _samples


cdef class PowerPipeline2D(Pipeline2D):

    cdef:
        public str name
        public SpectralFunction filter
        public bint display_progress
        double _display_timer
        double _display_update_time
        public bint accumulate
        readonly StatsArray2D frame
        double[:,::1] _working_mean, _working_variance
        char[:,::1] _working_touched
        StatsArray2D _display_frame
        list _resampled_filter
        tuple _pixels
        int _samples
        object _display_figure
        double _display_black_point, _display_white_point, _display_unsaturated_fraction, _display_gamma
        bint _display_auto_exposure
        public bint display_persist_figure
        bint _quiet

    cpdef object _start_display(self)

    cpdef object _update_display(self, int x, int y)

    cpdef object _refresh_display(self)

    cpdef object _render_display(self, StatsArray2D frame, str status=*)

    cpdef np.ndarray _generate_display_image(self, StatsArray2D frame)

    cpdef double _calculate_white_point(self, np.ndarray image)

    cpdef object display(self)

    cpdef object save(self, str filename)



cdef class PowerPixelProcessor(PixelProcessor):

    cdef:
        StatsBin bin
        double[::1] filter, _temp

    cpdef object reset(self)
