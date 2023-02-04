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
from raysect.optical.observer.base cimport PixelProcessor, Pipeline2D
from raysect.core.math cimport StatsArray3D, StatsArray1D


cdef class RGBPipeline2D(Pipeline2D):

    cdef:
        public str name
        public bint display_progress
        double _display_timer
        double _display_update_time
        public bint accumulate
        readonly StatsArray3D xyz_frame
        double[:,:,::1] _working_mean, _working_variance
        char[:,::1] _working_touched
        StatsArray3D _display_frame
        list _processors
        tuple _pixels
        int _samples
        object _display_figure
        double _display_sensitivity, _display_unsaturated_fraction
        bint _display_auto_exposure
        public bint display_persist_figure
        bint _quiet

    cpdef object _start_display(self)

    cpdef object _update_display(self, int x, int y)

    cpdef object _refresh_display(self)

    cpdef object _render_display(self, StatsArray3D frame, str status=*)

    cpdef np.ndarray _generate_display_image(self, StatsArray3D frame)

    cpdef double _calculate_sensitivity(self, np.ndarray image)

    cpdef np.ndarray _generate_srgb_image(self, double[:,:,::1] xyz_image_mv)

    cpdef object display(self)

    cpdef object save(self, str filename)


cdef class XYZPixelProcessor(PixelProcessor):

    cdef:
        double[:,::1] resampled_xyz
        StatsArray1D xyz

    cpdef object reset(self)
