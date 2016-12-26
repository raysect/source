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

from raysect.optical.observer._base cimport _ObserverBase, _FrameSamplerBase, _PipelineBase, PixelProcessor


cdef class FrameSampler2D(_FrameSamplerBase):
    """

    """
    pass


cdef class Pipeline2D(_PipelineBase):
    """
    """

    cpdef object initialise(self, tuple pixels, int pixel_samples, list spectral_slices):
        raise NotImplementedError("Virtual method must be implemented by a sub-class.")

    cpdef object update(self, int x, int y, tuple packed_result, int slice_id):
        raise NotImplementedError("Virtual method must be implemented by a sub-class.")

    cpdef object finalise(self):
        raise NotImplementedError("Virtual method must be implemented by a sub-class.")

    cpdef PixelProcessor pixel_processor(self, int slice_id):
        raise NotImplementedError("Virtual method must be implemented by a sub-class.")

    cpdef object _base_initialise(self, tuple pixel_config, int pixel_samples, list spectral_slices):
        self.initialise(pixel_config, pixel_samples, spectral_slices)

    cpdef object _base_update(self, tuple pixel, tuple packed_result, int slice_id):
        cdef int x, y
        x, y = pixel
        self.update(x, y, packed_result, slice_id)

    cpdef object _base_finalise(self):
        self.finalise()

    cpdef PixelProcessor _base_pixel_processor(self, int slice_id):
        return self.pixel_processor(slice_id)


cdef class Observer2D(_ObserverBase):

    def __init__(self, pixels, frame_sampler, processing_pipelines, render_engine=None, parent=None,
                 transform=None, name=None, pixel_samples=None, spectral_rays=None, spectral_samples=None,
                 min_wavelength=None, max_wavelength=None, ray_extinction_prob=None, ray_min_depth=None,
                 ray_max_depth=None, ray_importance_sampling=None, ray_important_path_weight=None):

        super().__init__(
            render_engine, parent, transform, name, pixel_samples, spectral_rays, spectral_samples,
            min_wavelength, max_wavelength, ray_extinction_prob, ray_min_depth,
            ray_max_depth, ray_importance_sampling, ray_important_path_weight
        )

        self.pixels = pixels
        self.frame_sampler = frame_sampler
        self.pipelines = processing_pipelines

    @property
    def pixels(self):
        return self._pixel_config

    @pixels.setter
    def pixels(self, value):
        value = tuple(value)
        if len(value) != 2:
            raise ValueError("Pixels must be a 2 element tuple defining the x and y resolution.")
        x, y = value
        if x <= 0:
            raise ValueError("Number of x pixels must be greater than 0.")
        if y <= 0:
            raise ValueError("Number of y pixels must be greater than 0.")
        self._pixel_config = value

    @property
    def frame_sampler(self):
        return self._frame_sampler

    @frame_sampler.setter
    def frame_sampler(self, value):
        if not isinstance(value, FrameSampler2D):
            raise TypeError("The frame sampler for a 2d observer must be a subclass of FrameSampler2D.")
        self._frame_sampler = value

    @property
    def pipelines(self):
        return self._pipelines

    @pipelines.setter
    def pipelines(self, value):

        pipelines = tuple(value)
        if len(pipelines) < 1:
            raise ValueError("At least one processing pipeline must be provided.")
        for pipeline in pipelines:
            if not isinstance(pipeline, Pipeline2D):
                raise TypeError("Processing pipelines for a 2d observer must be a subclass of Pipeline2D.")
        self._pipelines = pipelines
