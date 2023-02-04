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

cdef class SpectralSlice:

    def __init__(self, min_wavelength, max_wavelength, bins, slice_bins, slice_offset):

        # basic validation
        if bins <= 0:
            raise ValueError("The bin count must be greater than 0.")

        if min_wavelength <= 0:
            raise ValueError("The minimum wavelength must be greater than 0.")

        if max_wavelength <= 0:
            raise ValueError("The maximum wavelength must be greater than 0.")

        if min_wavelength >= max_wavelength:
            raise ValueError("The minimum wavelength must be less than the maximum wavelength.")

        if slice_bins <= 0:
            raise ValueError("The slice bin count must be greater than 0.")

        if slice_offset < 0:
            raise ValueError("The slice offset cannot be less that 0.")

        # check slice samples and offset are consistent with full sample count
        if (slice_offset + slice_bins) > bins:
            raise ValueError("The slice offset plus the bin count extends beyond the full bin count.")

        # calculate slice properties
        delta_wavelength = (max_wavelength - min_wavelength) / bins
        self.min_wavelength = min_wavelength + delta_wavelength * slice_offset
        self.max_wavelength = min_wavelength + delta_wavelength * (slice_offset + slice_bins)
        self.offset = slice_offset
        self.bins = slice_bins

        # store full spectral range
        self.total_bins = bins
        self.total_min_wavelength = min_wavelength
        self.total_max_wavelength = max_wavelength
