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

cimport cython

@cython.freelist(64)
cdef class PixelProcessor:
    """
    Base class for pixel processors.

    To optimally use computing resources, observers may use parallel processes
    to sample the world.

    Raysect observers launch multiple worker processes to sample the world, these
    processes send their results back to a single process that combines them into
    a frame.

    In order to distribute the processing of the returned spectra, it is
    necessary to perform the data processing on each worker.

    Each worker is given a pixel id and a number of spectral samples to collect
    for that pixel. The worker launches a ray to collect a sample and a spectrum is
    returned. When a spectrum is obtained, the worker calls add_sample() on each
    pixel processor associated with the pipelines attached to the observer. The pixel
    prcoessor processes the spectrum and accumulates the results in its internal buffer.

    When the pixel samples are complete the worker calls pack_results() on each pixel
    processor. These results are sent back to the process handling the frame assembly.

    """

    cpdef object add_sample(self, Spectrum spectrum, double sensitivity):
        """
        Processes a spectrum and adds it to internal buffer.

        This is a virtual method and must be implemented in a sub class.

        :param Spectrum spectrum: The sampled spectrum.
        :param float sensitivity: The pixel sensitivity.
        """
        raise NotImplementedError("Virtual method must be implemented by a sub-class.")

    cpdef tuple pack_results(self):
        """
        Packs the contents of the internal buffer.

        This method must return a tuple. The contents and length of the tuple are entirely
        defined by the needs of the pipeline.

        This is a virtual method and must be implemented in a sub class.

        :rtype: tuple
        """
        raise NotImplementedError("Virtual method must be implemented by a sub-class.")


