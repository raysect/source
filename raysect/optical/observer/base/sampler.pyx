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

cdef class FrameSampler1D:
    """
    Base class for 1D frame samplers.
    """

    cpdef list generate_tasks(self, int pixels):
        """
        Generates a list of tuples that selects the pixels to render.

        Must return a list of tuples where each tuple contains the id of a pixel
        to render. For example:

            tasks = [(1,), (5,), (512,), ...]

        This is a virtual method and must be implemented in a sub class.

        :param int pixels: The number of pixels in the frame.
        :rtype: list
        """
        raise NotImplementedError("Virtual method must be implemented by a sub-class.")


cdef class FrameSampler2D:
    """
    Base class for 2D frame samplers.
    """

    cpdef list generate_tasks(self, tuple pixels):
        """
        Generates a list of tuples that selects the pixels to render.

        Must return a list of tuples where each tuple contains the id of a pixel
        to render. For example:

            tasks = [(1, 10), (5, 53), (512, 354), ...]

        This is a virtual method and must be implemented in a sub class.

        :param tuple pixels: Contains the (x, y) pixel dimensions of the frame.
        :rtype: list
        """
        raise NotImplementedError("Virtual method must be implemented by a sub-class.")
