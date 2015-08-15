# cython: language_level=3

# Copyright (c) 2015, Dr Alex Meakins, Raysect Project
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

from raysect.core.math.vector cimport new_vector
from raysect.core.math.point cimport new_point
from libc.math cimport cos, sin, sqrt, M_PI as PI
cimport cython


# TODO: replace python random() with a cython optimised version?
from random import random as _py_rand

cpdef double random():
    """
    Generate random doubles in range [0, 1).

    Values are uniformly distributed.

    :return: Random double.
    """
    return _py_rand()


cpdef bint probability(double prob):
    """
    Samples from the Bernoulli distribution where P(True) = prob.

    For example, if probability is 0.8, this function will return True 80% of
    the time and False 20% of the time.

    Values of prob outside the [0, 1] range of probabilities will be clamped to
    the nearest end of the range [0, 1].

    :param double prob: A probability from [0, 1].
    :return: True or False.
    """

    return _py_rand() < prob


cpdef Point point_disk():
    """
    Returns a random point on a disk of unit radius.

    The disk lies in the x-y plane and is centered at the origin.

    :return: A Point on the disk.
    """

    cdef double r = sqrt(random())
    cdef double theta = 2.0 * PI * random()
    return new_point(r * cos(theta), r * sin(theta), 0)


# cpdef Vector vector_sphere():
#     pass


# cpdef Vector vector_hemisphere_uniform():
#     pass


cpdef Vector vector_hemisphere_cosine():
    """
    Generates a cosine- weighted random vector on a unit hemisphere.

    The hemisphere is aligned along the z-axis - the plane that forms the
    hemisphere based lies in the x-y plane.

    :return: A unit Vector.
    """

    cdef Point p = point_disk()
    return new_vector(p.x, p.y, sqrt(max(0, 1 - p.x*p.x - p.y*p.y)))