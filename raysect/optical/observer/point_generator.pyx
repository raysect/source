# cython: language_level=3

# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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


from raysect.core.math.point cimport Point3D, Point2D
from raysect.core.math.random import point_disk, uniform
from raysect.core import AffineMatrix3D


cdef class PointGenerator:
    """
    Base class for defining the sampling area for rays launched by an observer. Observers use the
    VectorGenerator and PointGenerator classes to build N rays for sampling.

    :param AffineMatrix3D transform: Transform relative to the camera origin.
    """

    def __init__(self, transform=None):
        if transform is None:
            transform = AffineMatrix3D()
        self.transform = transform

    def __call__(self, n):
        """
        :param int n: Generate n points that sample this observers surface area.
        """
        return self.sample(n)

    cpdef list sample(self, int n):
        """
        :param int n: Generate n vectors that sample this observers acceptance solid angle.
        """
        raise NotImplemented("The method sample(n) for this point generator needs to be implemented.")


cdef class SinglePoint(PointGenerator):
    """
    A dummy point generator that returns all samples at the origin.

    :param AffineMatrix3D transform: Transform relative to the camera origin.
    """

    cpdef list sample(self, int n):
        cdef list results
        cdef int i

        results = []
        for i in range(n):
            results.append(Point3D(0, 0, 0).transform(self.transform))
        return results


cdef class Disk(PointGenerator):
    """
    Generates a random Point3D on a disk.

    :param double radius: The radius of the disk.
    :param AffineMatrix3D transform: Transform relative to the camera origin.
    """

    def __init__(self, radius=1, transform=None):
        super().__init__(transform=transform)
        self.radius = radius

    cpdef list sample(self, int n):
        cdef list results
        cdef int i
        cdef double radius = self.radius
        cdef Point2D random_point

        results = []
        for i in range(n):
            random_point = point_disk()
            results.append(Point3D(random_point.x * radius, random_point.y * radius, 0).transform(self.transform))
        return results


cdef class Rectangle(PointGenerator):
    """
    Generates a random Point3D on a rectangle.

    :param double width: The width of the rectangular sampling area of this observer.
    :param double height: The height of the rectangular sampling area of this observer.
    :param AffineMatrix3D transform: Transform relative to the camera origin.
    """

    def __init__(self, width=1, height=1, transform=None):
        super().__init__(transform=transform)
        self.width = width
        self.height = height

    cpdef list sample(self, int n):
        cdef list results
        cdef int i
        cdef double u, v, width_offset = self.width/2, height_offset = self.height/2

        results = []
        for i in range(n):
            u = uniform()*self.width - width_offset
            v = uniform()*self.height - height_offset
            results.append(Point3D(u, v, 0).transform(self.transform))
        return results
