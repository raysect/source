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

from raysect.primitive import Intersect, Sphere, Cylinder

from raysect.core.math.affinematrix cimport AffineMatrix3D, translate
from raysect.core.classes cimport Material
from raysect.primitive.utility cimport EncapsulatedPrimitive
from libc.math cimport sqrt


# NOTES:
# thickness is center thickness unless otherwise stated
# calculate edge thickness and check it is >0 if it isn't then the radius of the faces and center width are not compatible with the diameter
# the back face is the negative most face along the Z-axis, front face is the positive most face
# the lens is aligned such that z=0 lies on the back lens surface (on the lens axis), the lens extends along the +ve z axis

DEF PAD_FACTOR = 1.000001

# todo: docstrings
# todo: add attributes for derived lens properties, e.g. focal length
# todo: add additional initalisation methods
cdef class BiConvex(EncapsulatedPrimitive):

    cdef:
        readonly double diameter
        readonly double center_thickness
        readonly double edge_thickness
        readonly double front_curvature
        readonly double back_curvature

    def __init__(self, double diameter, double center_thickness, double front_curvature, double back_curvature, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):

        self.diameter = diameter
        self.center_thickness = center_thickness
        self.front_curvature = front_curvature
        self.back_curvature = back_curvature
        self._calc_edge_thickness()
        radius = 0.5 * diameter

        # validate
        if diameter <= 0:
            raise ValueError("The lens diameter must be greater than zero.")

        if center_thickness <= 0:
            raise ValueError("The lens thickness must be greater than zero.")

        if front_curvature <= 0:
            raise ValueError("The front radius of curvature must be greater than zero.")

        if back_curvature <= 0:
            raise ValueError("The back radius of curvature must be greater than zero.")

        if front_curvature < radius:
            raise ValueError("The radius of curvature of the front face cannot be less than the barrel radius.")

        if back_curvature < radius:
            raise ValueError("The radius of curvature of the back face cannot be less than the barrel radius.")

        if self.edge_thickness < 0:
            raise ValueError("The curvatures and/or thickness are too small to produce a lens of the specified diameter.")

        # padding to add to the barrel cylinder to avoid potential numerical accuracy issues
        padding = center_thickness * PAD_FACTOR

        # construct lens using CSG
        front = Sphere(front_curvature, transform=translate(0, 0, center_thickness - front_curvature))
        back = Sphere(back_curvature, transform=translate(0, 0, back_curvature))
        barrel = Cylinder(0.5 * diameter, center_thickness + padding, transform=translate(0, 0, -0.5 * padding))
        lens = Intersect(barrel, Intersect(front, back))

        # attach to local root (performed in EncapsulatedPrimitive init)
        super().__init__(lens, parent, transform, material, name)

    cdef inline void _calc_edge_thickness(self):

        cdef double radius, radius_sqr, front_thickness, back_thickness

        # barrel radius
        radius = 0.5 * self.diameter
        radius_sqr = radius * radius

        # thickness of spherical surfaces
        front_thickness = self.front_curvature - sqrt(self.front_curvature * self.front_curvature - radius_sqr)
        back_thickness = self.back_curvature - sqrt(self.back_curvature * self.back_curvature - radius_sqr)

        # edge thickness is the length of the barrel without the curved surfaces
        self.edge_thickness = self.center_thickness - (front_thickness + back_thickness)

    def __str__(self):
        """String representation."""

        s = "<BiConvex at {}>".format(str(hex(id(self))))
        if self.name:
            return "{} {}".format(self.name, s)
        else:
            return s



cdef class BiConcave(EncapsulatedPrimitive):
    pass


cdef class PlanoConvex(EncapsulatedPrimitive):
    pass


cdef class PlanoConcave(EncapsulatedPrimitive):
    pass


cdef class PositiveMeniscus(EncapsulatedPrimitive):
    pass


cdef class NegativeMeniscus(EncapsulatedPrimitive):
    pass
































