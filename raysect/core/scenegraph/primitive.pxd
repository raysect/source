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

from raysect.core.ray cimport Ray
from raysect.core.material cimport Material
from raysect.core.intersection cimport Intersection
from raysect.core.math cimport Point3D, AffineMatrix3D
from raysect.core.scenegraph.node cimport Node
from raysect.core.scenegraph.signal cimport ChangeSignal
from raysect.core.boundingbox cimport BoundingBox3D
from raysect.core.boundingsphere cimport BoundingSphere3D


cdef class Primitive(Node):

    cdef Material _material

    cdef Material get_material(self)

    cpdef Intersection hit(self, Ray ray)

    cpdef Intersection next_intersection(self)

    cpdef bint contains(self, Point3D p) except -1

    cpdef BoundingBox3D bounding_box(self)

    cpdef BoundingSphere3D bounding_sphere(self)

    cpdef object instance(self, object parent=*, AffineMatrix3D transform=*, Material material=*, str name=*)

    cpdef object notify_geometry_change(self)

    cpdef object notify_material_change(self)