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

"""
Cython utility functions for generating coordinate transforms.

These functions can not be called from Python directly.

.. WARNING:: For speed, none of these functions perform any type or bounds
   checking. Supplying malformed data may result in data corruption or a
   segmentation fault.
"""

from raysect.core.math.affinematrix cimport new_affinematrix3d


cdef inline AffineMatrix3D local_to_surface(Vector3D normal, Vector3D tangent):
    """
    Returns a transform matrix from that maps from local space to surface space.

    Some calculations are most efficiently performed in a coordinate space
    aligned with the normal of a primitive surface. This convenience function
    generates a rotation from primitive local space to surface space.

    Surface space is defined by a normal and tangent defined in the local
    coordinate system. These vectors must be orthogonal and normalised, no
    checks are performed.

    :param Vector normal: Surface normal in local space.
    :param Vector tangent: Surface tangent in local space.
    :return: Transform matrix from local to surface space.
    """

    cdef Vector3D bitangent = normal.cross(tangent)

    return new_affinematrix3d(
        tangent.x, tangent.y, tangent.z, 0.0,
        bitangent.x, bitangent.y, bitangent.z, 0.0,
        normal.x, normal.y, normal.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    )

cdef inline AffineMatrix3D surface_to_local(Vector3D normal, Vector3D tangent):
    """
    Returns a transform matrix from that maps from surface space to local space.

    Some calculations are most efficiently performed in a coordinate space
    aligned with the normal of a primitive surface. This convenience function
    generates a rotation from surface space to primitive local space.

    Surface space is defined by a normal and tangent defined in the local
    coordinate system. These vectors must be orthogonal and normalised, no
    checks are performed.

    :param Vector normal: Surface normal in local space.
    :param Vector tangent: Surface tangent in local space.
    :return: Transform matrix from surface to local space.
    """

    cdef Vector3D bitangent = normal.cross(tangent)

    return new_affinematrix3d(
        tangent.x, bitangent.x, normal.x, 0.0,
        tangent.y, bitangent.y, normal.y, 0.0,
        tangent.z, bitangent.z, normal.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    )
