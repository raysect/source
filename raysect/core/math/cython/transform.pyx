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

"""
Cython utility functions for generating coordinate transforms.

These functions can not be called from Python directly.

.. WARNING:: For speed, none of these functions perform any type or bounds
   checking. Supplying malformed data may result in data corruption or a
   segmentation fault.
"""

from raysect.core.math.affinematrix cimport new_affinematrix3d


cdef AffineMatrix3D rotate_basis(Vector3D forward, Vector3D up):
    """
    Returns a rotation matrix defined by forward and up vectors.

    The +ve Z-axis of the resulting coordinate space will be aligned with the
    forward vector. The +ve Y-axis will be aligned to lie in the plane defined
    the forward and up vectors, along the projection of the up vector that
    lies orthogonal to the forward vector. The X-axis will lie perpendicular to
    the plane.

    The forward and upwards vectors need not be orthogonal. The up vector will
    be rotated in the plane defined by the two vectors until it is orthogonal.

    :param forward: A Vector3D object defining the forward direction.
    :param up: A Vector3D object defining the up direction.
    :return: An AffineMatrix3D object.
    """

    cdef Vector3D right = up.cross(forward)

    return new_affinematrix3d(
        right.x, up.x, forward.x, 0.0,
        right.y, up.y, forward.y, 0.0,
        right.z, up.z, forward.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    )

cdef AffineMatrix3D rotate_basis_inverse(Vector3D forward, Vector3D up):
    """
    Returns the inverse of the rotation matrix defined by forward and up vectors.

    The +ve Z-axis of the resulting coordinate space will be aligned with the
    forward vector. The +ve Y-axis will be aligned to lie in the plane defined
    the forward and up vectors, along the projection of the up vector that
    lies orthogonal to the forward vector. The X-axis will lie perpendicular to
    the plane.

    The forward and upwards vectors need not be orthogonal. The up vector will
    be rotated in the plane defined by the two vectors until it is orthogonal.

    :param forward: A Vector3D object defining the forward direction.
    :param up: A Vector3D object defining the up direction.
    :return: An AffineMatrix3D object.
    """

    cdef Vector3D right = up.cross(forward)

    return new_affinematrix3d(
        right.x, right.x, right.x, 0.0,
        up.y, up.y, up.y, 0.0,
        forward.z, forward.z, forward.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    )
