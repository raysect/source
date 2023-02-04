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

from raysect.core.math.point cimport Point3D, Point2D, new_point3d, new_point2d
from raysect.core.math.vector cimport Vector2D, new_vector2d, Vector3D, new_vector3d
from raysect.core.math.normal cimport Normal3D, new_normal3d
from raysect.core.math.affinematrix cimport AffineMatrix3D, new_affinematrix3d
from raysect.core.math.quaternion cimport Quaternion, new_quaternion, new_quaternion_from_matrix, new_quaternion_from_axis_angle
from raysect.core.math.transform cimport translate, rotate_x, rotate_y, rotate_z, rotate_vector, rotate, rotate_basis, to_cylindrical, from_cylindrical, extract_translation, extract_rotation
from raysect.core.math.units cimport *
from raysect.core.math.statsarray cimport StatsBin, StatsArray1D, StatsArray2D, StatsArray3D
from raysect.core.math.sampler cimport *
from raysect.core.math.polygon cimport triangulate2d

