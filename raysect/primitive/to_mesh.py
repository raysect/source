
# Copyright (c) 2014-2018, Dr Alex Meakins, Raysect Project
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


import numpy as np
from scipy.spatial import Delaunay

from raysect.core import Point3D, Vector3D
from raysect.primitive import Mesh, Box, Sphere, Cylinder, Cone, Parabola, Intersect, Union, Subtract

from raysect.primitive.mesh.subdivision import subdivide
from raysect.primitive.mesh.csg import perform_mesh_csg
from raysect.primitive.mesh.csg import Intersect as IntersectOperator, Union as UnionOperator,Subtract as SubtractOperator


def box_to_mesh(box):

    if not isinstance(box, Box):
        raise TypeError("The _box_to_mesh() function takes a Raysect Box primitive as an argument, "
                        "wrong type '{}' given.".format(type(box)))

    lower = box.lower
    upper = box.upper
    # more negative face in x-z plane
    p1a = lower  # lower corner in x-z plane
    p2a = Point3D(lower.x, lower.y, upper.z)
    p3a = Point3D(upper.x, lower.y, upper.z)  # upper corner in x-z plane
    p4a = Point3D(upper.x, lower.y, lower.z)
    # more positive face in x-z plane
    p1b = Point3D(lower.x, upper.y, lower.z)
    p2b = Point3D(lower.x, upper.y, upper.z)
    p3b = upper
    p4b = Point3D(upper.x, upper.y, lower.z)
    vertices = [[p1a.x, p1a.y, p1a.z], [p2a.x, p2a.y, p2a.z],
                [p3a.x, p3a.y, p3a.z], [p4a.x, p4a.y, p4a.z],
                [p1b.x, p1b.y, p1b.z], [p2b.x, p2b.y, p2b.z],
                [p3b.x, p3b.y, p3b.z], [p4b.x, p4b.y, p4b.z]]
    triangles = [[1, 0, 3], [1, 3, 2],  # front face (x-z)
                 [7, 4, 5], [7, 5, 6],  # rear face (x-z)
                 [5, 1, 2], [5, 2, 6],  # top face (x-y)
                 [3, 0, 4], [3, 4, 7],  # bottom face (x-y)
                 [4, 0, 5], [1, 5, 0],  # left face (y-z)
                 [2, 3, 7], [2, 7, 6]]  # right face (y-z)

    mesh = Mesh(vertices, triangles)
    mesh = subdivide(mesh)

    vertices = np.array(mesh.data.vertices)
    triangles = np.array(mesh.data.triangles)

    if box.parent:
        to_world = box.to_root()
    else:
        to_world = box.transform

    # Convert vertices to positions in world coordinates
    for i in range(vertices.shape[0]):
        p = Point3D(vertices[i, 0], vertices[i, 1], vertices[i, 2]).transform(to_world)
        vertices[i, 0] = p.x
        vertices[i, 1] = p.y
        vertices[i, 2] = p.z

    return vertices, triangles


def sphere_to_mesh(sphere, subdivision_count=2):

    if not isinstance(sphere, Sphere):
        raise TypeError("The _sphere_to_mesh() function takes a Raysect Box primitive as an argument, "
                        "wrong type '{}' given.".format(type(sphere)))

    # Calculate vertices and faces using the icosohedren method
    # We compute a regular icosohedren with 12 vertices and 20 faces.
    # Vertices given by all perturbations of:
    # (0, ±1, ±ϕ), (±1, ±ϕ, 0), (±ϕ, 0, ±1), where ϕ = golden ratio

    golden_ratio = 1.61803398875

    radius = sphere.radius

    v1 = Vector3D(-1.0, golden_ratio, 0.0).normalise() * radius
    v2 = Vector3D(1.0, golden_ratio, 0.0).normalise() * radius
    v3 = Vector3D(-1.0, -golden_ratio, 0.0).normalise() * radius
    v4 = Vector3D(1.0, -golden_ratio, 0.0).normalise() * radius
    v5 = Vector3D(0.0, -1.0, golden_ratio).normalise() * radius
    v6 = Vector3D(0.0, 1.0, golden_ratio).normalise() * radius
    v7 = Vector3D(0.0, -1.0, -golden_ratio).normalise() * radius
    v8 = Vector3D(0.0, 1.0, -golden_ratio).normalise() * radius
    v9 = Vector3D(golden_ratio, 0.0, -1.0).normalise() * radius
    v10 = Vector3D(golden_ratio, 0.0, 1.0).normalise() * radius
    v11 = Vector3D(-golden_ratio, 0.0, -1.0).normalise() * radius
    v12 = Vector3D(-golden_ratio, 0.0, 1.0).normalise() * radius

    vertices = [
        [v1.x, v1.y, v1.z],
        [v2.x, v2.y, v2.z],
        [v3.x, v3.y, v3.z],
        [v4.x, v4.y, v4.z],
        [v5.x, v5.y, v5.z],
        [v6.x, v6.y, v6.z],
        [v7.x, v7.y, v7.z],
        [v8.x, v8.y, v8.z],
        [v9.x, v9.y, v9.z],
        [v10.x, v10.y, v10.z],
        [v11.x, v11.y, v11.z],
        [v12.x, v12.y, v12.z],
    ]

    triangles = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]
    ]

    # Optional - subdivision of icosohedren to increase resolution
    num_vertices = 12
    num_triangles = 20
    for i in range(subdivision_count):
        for j in range(num_triangles):
            triangle = triangles[j]
            # extract current triangle vertices
            v0_id = triangle[0]
            v1_id = triangle[1]
            v2_id = triangle[2]
            v0 = Vector3D(vertices[v0_id][0], vertices[v0_id][1], vertices[v0_id][2])
            v1 = Vector3D(vertices[v1_id][0], vertices[v1_id][1], vertices[v1_id][2])
            v2 = Vector3D(vertices[v2_id][0], vertices[v2_id][1], vertices[v2_id][2])

            # subdivide with three new vertices
            v3 = (v0 + v1).normalise() * radius
            v3_id = num_vertices
            v4 = (v1 + v2).normalise() * radius
            v4_id = num_vertices + 1
            v5 = (v2 + v0).normalise() * radius
            v5_id = num_vertices + 2
            vertices.append([v3.x, v3.y, v3.z])
            vertices.append([v4.x, v4.y, v4.z])
            vertices.append([v5.x, v5.y, v5.z])

            # ... and three new faces
            triangles[j] = [v0_id, v3_id, v5_id]  # replace the first face
            triangles.append([v3_id, v1_id, v4_id])
            triangles.append([v4_id, v2_id, v5_id])
            triangles.append([v3_id, v4_id, v5_id])

            num_vertices += 3
            num_triangles += 3

    vertices = np.array(vertices)
    triangles = np.array(triangles)

    if sphere.parent:
        to_world = sphere.to_root()
    else:
        to_world = sphere.transform

    # Convert vertices to positions in world coordinates
    for i in range(vertices.shape[0]):
        p = Point3D(vertices[i, 0], vertices[i, 1], vertices[i, 2]).transform(to_world)
        vertices[i, 0] = p.x
        vertices[i, 1] = p.y
        vertices[i, 2] = p.z

    return vertices, triangles


def cylinder_to_mesh(cylinder, vertical_divisions=10, cylindrical_divisions=36, radial_divisions=5):

    if not isinstance(cylinder, Cylinder):
        raise TypeError("The _cylinder_to_mesh() function takes a Raysect Cylinder primitive as an argument, "
                        "wrong type '{}' given.".format(type(cylinder)))

    radius = cylinder.radius
    height = cylinder.height

    # first make the main cylinder
    theta_step = 360 / cylindrical_divisions

    vertices = []
    for i in range(vertical_divisions):
        z = (i / (vertical_divisions - 1)) * height
        for j in range(cylindrical_divisions):
            theta_rad = np.deg2rad(j * theta_step)
            vertices.append([radius * np.cos(theta_rad), radius * np.sin(theta_rad), z])

    triangles = []
    for i in range(vertical_divisions - 1):

        row_start = cylindrical_divisions * i
        next_row_start = cylindrical_divisions * (i + 1)

        for j in range(cylindrical_divisions):

            v1 = row_start + j
            if j != cylindrical_divisions - 1:
                v2 = row_start + j + 1
                v3 = next_row_start + j + 1
            else:
                v2 = row_start
                v3 = next_row_start
            v4 = next_row_start + j

            triangles.append([v1, v2, v3])
            triangles.append([v3, v4, v1])

    def _make_cap_triangles(n_cylindrical_segments, n_radial_segments, radius, z_height):

        working_cylindrical_segments = n_cylindrical_segments

        cap_vertices = []

        for i in range(n_radial_segments):

            working_radius = radius * (1 - (i / n_radial_segments))
            theta_step = 360 / working_cylindrical_segments

            for j in range(working_cylindrical_segments):
                theta_rad = np.deg2rad(j * theta_step)
                cap_vertices.append([working_radius * np.cos(theta_rad), working_radius * np.sin(theta_rad), z_height])

            working_cylindrical_segments -= int(n_cylindrical_segments/(n_radial_segments+1))
            if working_cylindrical_segments < 5:
                working_cylindrical_segments = 5

        # Finally, add centre point
        cap_vertices.append([0, 0, z_height])

        vertices_2d = np.array(cap_vertices)[:, 0:2]

        triangles = Delaunay(vertices_2d).simplices

        return cap_vertices, triangles

    # Make the upper and lower end caps
    lower_cap_vertices, lower_cap_triangles = _make_cap_triangles(cylindrical_divisions, radial_divisions, radius, 0)
    lower_cap_triangles = np.flip(lower_cap_triangles, 1)
    lower_cap_triangles += len(vertices)
    lower_cap_triangles = lower_cap_triangles.tolist()

    vertices += lower_cap_vertices
    triangles += lower_cap_triangles

    upper_cap_vertices, upper_cap_triangles = _make_cap_triangles(cylindrical_divisions, radial_divisions, radius, height)
    upper_cap_triangles += len(vertices)
    upper_cap_triangles = upper_cap_triangles.tolist()

    vertices += upper_cap_vertices
    triangles += upper_cap_triangles

    vertices = np.array(vertices)
    triangles = np.array(triangles)

    if cylinder.parent:
        to_world = cylinder.to_root()
    else:
        to_world = cylinder.transform

    # Convert vertices to positions in world coordinates
    for i in range(vertices.shape[0]):
        p = Point3D(vertices[i, 0], vertices[i, 1], vertices[i, 2]).transform(to_world)
        vertices[i, 0] = p.x
        vertices[i, 1] = p.y
        vertices[i, 2] = p.z

    return vertices, triangles


def cone_to_mesh(cone, vertical_divisions=10, cylindrical_divisions=36, base_radial_divisions=5):

    if not isinstance(cone, Cone):
        raise TypeError("The _cone_to_mesh() function takes a Raysect Cone primitive as an argument, "
                        "wrong type '{}' given.".format(type(cone)))

    radius = cone.radius
    height = cone.height

    # first make the cone body

    working_cylindrical_divisions = cylindrical_divisions

    cone_body_vertices = []
    for i in range(vertical_divisions):

        working_radius = radius * (1 - (i / vertical_divisions))
        working_height = height * (i / vertical_divisions)
        theta_step = 360 / working_cylindrical_divisions

        for j in range(working_cylindrical_divisions):
            theta_rad = np.deg2rad(j * theta_step)
            cone_body_vertices.append([working_radius * np.cos(theta_rad), working_radius * np.sin(theta_rad), working_height])

        working_cylindrical_divisions -= int(cylindrical_divisions / (vertical_divisions + 1))
        if working_cylindrical_divisions < 5:
            working_cylindrical_divisions = 5

    # Finally, add centre point
    cone_body_vertices.append([0, 0, height])

    # Triangulate the vertices
    vertices_2d = np.array(cone_body_vertices)[:, 0:2]
    cone_body_triangles = Delaunay(vertices_2d).simplices.tolist()


    # Now make the cone base

    working_cylindrical_divisions = cylindrical_divisions
    cone_base_vertices = []
    for i in range(base_radial_divisions):

        working_radius = radius * (1 - (i / base_radial_divisions))
        theta_step = 360 / working_cylindrical_divisions

        for j in range(working_cylindrical_divisions):
            theta_rad = np.deg2rad(j * theta_step)
            cone_base_vertices.append([working_radius * np.cos(theta_rad), working_radius * np.sin(theta_rad), 0])

        working_cylindrical_divisions -= int(cylindrical_divisions / (base_radial_divisions + 1))
        if working_cylindrical_divisions < 5:
            working_cylindrical_divisions = 5

    # Finally, add centre point
    cone_base_vertices.append([0, 0, 0])

    # Triangulate the vertices
    vertices_2d = np.array(cone_base_vertices)[:, 0:2]
    cone_base_triangles = np.flip(Delaunay(vertices_2d).simplices + len(cone_body_vertices), 1).tolist()

    # Combine the resulting triangles together
    vertices = cone_body_vertices + cone_base_vertices
    triangles = cone_body_triangles + cone_base_triangles

    vertices = np.array(vertices)
    triangles = np.array(triangles)

    if cone.parent:
        to_world = cone.to_root()
    else:
        to_world = cone.transform

    # Convert vertices to positions in world coordinates
    for i in range(vertices.shape[0]):
        p = Point3D(vertices[i, 0], vertices[i, 1], vertices[i, 2]).transform(to_world)
        vertices[i, 0] = p.x
        vertices[i, 1] = p.y
        vertices[i, 2] = p.z

    return vertices, triangles


def mesh_to_mesh(mesh):

    vertices = mesh.data.vertices.copy()
    triangles = mesh.data.triangles.copy()

    if mesh.parent:
        to_world = mesh.to_root()
    else:
        to_world = mesh.transform

    # Convert vertices to positions in world coordinates
    for i in range(vertices.shape[0]):
        p = Point3D(vertices[i, 0], vertices[i, 1], vertices[i, 2]).transform(to_world)
        vertices[i, 0] = p.x
        vertices[i, 1] = p.y
        vertices[i, 2] = p.z

    return vertices, triangles


# Intersect, Union, Subtract
def csg_to_mesh(csg_primitive):

    vertices, triangles = to_mesh(csg_primitive.primitive_a)
    mesh_a = Mesh(vertices, triangles)

    vertices, triangles = to_mesh(csg_primitive.primitive_b)
    mesh_b = Mesh(vertices, triangles)

    if csg_primitive.__class__ == Intersect:
        operator = IntersectOperator()
    elif csg_primitive.__class__ == Union:
        operator = UnionOperator()
    elif csg_primitive.__class__ == Subtract:
        operator = SubtractOperator()
    else:
        raise ValueError("Unidentified CSG primitive '{}'.".format(csg_primitive.__class__))

    mesh = perform_mesh_csg(mesh_a, mesh_b, operator=operator)

    vertices = mesh.data.vertices.copy()
    triangles = mesh.data.triangles.copy()

    if csg_primitive.parent:
        to_world = csg_primitive.to_root()
    else:
        to_world = csg_primitive.transform

    # Convert vertices to positions in world coordinates
    for i in range(vertices.shape[0]):
        p = Point3D(vertices[i, 0], vertices[i, 1], vertices[i, 2]).transform(to_world)
        vertices[i, 0] = p.x
        vertices[i, 1] = p.y
        vertices[i, 2] = p.z

    return vertices, triangles


_object_handlers = {
    Box: box_to_mesh,
    Sphere: sphere_to_mesh,
    Cylinder: cylinder_to_mesh,
    Cone: cone_to_mesh,
    Mesh: mesh_to_mesh,
    Intersect: csg_to_mesh,
    Union: csg_to_mesh,
    Subtract: csg_to_mesh
}


def to_mesh(primitive):

    try:
        handler = _object_handlers[primitive.__class__]
    except KeyError:
        raise ValueError("Unrecognised Raysect primitive, '{}'.".format(primitive.__class__))

    return handler(primitive)
