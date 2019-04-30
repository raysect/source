
from scipy.spatial import KDTree
from raysect.primitive import Mesh


def weld_vertices(mesh, tolerance=1E-6):
    """
    Welds the vertices of a mesh together within the specified tolerance.

    :param Mesh mesh: Mesh instance to weld.
    :param float tolerance: Distance within which vertices are assumed identical.
    :return: A new Mesh instance with welded vertices.
    """

    vertices = mesh.data.vertices
    num_vertices = len(vertices)
    triangles = mesh.data.triangles
    num_triangles = len(triangles)

    # construct KD-Tree of original vertex set
    kdtree = KDTree(vertices)

    # cycle through old vertices and only add them if greater than tolerance
    new_vertices = []
    unique_ids = []
    for i in range(num_vertices):
        unique_id = kdtree.query_ball_point(vertices[i], tolerance)[0]
        if not unique_id in unique_ids:
            new_vertices.append(vertices[unique_id].tolist())

    new_kdtree = KDTree(new_vertices)

    new_triangles = []
    for i in range(num_triangles):
        v1, v2, v3 = triangles[i]

        dist, new_v1 = new_kdtree.query(vertices[v1])
        if dist > tolerance:
            raise RuntimeError('An error has occurred')
        dist, new_v2 = new_kdtree.query(vertices[v2])
        if dist > tolerance:
            raise RuntimeError('An error has occurred')
        dist, new_v3 = new_kdtree.query(vertices[v3])
        if dist > tolerance:
            raise RuntimeError('An error has occurred')

        new_triangles.append([new_v1, new_v2, new_v3])

    return Mesh(new_vertices, new_triangles, smoothing=mesh.data.smoothing, closed=mesh.data.closed,
                parent=mesh.parent, transform=mesh.transform, material=mesh.material, name=mesh.name)
