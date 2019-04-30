
from raysect.core import Observer, World, Point3D, Vector3D, Node
from raysect.primitive.to_mesh import to_mesh


def _parse_nodes(node, mesh_list):

    for child in node.children:

        if type(child) == Node:
            _parse_nodes(child, mesh_list)
        elif isinstance(child, Observer):
            return
        else:
            mesh_list.append(to_mesh(child))


def visualise_scenegraph(world):
    """
    Creat an interactive Mayavi/VTK visualisation of a Raysect scene-graph.

    :param World world: The scene-graph to visualise interactively.
    """

    try:
        from mayavi import mlab

    except ImportError:
        print('Mayavi is a dependancy for Raysect visualisations. The Mayavi module was not found.')
        raise

    if not isinstance(world, World):
        raise TypeError("The visualisation function takes a Raysect World object as its argument.")

    fig = mlab.figure(size=(1024, 768), bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))

    meshes = []
    _parse_nodes(world, meshes)

    for mesh_description in meshes:

        vertices, triangles = mesh_description
        dx = vertices[:, 0]
        dy = vertices[:, 1]
        dz = vertices[:, 2]

        mlab.triangular_mesh(dx, dy, dz, triangles, color=(163 / 255.0, 163 / 255.0, 163 / 255.0),
                             figure=fig)  # , transparent=True, opacity=0.3)
