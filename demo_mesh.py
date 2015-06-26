from raysect.core.acceleration import Unaccelerated
from raysect.optical import World, translate, rotate, Point, Vector, Normal, Ray, d65_white, ConstantSF, SampledSF
from raysect.optical.observer.pinholecamera import PinholeCamera
from raysect.optical.material.emitter import UniformVolumeEmitter, UniformSurfaceEmitter, Checkerboard
from raysect.optical.material import debug
from raysect.primitive import Box, Sphere
from raysect.primitive.mesh import Mesh, Triangle
from matplotlib.pyplot import *
from numpy import array


from raysect.primitive.mesh.matt.stlmesh import StlMesh, BINARY
stlfile = "/home/alex/work/ccfe/MAST-M9-BEAM_DUMPS_+_GDC.stl"
mesh = StlMesh(stlfile, mode=BINARY)

triangles = []
for i in range(len(mesh.v0)):
    triangles.append(
        Triangle(
            Point(*(mesh.v0[i] / 1000)),
            Point(*(mesh.v1[i] / 1000)),
            Point(*(mesh.v2[i] / 1000))
        )
    )

world = World()

# test data:
# vertices = [
#     Point(1.4, -1, -1),   # left
#     Point(0, -1., 1.4),   # forward
#     Point(-1.4, -1, -1),  # right
#     Point(0, 1, 0)        # up
# ]
#
# normals = [
#     Normal(1.4, 0, -1),   # left
#     Normal(0, 0, 1.4),   # forward
#     Normal(-1.4, 0, -1),  # right
#     Normal(0, 1, 0)        # up
# ]
#
# #normals = [None, None, None, None]
#
# polygons = [
#     (0, 1, 2),
#     (0, 2, 3),
#     (0, 3, 1),
#     (2, 1, 3)
# ]
#
# triangles = []
# for i1, i2, i3 in polygons:
#     triangles.append(
#         Triangle(
#             vertices[i1],
#             vertices[i2],
#             vertices[i3],
#             normals[i1],
#             normals[i2],
#             normals[i3]
#         )
#     )

Mesh(triangles, True, world, translate(0, 0, 0) * rotate(0, 90, 0), debug.Light(Vector(0.2, 0.0, 1.0), 0.4))
# Mesh(vertices, polygons, world, translate(0, 0, 0)*rotate(0, 90, 0), debug.Normal('+x'))
# Mesh(vertices, polygons, world, translate(0, 0, 0)*rotate(0, 90, 0), debug.Exiting())

Box(Point(-50, -50, 50), Point(50, 50, 50.1), world, material=Checkerboard(4, d65_white, d65_white, 0.4, 0.8))
Box(Point(-100, -100, -100), Point(100, 100, 100), world, material=UniformSurfaceEmitter(d65_white, 0.1))

ion()
camera = PinholeCamera(fov=60, parent=world, transform=translate(0, 0, -4) * rotate(0, 0, 0), process_count=3)
camera.ray_max_depth = 15
camera.rays = 1
camera.spectral_samples = 15
camera.pixels = (256, 256)
camera.display_progress = True
camera.display_update_time = 5
camera.super_samples = 1
camera.observe()

ioff()
camera.save("demo_mesh_render.png")
camera.display()
show()

