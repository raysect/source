from raysect.core.acceleration import Unaccelerated
from raysect.optical import World, translate, rotate, Point, Vector, Normal, Ray, d65_white, ConstantSF, SampledSF
from raysect.optical.observer.pinholecamera import PinholeCamera
from raysect.optical.material.emitter import UniformVolumeEmitter, UniformSurfaceEmitter, Checkerboard
from raysect.optical.material import debug
from raysect.optical.material.glass_libraries import schott
from raysect.primitive import Box, Sphere, Subtract
from raysect.primitive.mesh import Mesh, Triangle
from raysect.primitive.mesh import import_obj
from matplotlib.pyplot import *
from numpy import array
from time import time

BASE_PATH = "/home/alex/Shared/alex/work/MAST/mesh/MAST-M9"

# Matt's STL test files, converted to obj (not public, we need to identify some CC files for demo purposes)
mesh_list = [
    # "MAST-M9-BEAM DUMPS + GDC",
     "MAST-M9-CENTRE COLUMN ARMOUR",
    # "MAST-M9-LOWER COIL ARMOUR",
    # "MAST-M9-LOWER COILS",
    # "MAST-M9-LOWER ELM COILS",
    # "MAST-M9_P3_COILS_LOW_RES",
    # "MAST-M9-UPPER COIL ARMOUR",
    # "MAST-M9-UPPER COILS",
    # "MAST-M9-UPPER ELM COILS",
    # MAST-M9-MAST VESSEL"
]

world = World()

# obj convert to raysect optimised mesh
# for name in mesh_list:
#
#     filename = BASE_PATH + "/obj/" + name + ".obj"
#     print("Reading {}...".format(filename))
#     mesh = import_obj(
#         filename, scaling=1e-3, # kdtree_min_triangles=1000000,
#         # parent=world,
#         transform=rotate(0, 90, 0),
#         # material=schott("LF5G19")
#         material=debug.Light(Vector(0.2, 0.0, 1.0), 0.4)
#     )
#
#     filename = BASE_PATH + "/rsom/" + name + ".rsom"
#     print("Writing {}...".format(filename))
#     mesh.dump(filename)
#     print()

# raysect mesh format (pickle)
for name in mesh_list:
    filename = BASE_PATH + "/rsom/" + name + ".rsom"
    print("Reading {}...".format(filename))
    mesh = Mesh(
        parent=world,
        transform=rotate(0, 90, 0),
        material=debug.Light(Vector(0.2, 0.0, 1.0), 0.4)
    )
    mesh.load(filename)


for i in range(5):
    for j in range(5):
        if i != 2 or j != 2:
            Mesh(instance=mesh,
                 parent=world,
                 transform=translate((i-2)*6, (j-2)*6, 0)*rotate(0, 90, 0),
                 material=debug.Light(Vector(0.2, 0.0, 1.0), 0.4)
            )

mesh.load(BASE_PATH + "/rsom/MAST-M9_P3_COILS_LOW_RES.rsom")


print("Rendering...")
# Subtract(mesh, Box(Point(-5, -5, -5), Point(5, 5, 0)), material=debug.Light(Vector(0.2, 0.0, 1.0), 0.4), parent=world)

Box(Point(-50, -50, 50), Point(50, 50, 50.1), world, material=Checkerboard(4, d65_white, d65_white, 0.4, 0.8))
Box(Point(-100, -100, -100), Point(100, 100, 100), world, material=UniformSurfaceEmitter(d65_white, 0.1))

ion()
camera = PinholeCamera(fov=60, parent=world, transform=translate(0, 0, -25) * rotate(0, 0, 0), process_count=4)
camera.ray_max_depth = 15
camera.rays = 1
camera.spectral_samples = 15
camera.pixels = (128, 128)
camera.display_progress = True
camera.display_update_time = 5
camera.super_samples = 3
camera.observe()

ioff()
camera.save("demo_mesh_render.png")
camera.display()
show()

