from raysect.core.acceleration import Unaccelerated
from raysect.optical import World, translate, rotate, Point, Vector, Normal, Ray, d65_white, ConstantSF, SampledSF
from raysect.optical.observer.pinholecamera import PinholeCamera
from raysect.optical.material.emitter import UniformVolumeEmitter, UniformSurfaceEmitter, Checkerboard
from raysect.optical.material import debug
from raysect.primitive import Box, Sphere
from raysect.primitive.mesh import Mesh, Triangle
from raysect.primitive.mesh import import_obj
from matplotlib.pyplot import *
from numpy import array
from time import time

# Matt's STL test files, converted to obj (not public, we need to identify some CC files for demo purposes)
mesh_list = [
    "/home/alex/work/ccfe/MAST_stl_files/obj/MAST-M9-BEAM DUMPS + GDC.obj",
    "/home/alex/work/ccfe/MAST_stl_files/obj/MAST-M9-CENTRE COLUMN ARMOUR.obj",
    "/home/alex/work/ccfe/MAST_stl_files/obj/MAST-M9-LOWER COIL ARMOUR.obj",
    "/home/alex/work/ccfe/MAST_stl_files/obj/MAST-M9-LOWER COILS.obj",
    "/home/alex/work/ccfe/MAST_stl_files/obj/MAST-M9-LOWER ELM COILS.obj",
    "/home/alex/work/ccfe/MAST_stl_files/obj/MAST-M9_P3_COILS_LOW_RES.obj",
    "/home/alex/work/ccfe/MAST_stl_files/obj/MAST-M9-UPPER COIL ARMOUR.obj",
    "/home/alex/work/ccfe/MAST_stl_files/obj/MAST-M9-UPPER COILS.obj",
    "/home/alex/work/ccfe/MAST_stl_files/obj/MAST-M9-UPPER ELM COILS.obj",
    # "/home/alex/work/ccfe/MAST_stl_files/obj/MAST-M9-MAST VESSEL.obj"
]

world = World()

# obj convert to raysect optimised mesh
# for filename in mesh_list:
#     print("Reading {}...".format(filename))
#     mesh = import_obj(
#         filename, scaling=1e-3, kdtree_max_depth=16,
#         parent=world,
#         transform=rotate(0, 90, 0),
#         material=debug.Light(Vector(0.2, 0.0, 1.0), 0.4)
#     )
#
#     print("Writing {}...".format(filename + ".rsom"))
#     mesh.to_file(filename + ".rsom")

# raysect mesh format (pickle)
for filename in mesh_list:
    print("Reading {}...".format(filename + ".rsom"))
    mesh = Mesh(
        parent=world,
        transform=rotate(0, 90, 0),
        material=debug.Light(Vector(0.2, 0.0, 1.0), 0.4)
    ).from_file(filename + ".rsom")


Box(Point(-50, -50, 50), Point(50, 50, 50.1), world, material=Checkerboard(4, d65_white, d65_white, 0.4, 0.8))
Box(Point(-100, -100, -100), Point(100, 100, 100), world, material=UniformSurfaceEmitter(d65_white, 0.1))

ion()
camera = PinholeCamera(fov=60, parent=world, transform=translate(0, 0, -4) * rotate(0, 0, 0), process_count=4)
camera.ray_max_depth = 15
camera.rays = 1
camera.spectral_samples = 15
camera.pixels = (1024, 1024)
camera.display_progress = True
camera.display_update_time = 5
camera.super_samples = 3
camera.observe()

ioff()
camera.save("demo_mesh_render.png")
camera.display()
show()

