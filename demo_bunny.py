from raysect.optical import World, translate, rotate, Point, Vector, Normal, Ray, d65_white, ConstantSF, SampledSF, Node
from raysect.optical.observer.pinholecamera import PinholeCamera
from raysect.optical.material.dielectric import Sellmeier, Dielectric
from raysect.optical.material.emitter import UniformVolumeEmitter
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.primitive import Box, Subtract
from raysect.primitive.mesh import Mesh
from raysect.primitive.mesh import import_obj
from matplotlib.pyplot import *
import time
from raysect.optical.material.glass_libraries import schott

# DIAMOND MATERIAL
diamond = Dielectric(Sellmeier(0.3306, 4.3356, 0.0, 0.1750**2, 0.1060**2, 0.0), ConstantSF(1.0), cutoff=1.0/256)

world = World()

# BUNNY
mesh = import_obj(
        "/home/alex/Desktop/Shared/alex/work/meshes/obj/bunny2.obj",
        scaling=1,
        parent=world,
        transform=translate(0, 0, 0)*rotate(165, 0, 0),
        material=diamond
        # material=schott("LF5G19")
    )

# LIGHT BOX
padding = 1e-5
enclosure_thickness = 0.001 + padding
glass_thickness = 0.003

light_box = Node(parent=world)

enclosure_outer = Box(Point(-0.10 - enclosure_thickness, -0.02 - enclosure_thickness, -0.10 - enclosure_thickness),
                      Point(0.10 + enclosure_thickness, 0.0, 0.10 + enclosure_thickness))
enclosure_inner = Box(Point(-0.10 - padding, -0.02 - padding, -0.10 - padding),
                      Point(0.10 + padding, 0.001, 0.10 + padding))
enclosure = Subtract(enclosure_outer, enclosure_inner, material=AbsorbingSurface(), parent=light_box)

glass_outer = Box(Point(-0.10, -0.02, -0.10),
                  Point(0.10, 0.0, 0.10))
glass_inner = Box(Point(-0.10 + glass_thickness, -0.02 + glass_thickness, -0.10 + glass_thickness),
                  Point(0.10 - glass_thickness, 0.0 - glass_thickness, 0.10 - glass_thickness))
glass = Subtract(glass_outer, glass_inner, material=schott("N-BK7"), parent=light_box)
glass.material.cutoff = 1.0/256

emitter = Box(Point(-0.10 + glass_thickness + padding, -0.02 + glass_thickness + padding, -0.10 + glass_thickness + padding),
              Point(0.10 - glass_thickness - padding, 0.0 - glass_thickness - padding, 0.10 - glass_thickness - padding),
              material=UniformVolumeEmitter(d65_white, 50), parent=light_box)

ion()
camera = PinholeCamera(fov=40, parent=world, transform=translate(0, 0.16, -0.4) * rotate(0, -12, 0), process_count=4)
camera.ray_max_depth = 15
camera.rays = 12
camera.spectral_samples = 2
camera.pixels = (128, 128)
camera.display_progress = True
camera.display_update_time = 15
camera.super_samples = 4
camera.observe()

ioff()
camera.save("renders/demo_mesh_render_{}.png".format(time.strftime("%Y-%m-%d_%H-%M-%S")))
camera.display()
show()

