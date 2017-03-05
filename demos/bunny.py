
# External imports
from matplotlib.pyplot import *
import time

# Internal imports
from raysect.optical import World, translate, rotate, Point3D, d65_white, ConstantSF, Node
from raysect.optical.observer import PinholeCamera
from raysect.optical.material.dielectric import Sellmeier, Dielectric
from raysect.optical.material.emitter import UniformVolumeEmitter
from raysect.optical.material import Lambert
from raysect.primitive import Box, Subtract
from raysect.primitive.mesh import Mesh
from raysect.primitive.mesh import import_obj
from raysect.optical.library import schott


"""
A Diamond Stanford Bunny on an Illuminated Glass Pedestal
---------------------------------------------------------

Bunny model source:
  Stanford University Computer Graphics Laboratory
  http://graphics.stanford.edu/data/3Dscanrep/
  Converted to obj format using MeshLab
"""

# DIAMOND MATERIAL
diamond = Dielectric(Sellmeier(0.3306, 4.3356, 0.0, 0.1750**2, 0.1060**2, 0.0), ConstantSF(1.0))

world = World()

# BUNNY
# mesh = import_obj("./resources/stanford_bunny.obj", scaling=1, parent=world,
#                   transform=translate(0, 0, 0)*rotate(165, 0, 0), material=diamond)  # material=schott("LF5G19")

mesh = Mesh.from_file("./resources/stanford_bunny.rsm", parent=world,
                  transform=translate(0, 0, 0)*rotate(165, 0, 0), material=diamond)

# LIGHT BOX
padding = 1e-5
enclosure_thickness = 0.001 + padding
glass_thickness = 0.003

light_box = Node(parent=world)

enclosure_outer = Box(Point3D(-0.10 - enclosure_thickness, -0.02 - enclosure_thickness, -0.10 - enclosure_thickness),
                      Point3D(0.10 + enclosure_thickness, 0.0, 0.10 + enclosure_thickness))
enclosure_inner = Box(Point3D(-0.10 - padding, -0.02 - padding, -0.10 - padding),
                      Point3D(0.10 + padding, 0.001, 0.10 + padding))
enclosure = Subtract(enclosure_outer, enclosure_inner, material=Lambert(ConstantSF(0.2)), parent=light_box)

glass_outer = Box(Point3D(-0.10, -0.02, -0.10),
                  Point3D(0.10, 0.0, 0.10))
glass_inner = Box(Point3D(-0.10 + glass_thickness, -0.02 + glass_thickness, -0.10 + glass_thickness),
                  Point3D(0.10 - glass_thickness, 0.0 - glass_thickness, 0.10 - glass_thickness))
glass = Subtract(glass_outer, glass_inner, material=schott("N-BK7"), parent=light_box)

emitter = Box(Point3D(-0.10 + glass_thickness + padding, -0.02 + glass_thickness + padding, -0.10 + glass_thickness + padding),
              Point3D(0.10 - glass_thickness - padding, 0.0 - glass_thickness - padding, 0.10 - glass_thickness - padding),
              material=UniformVolumeEmitter(d65_white, 50), parent=light_box)

# CAMERA
ion()
camera = PinholeCamera((256, 256), fov=40, parent=world, transform=translate(0, 0.16, -0.4) * rotate(0, -12, 0))
camera.pixel_samples = 50
camera.spectral_rays = 15
camera.spectral_bins = 15
camera.observe()

ioff()
rgb = camera.pipelines[0]
rgb.save("stanford_bunny_{}.png".format(time.strftime("%Y-%m-%d_%H-%M-%S")))

show()

