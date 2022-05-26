# External imports
import os
from matplotlib.pyplot import *
import time

# Internal imports
from raysect.optical import World, translate, rotate, Point3D, d65_white, ConstantSF, Node
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.optical.material.emitter import UniformVolumeEmitter
from raysect.optical.material import Lambert
from raysect.primitive import Box, Subtract
from raysect.primitive.mesh import import_obj
from raysect.optical.library import schott

"""
A Glass Stanford Bunny on an Illuminated Glass Pedestal
-------------------------------------------------------

Bunny model source:
  Stanford University Computer Graphics Laboratory
  http://graphics.stanford.edu/data/3Dscanrep/
  Converted to obj format using MeshLab
"""

base_path = os.path.split(os.path.realpath(__file__))[0]

world = World()

#  BUNNY
mesh = import_obj(os.path.join(base_path, "../resources/stanford_bunny.obj"), parent=world,
                      transform=translate(0, 0, 0)*rotate(165, 0, 0), material=schott("N-BK7"))

# LIGHT BOX
padding = 1e-5
enclosure_thickness = 0.001 + padding
glass_thickness = 0.003

light_box = Node(parent=world)

enclosure_outer = Box(
    Point3D(-0.10 - enclosure_thickness, -0.02 - enclosure_thickness, -0.10 - enclosure_thickness),
    Point3D(0.10 + enclosure_thickness, 0.0, 0.10 + enclosure_thickness)
)

enclosure_inner = Box(
    Point3D(-0.10 - padding, -0.02 - padding, -0.10 - padding),
    Point3D(0.10 + padding, 0.001, 0.10 + padding)
)

enclosure = Subtract(enclosure_outer, enclosure_inner, material=Lambert(ConstantSF(0.2)), parent=light_box)

glass_outer = Box(
    Point3D(-0.10, -0.02, -0.10),
    Point3D(0.10, 0.0, 0.10)
)

glass_inner = Box(
    Point3D(-0.10 + glass_thickness, -0.02 + glass_thickness, -0.10 + glass_thickness),
    Point3D(0.10 - glass_thickness, 0.0 - glass_thickness, 0.10 - glass_thickness)
)

glass = Subtract(glass_outer, glass_inner, material=schott("N-BK7"), parent=light_box)

emitter = Box(
    Point3D(-0.10 + glass_thickness + padding, -0.02 + glass_thickness + padding, -0.10 + glass_thickness + padding),
    Point3D(0.10 - glass_thickness - padding, 0.0 - glass_thickness - padding, 0.10 - glass_thickness - padding),
    material=UniformVolumeEmitter(d65_white, 50),
    parent=light_box
)

# CAMERA
rgb = RGBPipeline2D(display_unsaturated_fraction=0.96, name="sRGB")
sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=2000, cutoff=0.01)
camera = PinholeCamera((1024, 1024), parent=world, transform=translate(0, 0.16, -0.4) * rotate(0, -12, 0), pipelines=[rgb], frame_sampler=sampler)
camera.spectral_rays = 15
camera.spectral_bins = 15
camera.pixel_samples = 250
camera.ray_max_depth = 500
camera.ray_extinction_min_depth = 3
camera.ray_extinction_prob = 0.01

# RAY TRACE
ion()
name = 'stanford_bunny'
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
render_pass = 1
while not camera.render_complete:

    print("Rendering pass {}...".format(render_pass))
    camera.observe()
    rgb.save("{}_{}_pass_{}.png".format(name, timestamp, render_pass))
    print()

    render_pass += 1

ioff()
rgb.display()
