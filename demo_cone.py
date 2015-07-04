from raysect.core.acceleration import Unaccelerated
from raysect.optical import World, translate, rotate, Point, Vector, Ray, d65_white, ConstantSF, SampledSF
from raysect.optical.observer.pinholecamera import PinholeCamera
from raysect.optical.material.emitter import UniformVolumeEmitter, UniformSurfaceEmitter, Checkerboard
from raysect.optical.material.dielectric import Dielectric, Sellmeier
from raysect.optical.material.glass_libraries import schott
from raysect.primitive import Sphere, Box, Cylinder, Union, Intersect, Subtract, Cone
from matplotlib.pyplot import *
from numpy import array

import sys
sys.ps1 = 'SOMETHING'

world = World()

cone = Cone(material=schott("N-BK7"))

cyl_x = Cylinder(1, 4.2, transform=rotate(90, 0, 0)*translate(0, 0, -2.1))

Box(Point(-50, -50, 50), Point(50, 50, 50.1), world, material=Checkerboard(4, d65_white, d65_white, 0.4, 0.8))
Box(Point(-100, -100, -100), Point(100, 100, 100), world, material=UniformSurfaceEmitter(d65_white, 0.1))

ion()
camera = PinholeCamera(fov=45, parent=world, transform=translate(0, 0, -4) * rotate(0, 0, 0))
camera.ray_max_depth = 10
camera.rays = 10
camera.spectral_samples = 1
camera.pixels = (1024, 1024)
camera.display_progress = True
camera.display_update_time = 10
camera.observe()

ioff()
camera.save("render.png")
camera.display()
show()

