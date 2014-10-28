

from raysect.core.acceleration import Unaccelerated
from raysect.optical import World, translate, rotate, Point, Vector, Ray, d65_white, ConstantSF, SampledSF
from raysect.optical.observer.pinholecamera import PinholeCamera
from raysect.optical.material.emitter import UniformVolumeEmitter, UniformSurfaceEmitter, Checkerboard
from raysect.optical.material.glass import Glass, BK7, Sellmeier
from raysect.primitive import Sphere, Box, Cylinder, Union, Intersect, Subtract
from matplotlib.pyplot import *

from raysect.optical.optical_components.sphericalsinglets import planar_convex_lens

# kludge to fix matplotlib 1.4 ion() idiocy
import sys
sys.ps1 = 'SOMETHING'

world = World()

rota = 45
rotb = 0

planar_convex_lens(15, 3, parent=world, transform=translate(0, 0, 0) * rotate(90.0, 0.0, 0.0))



Box(Point(-50.0, -50.0, 50.0), Point(50.0, 50.0, 50.1), world, material=Checkerboard(4.0, d65_white, d65_white, 0.4, 0.8))

ion()
camera = PinholeCamera(fov=45, parent=world, transform=translate(0, 0, -8) * rotate(0, 0, 0))
camera.ray_max_depth = 15
camera.rays = 1
camera.spectral_samples = 15
camera.pixels = (512, 512)
camera.display_progress = True
camera.display_update_time = 10


camera.observe()

ioff()
camera.save("render.png")
camera.display()
show()

