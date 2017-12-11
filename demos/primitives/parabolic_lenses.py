
import matplotlib.pyplot as plt

from raysect.optical import World, translate, rotate, Point3D, d65_white, ConstantSF
from raysect.optical.observer import PinholeCamera
from raysect.optical.material.emitter import Checkerboard
from raysect.optical.material import Lambert, Dielectric
from raysect.optical.library import schott, Gold, RoughCopper

from raysect.primitive import Box, Sphere
from raysect.primitive.parabola import Parabola


rotation = 90.0

# Instantiate world object
world = World()

# Create lens objects
# Parabola(radius=0.5, height=0.1, parent=world, material=schott("N-BK7"), transform=translate(0.05, 0, 0) * rotate(90, 0, 0))
# Parabola(radius=0.1, height=0.1, parent=world, material=Dielectric(ConstantSF(1.4), ConstantSF(0.01)), transform=translate(0.05, 0, 0) * rotate(0, 0, 0))
# Parabola(radius=0.25, height=0.5, parent=world, material=Gold(), transform=translate(0, 0, 0)*rotate(0, -90, 0))
Parabola(radius=0.1, height=0.2, parent=world, material=schott("N-BK7"), transform=translate(0, 0, 0)*rotate(0, 100, 0))
# Sphere(radius=0.1, parent=world, material=Dielectric(ConstantSF(1.0), ConstantSF(0.02)), transform=translate(0, 0, 0) * rotate(0, 0, 0))
# Parabola(radius=1000, height=0.1, parent=world, material=schott("N-BK7"), transform=translate(0, 0, 0) * rotate(0, 0, 0))
# Parabola(radius=0.1, height=0.05, parent=world, material=Lambert(), transform=translate(0, 0, 0) * rotate(270, 0, 0))

# Background Checkerboard
Box(Point3D(-50.0, -50.0, 50), Point3D(50.0, 50.0, 50.1), world, material=Checkerboard(10, d65_white, d65_white, 0.4, 0.8))

# Instantiate camera object, and configure its settings.
plt.ion()
camera = PinholeCamera((256, 256), fov=45, parent=world, transform=translate(0.5, 0, -0.5) * rotate(45, 0, 0))
camera.pixel_samples = 50
camera.spectral_rays = 1
camera.spectral_bins = 20

# Start ray tracing
camera.observe()
plt.ioff()
plt.show()
