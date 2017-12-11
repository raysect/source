
import matplotlib.pyplot as plt

from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.optical.observer import PinholeCamera
from raysect.optical.material.emitter import Checkerboard
from raysect.optical.library import schott

from raysect.primitive import Box


# Import the new lens classes
from raysect.primitive.lens.spherical import *

rotation = 90.0

# Instantiate world object
world = World()

# Create lens objects
BiConvex(0.0254, 0.0052, 0.0506, 0.0506, parent=world, transform=translate(0.02, 0.02, 0) * rotate(rotation, 0.0, 0.0), material=schott("N-BK7"))
BiConcave(0.0254, 0.0030, 0.052, 0.052, parent=world, transform=translate(-0.02, 0.02, 0) * rotate(rotation, 0.0, 0.0), material=schott("N-BK7"))
PlanoConvex(0.0254, 0.0053, 0.0258, parent=world, transform=translate(0.02, -0.02, 0) * rotate(rotation, 0.0, 0.0), material=schott("N-BK7"))
PlanoConcave(0.0254, 0.0035, 0.0257, parent=world, transform=translate(-0.02, -0.02, 0) * rotate(rotation, 0.0, 0.0), material=schott("N-BK7"))
Meniscus(0.0254, 0.0036, 0.0321, 0.0822, parent=world, transform=translate(0, 0, 0) * rotate(rotation, 0.0, 0.0), material=schott("N-BK7"))

# Background Checkerboard
Box(Point3D(-50.0, -50.0, 0.1), Point3D(50.0, 50.0, 0.2), world, material=Checkerboard(0.01, d65_white, d65_white, 0.4, 0.8))

# Instantiate camera object, and configure its settings.
plt.ion()
camera = PinholeCamera((512, 512), fov=45, parent=world, transform=translate(0, 0, -0.1) * rotate(0, 0, 0))
camera.pixel_samples = 100
camera.spectral_rays = 1
camera.spectral_bins = 20

# Start ray tracing
camera.observe()

camera.pipelines[0].save("lens_render.png")
plt.ioff()
plt.show()
