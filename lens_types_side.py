
from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.optical.observer import PinholeCamera
from raysect.optical.material.emitter import Checkerboard
from raysect.optical.material.glass_libraries import schott
from raysect.optical.material import Gold, Silver

from raysect.primitive import Box
from matplotlib.pyplot import *
import sys
sys.ps1 = 'SOMETHING'

# Import the new lens classes
from raysect.primitive.optical.spherical import *

# Instantiate world object
world = World()

# Create a lens object
lens = BiConcave(0.0254, 0.003, 0.0396, 0.0396, parent=world, transform=translate(0, 0, 0) * rotate(90.0, 0.0, 0.0), material=schott("N-BK7"))
# lens = PlanoConcave(0.0254, 0.003, 0.0396, parent=world, transform=translate(0, 0, 0) * rotate(90.0, 0.0, 0.0), material=schott("N-BK7"))
# lens = PlanoConvex(0.0254, 0.006, 0.0396, parent=world, transform=translate(0, 0, 0) * rotate(90.0, 0.0, 0.0), material=schott("N-BK7"))
print(lens.edge_thickness)


# Background Checkerboard
Box(Point3D(-50.0, -50.0, -50.0), Point3D(50.0, 50.0, 50.0), world, material=Checkerboard(4.0, d65_white, d65_white, 0.4, 0.8))

# Instantiate camera object, and configure its settings.
ion()
camera = PinholeCamera(fov=45, parent=world, transform=translate(0, 0, -0.05) * rotate(0, 0, 0))
camera.ray_min_depth = 3
camera.ray_max_depth = 200
camera.ray_extinction_prob = 0.01
camera.pixel_samples = 50
camera.rays = 1
camera.spectral_samples = 15
camera.pixels = (128, 128)
camera.display_progress = True
camera.display_update_time = 10
camera.sub_sample = True

# Start ray tracing
camera.observe()

ioff()
camera.save("render.png")
camera.display()
show()