
from raysect.optical import World, translate, rotate, Point, d65_white
from raysect.optical.observer.pinholecamera import PinholeCamera
from raysect.optical.material.emitter import Checkerboard
from raysect.primitive import Box
from matplotlib.pyplot import *
import sys
sys.ps1 = 'SOMETHING'

# Import the new lens classes
from raysect.optical.optical_components.thorlabs import Thorlabs

# Instantiate world object
world = World()
thorlabs_components = Thorlabs()

# Create a lens object
thorlabs_components("LC1259", parent=world, transform=translate(-0.015, 0.0235, 0.01) * rotate(0, 0.0, 0.0))
thorlabs_components("LC1258", parent=world, transform=translate(-0.015, 0, 0.01) * rotate(0, 0.0, 0.0))
thorlabs_components("LC1254", parent=world, transform=translate(-0.015, -0.0235, 0.01) * rotate(0, 0.0, 0.0))
# planar_convex_lens(0.1, 0.02, parent=world, transform=translate(-0.015, 0, 0) * rotate(0, 0.0, 0.0))
thorlabs_components("LA1289", parent=world, transform=translate(0.015, 0.0235, 0) * rotate(0, 0.0, 0.0))
thorlabs_components("LA1213", parent=world, transform=translate(0.015, 0, 0) * rotate(0, 0.0, 0.0))
thorlabs_components("LA1304", parent=world, transform=translate(0.015, -0.0235, 0) * rotate(0, 0.0, 0.0))

# Background Checkerboard
Box(Point(-10.0, -10.0, 0.10), Point(10.0, 10.0, 0.4), world, material=Checkerboard(0.01, d65_white, d65_white, 0.4, 0.8))

# Instantiate camera object, and configure its settings.
ion()
camera = PinholeCamera(fov=45, parent=world, transform=translate(0, 0, -0.085) * rotate(0, 0, 0))
camera.ray_max_depth = 15
camera.rays = 1
camera.spectral_samples = 15
camera.pixels = (512, 512)
camera.display_progress = True
camera.display_update_time = 10

# Start ray tracing
camera.observe()

ioff()
camera.save("render.png")
camera.display()
show()