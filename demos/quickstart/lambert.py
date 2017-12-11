
# External imports
import matplotlib.pyplot as plt

# Raysect imports
from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.optical.observer import PinholeCamera, RGBPipeline2D
from raysect.optical.material import Lambert, Checkerboard
from raysect.optical.library import schott
from raysect.primitive import Sphere, Box


# 1. Create Primitives
# --------------------

# Box defining the ground plane
ground = Box(lower=Point3D(-50, -1.51, -50), upper=Point3D(50, -1.5, 50), material=Lambert())

# checker board wall that acts as emitter
emitter = Box(lower=Point3D(-10, -10, 10), upper=Point3D(10, 10, 10.1),
              material=Checkerboard(4, d65_white, d65_white, 0.1, 2.0), transform=rotate(45, 0, 0))

# Sphere
# Note that the sphere must be displaced slightly above the ground plane to prevent numerically issues that could
# cause a light leak at the intersection between the sphere and the ground.
sphere = Sphere(radius=1.5, transform=translate(0, 0.0001, 0), material=schott("N-BK7"))


# 2. Add Observer
# ---------------

# Process the ray-traced spectra with the RGB pipeline.
rgb = RGBPipeline2D()

# camera
camera = PinholeCamera((512, 512), pipelines=[rgb], transform=translate(0, 10, -10) * rotate(0, -45, 0))

# camera - pixel sampling settings
camera.fov = 45
camera.pixel_samples = 250

# camera - ray sampling settings
camera.spectral_rays = 1
camera.spectral_bins = 20
camera.ray_max_depth = 100
camera.ray_extinction_prob = 0.1
camera.min_wavelength = 375.0
camera.max_wavelength = 740.0


# 3. Build Scenegraph
# -------------------

world = World()

sphere.parent = world
ground.parent = world
emitter.parent = world
camera.parent = world


# 4. Observe()
# ------------

plt.ion()
camera.observe()

plt.ioff()
rgb.save("render.png")
rgb.display()
plt.show()
