
# External imports
import matplotlib.pyplot as plt

# Raysect imports
from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.optical.observer import FibreOptic, PowerPipeline0D, SpectralPowerPipeline0D
from raysect.optical.material import Lambert, Checkerboard
from raysect.optical.library import schott
from raysect.primitive import Sphere, Box
from raysect.core.math import Vector3D


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


# 2. Build Scenegraph
# -------------------

world = World()

sphere.parent = world
ground.parent = world
emitter.parent = world


# 3. Add Observer
# ---------------

spectra = SpectralPowerPipeline0D()
power = PowerPipeline0D()
fibre = FibreOptic([spectra, power], acceptance_angle=2, radius=0.001, transform=translate(0, 0, -5), parent=world)
fibre.spectral_bins = 500
fibre.spectral_rays = 1
fibre.pixel_samples = pixel_samples=1000


# 4. Observe()
# ------------

plt.ion()
fibre.observe()


# You can access the spectral data directly to make your own plots.
plt.figure()
plt.plot(spectra.wavelengths, spectra.samples.mean)
plt.ioff()
plt.show()
