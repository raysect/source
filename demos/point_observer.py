
# External imports
import matplotlib.pyplot as plt

# Raysect imports
from raysect.optical import World, translate, rotate, Point3D, d65_white, Ray
from raysect.optical.material import Lambert, Checkerboard
from raysect.optical.library import schott
from raysect.primitive import Sphere, Box
from raysect.optical.observer import FibreOptic, PowerPipeline0D, SpectralPipeline0D


# Box defining the ground plane
ground = Box(lower=Point3D(-50, -1.51, -50), upper=Point3D(50, -1.5, 50), material=Lambert())

# checker board wall that acts as emitter
emitter = Box(lower=Point3D(-10, -10, 10), upper=Point3D(10, 10, 10.1),
              material=Checkerboard(4, d65_white, d65_white, 0.1, 2.0), transform=rotate(45, 0, 0))

# Sphere
sphere = Sphere(radius=1.5, transform=translate(0, 0.0001, 0), material=schott("N-BK7"))


# 3. Build Scenegraph
# -------------------

world = World()
sphere.parent = world
ground.parent = world
emitter.parent = world


# 4. Observe()
# ------------

spectra = SpectralPipeline0D()
power = PowerPipeline0D()
fibre = FibreOptic([spectra, power], acceptance_angle=45, radius=0.0005, spectral_bins=500, spectral_rays=1, pixel_samples=1000, transform=translate(0, 0, -5), parent=world)
fibre.observe()

