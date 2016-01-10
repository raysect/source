
# External imports
import matplotlib.pyplot as plt

# Raysect imports
from raysect.optical import World, translate, rotate, Point3D, d65_white, Ray
from raysect.optical.material import Lambert, schott, Checkerboard
from raysect.primitive import Sphere, Box
from raysect.core.math import Vector3D
from raysect.optical.observer.point_observers import LineOfSight, OpticalFibre


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

fibre = OpticalFibre(spectral_rays=20, pixel_samples=100, transform=translate(0, 0, -5), parent=world)
fibre.observe()

spectrum = fibre.spectrum

plt.plot(spectrum.wavelengths, spectrum.samples)
plt.show()
