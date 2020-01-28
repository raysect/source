
import os
import time
from matplotlib.pyplot import *
import numpy as np

from raysect.optical import World, translate, rotate, Point3D
from raysect.optical.library import RoughTitanium
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.primitive import Box
from raysect.optical.material import RegularGridCylinder

"""
Demonstration of discrete emitter
---------------------------------

This file demonstrates how to define the regular grid emitter with discrete spectrum.
"""

# axisymmetric cylindrical grid
zmin = -0.25
zmax = 0.25
rmin = 0
rmax = np.sqrt(2.)
r, dr = np.linspace(rmin, rmax, 401, retstep=True)
integration_step = 0.02
r = r[:-1] + 0.5 * dr  # moving to the grid cell centers

# spectral emission profile
wavelengths = np.array([410.2, 434.0, 486.1, 656.5])  # Balmer series in visible spectrum
min_wavelength = 380.
max_wavelength = 690.
x = min_wavelength + (r - rmin) / (rmax - rmin) * (max_wavelength - min_wavelength)
emission = np.zeros((r.size, 1, 1, 4))
sigma = 10.
for i, wl in enumerate(wavelengths):
    emission[:, 0, 0, i] = np.exp(-0.5 * ((x - wl) / sigma) ** 2) / (sigma * np.sqrt(2. * np.pi))  # emission must be in W / (m^3 str)

# scene
world = World()
emitter = RegularGridCylinder(emission, wavelengths, radius_outer=rmax, height=zmax - zmin, radius_inner=rmin,
                              step=integration_step, contineous=False, parent=world)  # discrete emitter
emitter.transform = translate(0, rmax, 0) * rotate(30, 0, 0) * translate(0, 0, zmin)
floor = Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=RoughTitanium(0.1))

# camera
rgb_pipeline = RGBPipeline2D(display_update_time=5)
sampler = RGBAdaptiveSampler2D(rgb_pipeline, min_samples=100, fraction=0.2)
camera = PinholeCamera((512, 512), parent=world, transform=translate(0, 5, -4.5) * rotate(0, -45, 0), pipelines=[rgb_pipeline], frame_sampler=sampler)
camera.min_wavelength = min_wavelength
camera.max_wavelength = max_wavelength
camera.spectral_bins = 15
camera.spectral_rays = 1
camera.pixel_samples = 200

# Here, ray spectral properties do not change during the rendering,
# so we build the cache before the first camera.observe() call to reduce memory consumption
# in multiprocess rendering.
emitter.material.cache_build(camera.min_wavelength, camera.max_wavelength, camera.spectral_bins)
# start ray tracing
os.nice(15)
ion()
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
for p in range(1, 1000):
    print("Rendering pass {}...".format(p))
    camera.observe()
    rgb_pipeline.save("demo_3_discrete_spectrum_{}_pass_{}.png".format(timestamp, p))
    print()

# display final result
ioff()
rgb_pipeline.display()
show()
