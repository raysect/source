
import os
import time
from matplotlib.pyplot import *
import numpy as np

from raysect.optical import World, translate, rotate, Point3D
from raysect.optical.library import RoughTitanium
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.primitive import Box
from raysect.optical.material import CylindricalRegularEmitter

"""
CylindricalRegularEmitter Demonstration
---------------------------------------

This file demonstrates how to use CylindricalRegularEmitter material to effectively
integrate through the emission profiles defined on a regular grid.

The same emission profile as in RegularGridBox demo is defined here in cylindrical
coordinates. Due to rectangular shape of the emission profile, we cannot use
RegularGridCylinder, so here we assign CylindricalRegularEmitter material to a Box.
"""

# grid parameters
xmin = ymin = -1.
xmax = ymax = 1.
zmin = -0.25
zmax = 0.25
rmin = 0
rmax = np.sqrt(2.)
r, dr = np.linspace(rmin, rmax, 201, retstep=True)
integration_step = 0.05
r = r[:-1] + 0.5 * dr  # moving to the grid cell centers
grid_shape = (200, 1, 1)
grid_steps = (dr, 360, zmax - zmin)  # axisymmetric emission

# spectral emission profile
min_wavelength = 375.
max_wavelength = 740.
spectral_points = 50
wavelengths, delta_wavelength = np.linspace(min_wavelength, max_wavelength, spectral_points, retstep=True)
wvl_centre = 0.5 * (max_wavelength + min_wavelength)
wvl_range = min_wavelength - max_wavelength
shift = 2 * (wavelengths - wvl_centre) / wvl_range + 5.
emission = np.cos(shift[None, None, None, :] * r[:, None, None, None])**4

# scene
world = World()
material = CylindricalRegularEmitter(grid_shape, grid_steps, emission, wavelengths, rmin=rmin)
material.integrator.step = integration_step
# we need to align the internal coordinate system of the box with the material grid
emitter = Box(lower=Point3D(xmin, ymin, 0), upper=Point3D(xmax, ymax, zmax - zmin), parent=world,
              material=material, transform=translate(0, 1, 0) * rotate(30, 0, 0) * translate(0, 0, zmin))
floor = Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=RoughTitanium(0.1))

# camera
rgb_pipeline = RGBPipeline2D(display_update_time=5)
sampler = RGBAdaptiveSampler2D(rgb_pipeline, min_samples=100, fraction=0.2)
camera = PinholeCamera((512, 512), parent=world, transform=translate(0, 4, -3.5) * rotate(0, -45, 0), pipelines=[rgb_pipeline], frame_sampler=sampler)
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
    rgb_pipeline.save("demo_regular_grid_cylinder_{}_pass_{}.png".format(timestamp, p))
    print()

# display final result
ioff()
rgb_pipeline.display()
show()
