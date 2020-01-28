
import os
import time
from matplotlib.pyplot import *
import numpy as np

from raysect.optical import World, translate, rotate, Point3D
from raysect.optical.library import RoughTitanium
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.primitive import Box
from raysect.optical.material import RegularGridBox

"""
RegularGridBox Demonstration
----------------------------

This file demonstrates how to use RegularGridBox to effectively integrate through
the emission profiles defined on a regular grid.

This demonstration uses exactly the same emission profile as
demos/material/volume.py demonstration. It is recomended to run the
volume.py demo first for better understanding.

Notice tenfold speedup compared to the Raysect's volume.py demo achieved by
pre-calculating the emission profile on a regular grid.

Even higher speedup can be achieved for smaller integration steps. Reducing the
integration step to 0.01 in both demos (along with doubling the grid resolution
in this demo) results in 50x speedup.

"""

# grid parameters
xmin = ymin = -1.
xmax = ymax = 1.
zmin = -0.25
zmax = 0.25
x, dx = np.linspace(xmin, xmax, 101, retstep=True)
y, dy = np.linspace(ymin, ymax, 101, retstep=True)
integration_step = 0.05
# x, dx = np.linspace(xmin, xmax, 201, retstep=True)
# y, dy = np.linspace(ymin, ymax, 201, retstep=True)
# integration_step = 0.01
x = x[:-1] + 0.5 * dx  # moving to the grid cell centers
y = y[:-1] + 0.5 * dy

# spectral emission profile
min_wavelength = 375.
max_wavelength = 740.
spectral_points = 50
wavelengths, delta_wavelength = np.linspace(min_wavelength, max_wavelength, spectral_points, retstep=True)
wvl_centre = 0.5 * (max_wavelength + min_wavelength)
wvl_range = min_wavelength - max_wavelength
shift = 2 * (wavelengths - wvl_centre) / wvl_range + 5.
radius = np.sqrt((x * x)[:, None] + (y * y)[None, :])
emission = np.cos(shift[None, None, None, :] * radius[:, :, None, None])**4

# scene
world = World()
emitter = RegularGridBox(emission, wavelengths, xmax=xmax - xmin, ymax=ymax - ymin, zmax=zmax - zmin,
                         step=integration_step, parent=world)
emitter.transform = translate(0, 1., 0) * rotate(30, 0, 0) * translate(xmin, ymin, zmin)
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
    rgb_pipeline.save("demo_1_cartesian_regular_grid_{}_pass_{}.png".format(timestamp, p))
    print()

# display final result
ioff()
rgb_pipeline.display()
show()
