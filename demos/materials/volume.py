
import os
import time
from matplotlib.pyplot import *
from numpy import sqrt, cos

from raysect.optical import World, translate, rotate, Point3D
from raysect.optical.library import RoughTitanium
from raysect.optical.material import InhomogeneousVolumeEmitter
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.primitive import Box
from raysect.core.workflow import MulticoreEngine


class CosGlow(InhomogeneousVolumeEmitter):

    def emission_function(self, point, direction, spectrum, world, ray, primitive, to_local, to_world):

        wvl_centre = 0.5 * (spectrum.max_wavelength + spectrum.min_wavelength)
        wvl_range = spectrum.min_wavelength - spectrum.max_wavelength
        shift = 2 * (spectrum.wavelengths - wvl_centre) / wvl_range
        radius = sqrt(point.x**2 + point.y**2)
        spectrum.samples += cos((shift + 5) * radius)**4
        return spectrum


# scene
world = World()
emitter = Box(Point3D(-1, -1, -0.25), Point3D(1, 1, 0.25), material=CosGlow(), parent=world, transform=translate(0, 1, 0) * rotate(30, 0, 0))
floor = Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=RoughTitanium(0.1))

# camera
rgb_pipeline = RGBPipeline2D(display_update_time=5)
sampler = RGBAdaptiveSampler2D(rgb_pipeline, min_samples=100, fraction=0.2)
camera = PinholeCamera((512, 512), parent=world, transform=translate(0, 4, -3.5) * rotate(0, -45, 0), pipelines=[rgb_pipeline], frame_sampler=sampler)
camera.spectral_bins = 15
camera.spectral_rays = 1
camera.pixel_samples = 200
camera.render_engine = MulticoreEngine(4)

# integration resolution
emitter.material.integrator.step = 0.05

# start ray tracing
os.nice(15)
ion()
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
for p in range(1, 1000):
    print("Rendering pass {}...".format(p))
    camera.observe()
    rgb_pipeline.save("demo_volume_{}_pass_{}.png".format(timestamp, p))
    print()

# display final result
ioff()
rgb_pipeline.display()
show()
