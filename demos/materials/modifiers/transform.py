
# External imports
from matplotlib.pyplot import *
from numpy import sqrt, cos
import time

# Raysect imports
from raysect.optical import World, translate, rotate, Point3D
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.optical.material import VolumeTransform, InhomogeneousVolumeEmitter
from raysect.optical.library import RoughTitanium
from raysect.primitive import Box


# volume texture material
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

# boxes
box_unshifted = Box(
    Point3D(-1, -1, -0.25), Point3D(1, 1, 0.25),
    material=CosGlow(),
    parent=world, transform=translate(3.2, 1, 0) * rotate(-30, 0, 0)
)

box_down_shifted = Box(
    Point3D(-1, -1, -0.25), Point3D(1, 1, 0.25),
    material=VolumeTransform(CosGlow(), translate(0, 0.5, 0)),
    parent=world, transform=translate(1.1, 1, 0.8) * rotate(-10, 0, 0)
)

box_up_shifted = Box(
    Point3D(-1, -1, -0.25), Point3D(1, 1, 0.25),
    material=VolumeTransform(CosGlow(), translate(0, -0.5, 0)),
    parent=world, transform=translate(-1.1, 1, 0.8) * rotate(10, 0, 0)
)

box_rotated = Box(
    Point3D(-1, -1, -0.25), Point3D(1, 1, 0.25),
    material=VolumeTransform(CosGlow(), rotate(0, 90, 0)),
    parent=world, transform=translate(-3.2, 1, 0) * rotate(30, 0, 0)
)

floor = Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=RoughTitanium(0.1))

# camera
rgb_pipeline = RGBPipeline2D(display_update_time=5)
sampler = RGBAdaptiveSampler2D(rgb_pipeline, min_samples=100, fraction=0.2)
camera = PinholeCamera((512, 256), parent=world, transform=translate(0, 6.5, -10) * rotate(0, -30, 0), pipelines=[rgb_pipeline], frame_sampler=sampler)
camera.spectral_bins = 15
camera.spectral_rays = 1
camera.pixel_samples = 100

# integration resolution
box_unshifted.material.integrator.step = 0.05
box_down_shifted.material.material.integrator.step = 0.05
box_up_shifted.material.material.integrator.step = 0.05
box_rotated.material.material.integrator.step = 0.05

# start ray tracing
ion()
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
for p in range(1, 1000):
    print("Rendering pass {}...".format(p))
    camera.observe()
    rgb_pipeline.save("modifier_volume_transform_{}_pass_{}.png".format(timestamp, p))
    print()

# display final result
ioff()
rgb_pipeline.display()
show()
