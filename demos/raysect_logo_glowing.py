
from matplotlib.pyplot import *
from numpy import array

from raysect.primitive import Sphere, Box

from raysect.optical import World, Node, translate, rotate, Point3D, d65_white, ConstantSF, InterpolatedSF
from raysect.optical.observer import PinholeCamera
from raysect.optical.material import UniformVolumeEmitter, Lambert
from raysect.optical.library import *


world = World()

wavelengths = array([300, 490, 510, 590, 610, 800])
red_spectrum = array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]) * 0.98
green_spectrum = array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]) * 0.85
blue_spectrum = array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * 0.98
yellow_spectrum = array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]) * 0.85
cyan_spectrum = array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]) * 0.85
purple_spectrum = array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0]) * 0.95

red_glow = UniformVolumeEmitter(InterpolatedSF(wavelengths, red_spectrum))
green_glow = UniformVolumeEmitter(InterpolatedSF(wavelengths, green_spectrum))
blue_glow = UniformVolumeEmitter(InterpolatedSF(wavelengths, blue_spectrum))
yellow_glow = UniformVolumeEmitter(InterpolatedSF(wavelengths, yellow_spectrum))
cyan_glow = UniformVolumeEmitter(InterpolatedSF(wavelengths, cyan_spectrum))
purple_glow = UniformVolumeEmitter(InterpolatedSF(wavelengths, purple_spectrum))

# diffuse ground plane
Box(Point3D(-100, -100, -0.1), Point3D(100, 100, 0), world, transform=translate(0, 0, 0), material=Lambert(ConstantSF(1.0)))
# Box(Point3D(-100, -100, -0.1), Point3D(100, 100, 0), world, transform=translate(0, 0, 0), material=RoughAluminium(0.3))

# glass
node = Node(parent=world, transform=rotate(0, 0, 90))
Box(Point3D(-0.5, 0, -2.5), Point3D(0.5, 0.25, 0.5), node, rotate(0, 0, 0) * translate(0, 1, -0.500001), schott('N-BK7'))
Box(Point3D(-0.5, 0, -2.5), Point3D(0.5, 0.25, 0.5), node, rotate(0, 0, 60) * translate(0, 1, -0.500001), schott('N-BK7'))
Box(Point3D(-0.5, 0, -2.5), Point3D(0.5, 0.25, 0.5), node, rotate(0, 0, 120) * translate(0, 1, -0.500001), schott('N-BK7'))
Box(Point3D(-0.5, 0, -2.5), Point3D(0.5, 0.25, 0.5), node, rotate(0, 0, 180) * translate(0, 1, -0.500001), schott('N-BK7'))
Box(Point3D(-0.5, 0, -2.5), Point3D(0.5, 0.25, 0.5), node, rotate(0, 0, 240) * translate(0, 1, -0.500001), schott('N-BK7'))
Box(Point3D(-0.5, 0, -2.5), Point3D(0.5, 0.25, 0.5), node, rotate(0, 0, 300) * translate(0, 1, -0.500001), schott('N-BK7'))

# emitters
Box(Point3D(-0.48, 0.02, -2.48), Point3D(0.48, 0.23, 0.48), node, rotate(0, 0, 0) * translate(0, 1, -0.500001), red_glow)
Box(Point3D(-0.48, 0.02, -2.48), Point3D(0.48, 0.23, 0.48), node, rotate(0, 0, 60) * translate(0, 1, -0.500001), yellow_glow)
Box(Point3D(-0.48, 0.02, -2.48), Point3D(0.48, 0.23, 0.48), node, rotate(0, 0, 120) * translate(0, 1, -0.500001), green_glow)
Box(Point3D(-0.48, 0.02, -2.48), Point3D(0.48, 0.23, 0.48), node, rotate(0, 0, 180) * translate(0, 1, -0.500001), cyan_glow)
Box(Point3D(-0.48, 0.02, -2.48), Point3D(0.48, 0.23, 0.48), node, rotate(0, 0, 240) * translate(0, 1, -0.500001), blue_glow)
Box(Point3D(-0.48, 0.02, -2.48), Point3D(0.48, 0.23, 0.48), node, rotate(0, 0, 300) * translate(0, 1, -0.500001), purple_glow)

camera = PinholeCamera((1920, 1080), fov=150, parent=world, transform=translate(0, 0, -6.5) * rotate(0, 0, 0))

ion()
camera.ray_max_depth = 500
camera.ray_extinction_prob = 0.01
camera.pixel_samples = 250
camera.spectral_rays = 1
camera.spectral_bins = 21

# start ray tracing
ion()
p = 1
while not camera.render_complete:

    print("Rendering pass {}...".format(p))
    camera.observe()
    camera.pipelines[0].save("raysect_logo_glow_{}.png".format(p))
    print()

    p += 1

ioff()
camera.pipelines[0].display()