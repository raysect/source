
from raysect.core import Point3D, translate, rotate_x
from raysect.primitive import Sphere, Box, Parabola
from raysect.primitive.csg import Subtract
from raysect.optical import World
from raysect.optical.material.debug import PerfectReflectingSurface
from raysect.optical.observer.nonimaging import CircularPixel
from raysect.optical.material.emitter import UnityVolumeEmitter
from math import sqrt


# set-up scenegraph
world = World()

# After manipulation the mirror has a focal point at (0, 0, 0), vertex at (0, 0, -5), base plane centred at (0, 0, 5).
radius = sqrt(200)
parabola = Parabola(radius=radius, height=10)
box = Box(lower=Point3D(-15, -15, 0), upper=Point3D(15, 15, 10.2))
parabolic_mirror = Subtract(box, parabola, transform=translate(0, 0, -5), material=PerfectReflectingSurface())
parabolic_mirror.parent = world

# The observer is a circular plane at the base of the parabola.
observing_plane = CircularPixel(radius=radius, spectral_samples=10, pixel_samples=1000,
                                parent=world, transform=translate(0, 0, -5.1))

# Emitter is a sphere located at the origin.
# Volume of the sphere is 4/3 * Pi * r^3 => 4/3 Pi
# UnityVolumeEmitter emits 1W/str/m^3/ x nm, where x is the wavelength interval
# Therefore total emission is 1 W * 4Pi str * 4/3 Pi m^3 per x nm => 16/3 Pi^2 W / x nm => 52.6379 W/ x nm
emitter = Sphere(radius=1.0, parent=world, material=UnityVolumeEmitter())

print("Starting observations")
observing_plane.observe()
measured_emission = observing_plane.spectrum.samples[0]

print()
print('Expected emission => 52.6379 W')
print('Measured emission => {} W'.format(measured_emission))

