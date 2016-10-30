
from raysect.core import Point3D, translate, rotate_x, rotate_basis, Vector3D
from raysect.primitive import Sphere, Box, Parabola
from raysect.primitive.csg import Subtract
from raysect.optical import World
from raysect.optical.material.debug import PerfectReflectingSurface
from raysect.optical.observer.nonimaging import RectangularPixel
from raysect.optical.material.emitter import UnityVolumeEmitter
from math import sqrt


# set-up scenegraph
world = World()

# Emitter is a sphere located at the origin.
# Volume of the sphere is 4/3 * Pi * r^3 => 4/3 Pi
# UnityVolumeEmitter emits 1W/str/m^3/ x nm, where x is the wavelength interval
# Therefore total emission is 1 W * 4Pi str * 4/3 Pi m^3 per x nm => 16/3 Pi^2 W / x nm => 52.6379 W/ x nm
emitter = Sphere(radius=1.0, parent=world, material=UnityVolumeEmitter())

# The observer planes that make up a cube
observing_plane1 = RectangularPixel(width=3.0, height=3.0, spectral_samples=10, pixel_samples=100000,
                                    parent=world, transform=translate(0, 0, -3)*rotate_basis(Vector3D(0, 0, 1), Vector3D(1, 0, 0)))

observing_plane2 = RectangularPixel(width=3.0, height=3.0, spectral_samples=10, pixel_samples=100000,
                                    parent=world, transform=translate(0, 0, 3)*rotate_basis(Vector3D(0, 0, -1), Vector3D(1, 0, 0)))

observing_plane3 = RectangularPixel(width=3.0, height=3.0, spectral_samples=10, pixel_samples=100000,
                                    parent=world, transform=translate(-3, 0, 0)*rotate_basis(Vector3D(1, 0, 0), Vector3D(0, 0, 1)))

observing_plane4 = RectangularPixel(width=3.0, height=3.0, spectral_samples=10, pixel_samples=100000,
                                    parent=world, transform=translate(3, 0, 0)*rotate_basis(Vector3D(-1, 0, 0), Vector3D(0, 0, 1)))

observing_plane5 = RectangularPixel(width=3.0, height=3.0, spectral_samples=10, pixel_samples=100000,
                                    parent=world, transform=translate(0, -3, 0)*rotate_basis(Vector3D(0, 1, 0), Vector3D(0, 0, 1)))

observing_plane6 = RectangularPixel(width=3.0, height=3.0, spectral_samples=10, pixel_samples=100000,
                                    parent=world, transform=translate(0, 3, 0)*rotate_basis(Vector3D(0, -1, 0), Vector3D(0, 0, 1)))


print("Starting observations")
observing_plane1.observe()
observing_plane2.observe()
observing_plane3.observe()
observing_plane4.observe()
observing_plane5.observe()
observing_plane6.observe()

measured_emission = observing_plane1.spectrum.samples[0]
measured_emission += observing_plane2.spectrum.samples[0]
measured_emission += observing_plane3.spectrum.samples[0]
measured_emission += observing_plane4.spectrum.samples[0]
measured_emission += observing_plane5.spectrum.samples[0]
measured_emission += observing_plane6.spectrum.samples[0]

print()
print('Expected emission => 52.6379 W')
print('Measured emission => {} W'.format(measured_emission))




