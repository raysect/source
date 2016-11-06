
from raysect.core import Point3D, translate, rotate_x, rotate_basis, Vector3D
from raysect.primitive import Sphere, Box, Parabola
from raysect.primitive.csg import Subtract
from raysect.optical import World, ConstantSF
from raysect.optical.material.debug import PerfectReflectingSurface
from raysect.optical.observer.nonimaging import RectangularPixel
from raysect.optical.material.emitter import UnityVolumeEmitter, UniformSurfaceEmitter
from math import sqrt, pi


# set-up scenegraph
world = World()

sphere_radius = 1
cube_size = 5

# sanity check!
if cube_size <= 2 * sphere_radius:
    raise ValueError("The cube dimensions must be larger that the sphere.")

# Emitter is a sphere volume emitter located at the origin
# Volume of the sphere is 4/3 * Pi * r^3 => 4/3 Pi
# UnityVolumeEmitter emits 1W/str/m^3/ x nm, where x is the wavelength interval
# Therefore total emission is 1 W * 4Pi str * 4/3 Pi m^3 per x nm => 16/3 Pi^2 W / x nm => 52.6379 W/ x nm

# emitter = Sphere(radius=sphere_radius, parent=world, material=UnityVolumeEmitter())
# calculated_emission = 16 / 3 * pi**2 * sphere_radius**3


# Emitter is a sphere volume emitter located at the origin
# Volume of the sphere is 4/3 * Pi * r^3 => 4/3 Pi
# UnityVolumeEmitter emits 1W/str/m^3/ x nm, where x is the wavelength interval
# Therefore total emission is 1 W * 4Pi str * 4/3 Pi m^3 per x nm => 16/3 Pi^2 W / x nm => 52.6379 W/ x nm

emitter = Sphere(radius=sphere_radius, parent=world, material=UniformSurfaceEmitter(ConstantSF(1.0)))
calculated_emission = 8 * pi**2 * sphere_radius**2

# The observer plane covers 1 side of a cube - to work out total power, multiply result by 6
observing_plane = RectangularPixel(width=cube_size, height=cube_size, spectral_samples=1, pixel_samples=1000000,
                                    parent=world, transform=translate(0, 0, cube_size/2.0)*rotate_basis(Vector3D(0, 0, -1), Vector3D(1, 0, 0)))


print("Starting observations")
observing_plane.observe()
measured_emission = 6 * observing_plane.spectrum.samples[0] * cube_size**2 * 2 * pi

print()
print('Expected emission => {} W'.format(calculated_emission))
print('Measured emission => {} W'.format(measured_emission))




