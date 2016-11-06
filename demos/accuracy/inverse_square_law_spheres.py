

from raysect.core import Point3D, translate, rotate_x, rotate_basis, Vector3D
from raysect.primitive import Sphere
from raysect.optical import World
from raysect.optical.observer.nonimaging import SightLine
from raysect.optical.material.emitter import UnityVolumeEmitter
from math import sqrt
from numpy import pi


# set-up scenegraph
world = World()

# Emitter is a sphere located at the origin.
# Volume of the sphere is 4/3 * Pi * r^3 => 4/3 Pi
# UnityVolumeEmitter emits 1W/str/m^3/ x nm, where x is the wavelength interval
# Therefore total emission is 1 W * 4Pi str * 4/3 Pi m^3 per x nm => 16/3 Pi^2 W / x nm => 52.6379 W/ x nm
sphere1 = Sphere(radius=0.5, material=UnityVolumeEmitter(), transform=translate(0, 0, 1.0))
sphere2 = Sphere(radius=0.5, material=UnityVolumeEmitter(), transform=translate(0, 0, 10.0))
sphere3 = Sphere(radius=0.5, material=UnityVolumeEmitter(), transform=translate(0, 0, 100.0))
sphere4 = Sphere(radius=0.5, material=UnityVolumeEmitter(), transform=translate(0, 0, 1000.0))
spheres = [sphere1, sphere2, sphere3, sphere4]
distances = [1.0, 10.0, 100.0, 1000.0]

sl = SightLine(pixel_samples=1, parent=world)

print("Individual Sphere observations")
print()

for i, sphere in enumerate(spheres):
    sphere.parent = world

    sl.observe()
    measured_emission = sl.spectrum.samples[0]

    d = distances[i]
    solid_angle = pi * 0.5 ** 2 / (4 * pi * d**2)

    print('sphere distance d => {} m'.format(d))
    print('sphere solid angle => {:.4G} str'.format(solid_angle))
    print('Measured emission => {} W / str / x nm'.format(measured_emission))
    sphere.parent = None

print()
print("All Spheres together")
print()

for sphere in spheres:
    sphere.parent = world

sl.observe()
measured_emission = sl.spectrum.samples[0]
print('effective solid angle ???')
print('Measured emission => {} W / x nm'.format(measured_emission))
