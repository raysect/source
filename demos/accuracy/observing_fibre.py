
import numpy as np
from math import pi

from raysect.core import translate, rotate
from raysect.primitive import Sphere
from raysect.optical import World
from raysect.optical.observer import PowerPipeline0D, FibreOptic
from raysect.optical.material.emitter import UnityVolumeEmitter


samples = 100000

sphere_radius = 0.5
fibre_distance = 2
fibre_radius = 0.005
fibre_half_angle = 20  # degrees

min_wl = 400
max_wl = 401

# sanity checks!
if fibre_distance <= sphere_radius:
    raise ValueError("The fibre must be outside the sphere.")
cone_radius_on_axis = np.tan(np.deg2rad(fibre_half_angle)) * fibre_distance
if cone_radius_on_axis <= sphere_radius + fibre_radius:
    raise ValueError("The fibre's acceptance angle must be large enough to see all of the sphere's radiance.")

# set-up scenegraph
world = World()
emitter = Sphere(radius=sphere_radius, parent=world, material=UnityVolumeEmitter())

# The observer plane covers 1 side of a cube - to work out total power, multiply result by 6
fibre_power = PowerPipeline0D(accumulate=False)
optical_fibre = FibreOptic([fibre_power], acceptance_angle=fibre_half_angle, radius=fibre_radius,
                           min_wavelength=min_wl, max_wavelength=max_wl,
                           spectral_bins=1, pixel_samples=samples,
                           parent=world, transform=rotate(0, 0, 0)*translate(0, 0, -fibre_distance))


approximate_fraction_of_sphere = np.pi*fibre_radius**2 / (4 * np.pi * fibre_distance**2)


# Emitter is a sphere volume emitter located at the origin
# Volume of the sphere is 4/3 * Pi * r^3, emission over 4 * pi
# UnityVolumeEmitter emits 1W/str/m^3/ x nm, where x is the wavelength interval, integrated over length

print("Starting observations with volume emitter...")
calculated_volume_emission = 16 / 3 * pi**2 * sphere_radius**3 * (max_wl - min_wl)


optical_fibre.observe()
measured_volume_emission = fibre_power.value.mean / approximate_fraction_of_sphere
measured_volume_error = fibre_power.value.error() / approximate_fraction_of_sphere

print()
print('Expected volume emission => {} W'.format(calculated_volume_emission))
print('Measured volume emission => {} +/- {} W'.format(measured_volume_emission, measured_volume_error))
print()
