
import os
from math import pi, sqrt

from raysect.primitive import Sphere, import_obj
from raysect.optical import World, ConstantSF
from raysect.optical.observer import MeshCamera, PowerPipeline1D
from raysect.optical.material.emitter import UnityVolumeEmitter, UniformSurfaceEmitter


samples = 100000

sphere_radius = 0.5
cube_size = 2

min_wl = 400
max_wl = 401

# sanity check!
if cube_size <= 2 * sphere_radius:
    raise ValueError("The cube dimensions must be larger that the sphere.")

# set-up scenegraph
world = World()
emitter = Sphere(radius=sphere_radius, parent=world)


base_path = os.path.split(os.path.realpath(__file__))[0]
mesh = import_obj(os.path.join(base_path, "../resources/box_normals_inwards.obj"), scaling=2.0)

power = PowerPipeline1D(accumulate=False)
observer = MeshCamera(mesh, pipelines=[power], parent=world,
                     min_wavelength=min_wl, max_wavelength=max_wl,
                     spectral_bins=1, pixel_samples=samples)

# Emitter is a sphere volume emitter located at the origin
# Volume of the sphere is 4/3 * Pi * r^3, emission over 4 * pi
# UnityVolumeEmitter emits 1W/str/m^3/ x nm, where x is the wavelength interval, integrated over length

print("Starting observations with volume emitter...")
calculated_volume_emission = 16 / 3 * pi**2 * sphere_radius**3 * (max_wl - min_wl)

emitter.material = UnityVolumeEmitter()
observer.observe()
measured_volume_emission = power.frame.mean.sum()
measured_volume_error = power.frame.errors().sum() / sqrt(12)  # 12 triangles in mesh

# Emitter is a sphere surface emitter located at the origin
# Surface area of the sphere is 4 * Pi * r^2, lambert emitter
# UniformSurfaceEmitter is configured to emit 1W/str/m^2/ x nm, where x is the wavelength interval

print("Starting observations with surface emitter...")
calculated_surface_emission = 4 * pi**2 * sphere_radius**2 * (max_wl - min_wl)

emitter.material = UniformSurfaceEmitter(ConstantSF(1.0))
observer.observe()
measured_surface_emission = power.frame.mean.sum()
measured_surface_error = power.frame.errors().sum() / sqrt(12)  # 12 triangles in mesh

print()
print('Expected volume emission => {} W'.format(calculated_volume_emission))
print('Measured volume emission => {} +/- {} W'.format(measured_volume_emission, measured_volume_error))

print()
print('Expected surface emission => {} W'.format(calculated_surface_emission))
print('Measured surface emission => {} +/- {} W'.format(measured_surface_emission, measured_surface_error))
