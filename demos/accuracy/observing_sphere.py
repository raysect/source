
from math import pi

from raysect.core import Point3D, translate, rotate_basis
from raysect.core.math.sampler import HemisphereCosineSampler, SphereSampler, HemisphereUniformSampler
from raysect.primitive import Sphere
from raysect.optical import World, ConstantSF
from raysect.optical.observer import PowerPipeline0D, Observer0D
from raysect.optical.material.emitter import UnityVolumeEmitter, UniformSurfaceEmitter, UniformVolumeEmitter


class ObservingSphere(Observer0D):

    def __init__(self, pipelines, radius=None, parent=None, transform=None, name=None,
                 render_engine=None, pixel_samples=None, samples_per_task=None, spectral_rays=None, spectral_bins=None,
                 min_wavelength=None, max_wavelength=None, ray_extinction_prob=None, ray_extinction_min_depth=None,
                 ray_max_depth=None, ray_importance_sampling=None, ray_important_path_weight=None):

        super().__init__(pipelines, parent=parent, transform=transform, name=name, render_engine=render_engine,
                         pixel_samples=pixel_samples, samples_per_task=samples_per_task, spectral_rays=spectral_rays,
                         spectral_bins=spectral_bins, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                         ray_extinction_prob=ray_extinction_prob, ray_extinction_min_depth=ray_extinction_min_depth,
                         ray_max_depth=ray_max_depth, ray_importance_sampling=ray_importance_sampling,
                         ray_important_path_weight=ray_important_path_weight)

        self._radius = radius or 1.0
        self._vector_sampler = HemisphereCosineSampler()
        self._sphere_sampler = SphereSampler()
        self._solid_angle = 2 * pi
        self._collection_area = 4 * pi * self._radius**2

    def _generate_rays(self, template, ray_count):

        origin_vectors = self._sphere_sampler(ray_count)
        directions = self._vector_sampler(ray_count)

        rays = []
        for n in range(ray_count):

            # calculate surface point
            origin = origin_vectors[n].copy()
            origin.length = self._radius
            origin = Point3D(*origin)

            # calculate surface normal
            normal = -origin_vectors[n]

            # transform sampling direction from surface space
            direction = directions[n].transform(rotate_basis(normal, normal.orthogonal()))

            # USE WITH HEMISPHERECOSINESAMPLER
            # cosine weighted distribution, projected area weight is
            # implicit in distribution, so set weight appropriately
            rays.append((template.copy(origin , direction), 0.5))

        return rays

    def _pixel_sensitivity(self):
        return self._solid_angle * self._collection_area


samples = 100000

emitting_sphere_radius = 0.5
collection_sphere_radius = 5

min_wl = 400
max_wl = 401

# sanity check!
if collection_sphere_radius <= emitting_sphere_radius:
    raise ValueError("The collecting sphere radius must be larger that the emitting sphere.")

power = PowerPipeline0D(accumulate=False)

# set-up scenegraph
world = World()
emitter = Sphere(radius=emitting_sphere_radius, parent=world, transform=translate(4, 0, 0))
observer = ObservingSphere([power], radius=collection_sphere_radius,
                           min_wavelength=min_wl, max_wavelength=max_wl,
                           spectral_bins=1, pixel_samples=samples,
                           parent=world)


# Emitter is a sphere volume emitter located at the origin
# Volume of the sphere is 4/3 * Pi * r^3, emission over 4 * pi
# UnityVolumeEmitter emits 1W/str/m^3/ x nm, where x is the wavelength interval, integrated over length

print("Starting observations with volume emitter...")
calculated_volume_emission = (4 * pi) * (4/3 * pi * emitting_sphere_radius**3) * (max_wl - min_wl)

emitter.material = UnityVolumeEmitter()
# emitter.material = UniformVolumeEmitter(ConstantSF(1.0))
observer.observe()

measured_volume_emission = power.value.mean
measured_volume_error = power.value.error()

volume_deviation = (measured_volume_emission - calculated_volume_emission) / measured_volume_error

# Emitter is a sphere surface emitter located at the origin
# Surface area of the sphere is 4 * Pi * r^2, lambert emitter
# UniformSurfaceEmitter is configured to emit 1W/str/m^2/ x nm, where x is the wavelength interval

print("Starting observations with surface emitter...")
calculated_surface_emission = pi * (4 * pi * emitting_sphere_radius**2) * (max_wl - min_wl)

emitter.material = UniformSurfaceEmitter(ConstantSF(1.0))
observer.observe()

measured_surface_emission = power.value.mean
measured_surface_error = power.value.error()

surface_deviation = (measured_surface_emission - calculated_surface_emission) / measured_surface_error

print()
print('Expected volume emission => {:0.2f} W'.format(calculated_volume_emission))
print('Measured volume emission => {:0.2f} +/- {:0.2f} W'.format(measured_volume_emission, measured_volume_error))
print('Deviation: {:0.2f}'.format(volume_deviation))

print()
print('Expected surface emission => {:0.2f} W'.format(calculated_surface_emission))
print('Measured surface emission => {:0.2f} +/- {:0.2f} W'.format(measured_surface_emission, measured_surface_error))
print('Deviation: {:0.2f}'.format(surface_deviation))
