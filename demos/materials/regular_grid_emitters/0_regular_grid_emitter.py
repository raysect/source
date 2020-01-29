import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from matplotlib import pyplot as plt

from raysect.optical import InterpolatedSF, Ray
from raysect.optical.material import RegularGridEmitter

"""
Demonstration of RegularGridEmitter functionality
-------------------------------------------------

This file demonstrates basic functions of RegularGridEmitter, the emitter material created
to speed up volume integration through the emission profiles defined on a regular 3D spatial grids.
"""


""" Part 1: continuous spectrum. """
print('Part 1: continuous spectrum.\n')


# Let's define a simple 3x3x3 gird.
grid_shape = (3, 3, 3)  # grid shape
grid_steps = (1., 1., 1.)  # grid steps (in meters)


# Let's define the spectral range and the emission law.
min_wavelength = 375.
max_wavelength = 740.
spectral_points = 365
wavelengths = np.linspace(min_wavelength, max_wavelength, spectral_points)
xwl = (wavelengths - min_wavelength) / (max_wavelength - min_wavelength) * 3 * np.pi
cos2_law = np.cos(xwl)**2
sin2_law = np.sin(xwl)**2


# Assume that only 9 out of 27 grid cells emit (8 corners and a central cell).
emission = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], spectral_points))
emission[::2, ::2, ::2, :] = cos2_law[None, None, None, :]  # W / (m^3 str nm)
emission[1, 1, 1, :] = 2. * sin2_law  # W / (m^3 str nm)


# Let's define the emitting material.
material = RegularGridEmitter(grid_shape, grid_steps, emission, wavelengths)
# The emission profile is stored as a compressed sparse column matrix (scipy.sparse.csc_matrix).
# It can be passed as a csc_matrix (or any other scipy sparce matrix).
nvoxel = grid_shape[0] * grid_shape[1] * grid_shape[2]
emission_csc = csc_matrix(emission.reshape(nvoxel, spectral_points))
material = RegularGridEmitter(grid_shape, grid_steps, emission_csc, wavelengths)
# By default, the spectral emission profile is extrpolated outside the specified spectral range.
# If emission profile must be zero outside the spectral range, 'extrapolate' parameter must be set to False.
material.extrapolate = False


# Built-in function integrate(wl0, wl1) allows to integrate the emission profile in a spectral range
# between wl0 and wl1.
wl0 = 500.
wl1 = 650.
wl_integral = material.integrate(wl0, wl1)
# wl_integral is a one-column csc_matrix.
# We can compare the results with the results of InterpolatedSF.integrate()
isf_corner = InterpolatedSF(wavelengths, cos2_law)
print('RegularGridEmitter integral (corner): %g' % wl_integral[0, 0])
print('InterpolatedSF integral (corner): %g' % isf_corner.integrate(wl0, wl1))
print()
# To get the flat index of the central cell we can use voxel_index() function.
ivoxel = material.voxel_index(1, 1, 1)
isf_center = InterpolatedSF(wavelengths, 2. * sin2_law)
print('RegularGridEmitter integral (center): %g' % wl_integral[ivoxel, 0])
print('InterpolatedSF integral (center): %g' % isf_center.integrate(wl0, wl1))
print()


# Similar to SpectralFunction, RegularGridEmitter resamples spectral emission profile
# for the specific spectral properties of the ray: ray.min_wavelength, ray.max_wavelength, ray.bins.
# And similar to SpectralFunction, the resampled emission profile is cached.
# Both the special integrators and the built-in emission_function(), which is used only with general
# integrators, check if the cache is valid for the ray spectral properties before performing calculations
# and rebuild the cache if necessary.
# However, it is possible to build the cache manually with cache_build() function.
# Cache validity can be checked with cache_valid() function.
ray = Ray(min_wavelength=400., max_wavelength=700., bins=60)
print('Is the cache valid? %s' % material.cache_valid(ray.min_wavelength, ray.max_wavelength, ray.bins))
material.cache_build(ray.min_wavelength, ray.max_wavelength, ray.bins)
print('And now? %s' % material.cache_valid(ray.min_wavelength, ray.max_wavelength, ray.bins))
print()
# Note that calling the cache_build() function again with the same arguments will do nothing because
# it checks for cache validity first. However, it is possible to forcefully rebuild the cache
# by specifying a forced argument.
material.cache_build(ray.min_wavelength, ray.max_wavelength, ray.bins, forced=True)
# While the cache_build() is called automatically, it may be useful to call it in advance in the case of
# multi-process rendering in the scenarios when ray spectral properties do not change during the rendering
# (e.g., when dispersive rendering is turned off).
# When cache_build() is called after the camera.observe(), each process will create its own cache, but when
# cache_build() is called before camera.observe(), all processes will use the same cache while it's valid.
# Thus when working with large spatial grids, a lot of memory can be saved.


# Let's compare sampled emission with the original values.
sample_wavelengths, sample_delta = np.linspace(ray.min_wavelength, ray.max_wavelength, ray.bins + 1, retstep=True)
sample_wavelengths = sample_wavelengths[:-1] + 0.5 * sample_delta
samples_corner = np.zeros(sample_wavelengths.size)
material.add_emission_to_array(samples_corner, 0, 0, 0, 1.0)
samples_center = np.zeros(sample_wavelengths.size)
material.add_emission_to_array(samples_center, 1, 1, 1, 1.0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(wavelengths, cos2_law, label='Original (corner)')
ax.plot(wavelengths, 2. * sin2_law, label='Original (center)')
ax.plot(sample_wavelengths, samples_corner, ls='', marker='o', label='Sampled (corner)')
ax.plot(sample_wavelengths, samples_center, ls='', marker='o', label='Sampled (center)')
ax.set_xlabel('wavelength, nm')
ax.legend(loc=1, ncol=2, frameon=False, bbox_to_anchor=(1.0, 1.15))
fig.savefig('demo_0_regular_grid_emitter.png')


# Note that sampling conserves the wavelength integral.
print('Integral (%g nm - %g nm): %g' % (ray.min_wavelength, ray.max_wavelength,
                                        material.integrate(ray.min_wavelength, ray.max_wavelength)[0, 0]))
print('Sampled integral (%g nm - %g nm): %g' % (ray.min_wavelength, ray.max_wavelength,
                                                samples_corner.sum() * sample_delta))
print()


""" Part 2: Discrete spectrum. """
print('Part 2: Discrete spectrum.\n')


# Some materials (e.g. atoms) emit only at certain wavelengths.
# Assume that each cell from the 3x3x3 grid emit only at its own unique wavelength (spectral line).
wavelengths = np.linspace(min_wavelength, max_wavelength, nvoxel)
emission = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], nvoxel))
for i in range(grid_shape[0]):
    for j in range(grid_shape[1]):
        for k in range(grid_shape[2]):
            emission[i, j, k, i * grid_shape[0] * grid_shape[1] + j * grid_shape[1] + k] = 1.   # W / (m^3 str)
# In the case of discrete emitters, the emission must be provided in W/(m^3 str) (not in W/(m^3 str nm)).
# A special option 'continuous' must be set to False at initialisation.
material = RegularGridEmitter(grid_shape, grid_steps, emission, wavelengths, continuous=False)


# When integrating over wavelngth, this emitter acts similar to the delta-function.
# The integral between wl0 and wl1 is just a sum of the emissions at all spectral lines between wl0 and wl1,
# or zero if there are no spectral lines between wl0 and wl1.
total_integral = material.integrate(min_wavelength - 0.1, max_wavelength + 0.1).toarray().reshape(grid_shape)
print('Total integral for all cells:\n', total_integral)
partial_integral = material.integrate(500, 620).toarray().reshape(grid_shape)
print('Integral (500 nm - 600 nm) for all cells:\n', partial_integral)
print()


# Let's create a sample for the discrete emitter
ray = Ray(min_wavelength=400., max_wavelength=700., bins=150)
material.cache_build(ray.min_wavelength, ray.max_wavelength, ray.bins)
# The cache contains the emission converted from W/(m^3 str) to W/(m^3 str nm).
# Let's check this.
samples = np.zeros(ray.bins)
material.add_emission_to_array(samples, 1, 1, 1, 1.0)
ivoxel = material.voxel_index(1, 1, 1)
print('Emission at %g nm: %g W/(m^3 str)' % (wavelengths[ivoxel], emission[1, 1, 1, ivoxel]))
# All elements of 'samples' array except one are zeros.
bin_index = np.argmax(samples)
samples_delta = (ray.max_wavelength - ray.min_wavelength) / ray.bins
bin_min_wl = ray.min_wavelength + bin_index * samples_delta
print('Emission in %g nm - %g nm: %g W/(m^3 str nm)' % (bin_min_wl, bin_min_wl + samples_delta, samples[bin_index]))
print('Spectral step: %g nm' % samples_delta)
print()
# Like in the case of continuous spectrum, sampling conserves the wavelength integral.


""" Part 3: Memory saving. """
print('Part 3: Memory saving.\n')


# Working with large 3D spatial grids and high spectral resolution consumes a lot of memory.
# RegularGridEmitter stores both the original emission profile and the cache in sparse matrices.
# If the emission lower than a certain threshold can be neglected, explicitly zeroing the values lower
# than this threshold before passing the emission to RegularGridEmitter, allow to save some memory.

# If the emission spectrum has some regions where the material does not emit, the emission should be
# explicitly zeroed at the borders of these regions. Otherwise, the spectrum will be interpolated
# between these regions during the sampling and the cache will be filled with negligable in terms of
# physics but non-zero values. This will not only increase memory consumption but also the computing time.

# As shown above, the emission can be passed to RegularGridEmitter as a scipy csc_matrix. This saves memory.
# Building the cache manually before camera.observe() call when possible also saves memory in case of
# multi-process rendering.


# It is possible to store the cache in float32 instead of float64 with a 'cache_32bit' option. Do this, if
# memory consumption is too large for your machine.
# Setting this option when the cache is already built in 64-bit, deletes the current cache.
print('Is cache 32-bit? %s' % material.cache_32bit)
print('Is cache empty? %s' % material.cache_empty())
material.cache_32bit = True
print('Is cache empty now? %s' % material.cache_empty())
material.cache_build(ray.min_wavelength, ray.max_wavelength, ray.bins)
print('Is cache 32-bit now? %s' % material.cache_32bit)
print()
# Note that all computations are perfromed with double precision, so converting from float32 to
# float64 on the fly will consume some time.


# If none of the above solutions work and memory consumption is still too large, there is an option to
# initialise RegularGridEmitter with a dummy emission array and then provide the pre-calculated cache.
# Of course, this solution will work only if ray spectral properties do not change during the rendering.
wavelengths, delta = np.linspace(ray.min_wavelength, ray.max_wavelength, ray.bins + 1, retstep=True)
wavelengths = wavelengths[:-1] + 0.5 * delta
xwl = (wavelengths - ray.min_wavelength) / (ray.max_wavelength - ray.min_wavelength) * 3 * np.pi
cos2_law = np.cos(xwl)**2
sin2_law = np.sin(xwl)**2
emission = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], ray.bins))
emission[::2, ::2, ::2, :] = cos2_law[None, None, None, :]  # W / (m^3 str nm)
emission[1, 1, 1, :] = 2. * sin2_law  # W / (m^3 str nm)
# Cache is stored as a csr_matrix and must be provided in that form.
cache = csr_matrix(emission.reshape(nvoxel, ray.bins), dtype=np.float32)
material = RegularGridEmitter(grid_shape, grid_steps, csr_matrix((nvoxel, 2)),
                              np.array([ray.min_wavelength, ray.max_wavelength]))  # initialising with a dummy emission array
material.cache_override(cache, ray.min_wavelength, ray.max_wavelength)  # overriding the cache
print('Is cache valid? %s' % material.cache_valid(ray.min_wavelength, ray.max_wavelength, ray.bins))
print('Is cache 32-bit? %s' % material.cache_32bit)  # we provided the cache in float32
print()

plt.show()
