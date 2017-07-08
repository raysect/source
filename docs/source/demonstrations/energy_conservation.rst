
.. _demo_energy_conservation:

Checking Energy Conservation
============================

Its important that ray-tracers conserve energy. This demonstration gives an example case where we prove
that the total power collected on the boundary equals to total power radiated from a simple emitting source.
The principle can be extended to test cases with more complicated emission sources and boundaries.

As usual, we start by importing everything we need. ::

   from math import pi
   from raysect.core import translate, rotate
   from raysect.primitive import Sphere
   from raysect.optical import World, ConstantSF
   from raysect.optical.observer import Pixel, PowerPipeline0D
   from raysect.optical.material.emitter import UnityVolumeEmitter, UniformSurfaceEmitter

The emitting source will be a sphere at the origin, surrounded by an observing box. The number of samples
requested will be relatively large so we can ensure good accuracy in the calculations. Note that this calculation
isn't spectrally resolved so we only need a narrow spectral range with a single wavelength bin. Initialise the
constants and place the sphere in the scene. ::

   samples = 100000

   sphere_radius = 0.5
   cube_size = 2

   min_wl = 400
   max_wl = 401

   # set-up scenegraph
   world = World()
   emitter = Sphere(radius=sphere_radius, parent=world)

Note that due to symmetry we don't need to observe each face of the cube. We only need to observe 1 side and
multiply by 6. You could also model it with 6 individual pixel faces. ::

   power = PowerPipeline0D(accumulate=False)
   observing_plane = Pixel([power], x_width=cube_size, y_width=cube_size,
                           min_wavelength=min_wl, max_wavelength=max_wl,
                           spectral_bins=1, pixel_samples=samples,
                           parent=world, transform=rotate(0, 0, 0)*translate(0, 0, -cube_size / 2))


First, let's consider the case where the sphere is a uniform volume emitter with emission of 1W/m^3/str/nm).
The theoretical total emitted power would be given by 1W x the volume of the sphere x 4Pi steradians x
the wavelength range sampled. Lets implement the calculation and compare it to the measured value. ::

   print("Starting observations with volume emitter...")
   calculated_volume_emission = 16 / 3 * pi**2 * sphere_radius**3 * (max_wl - min_wl)

   emitter.material = UnityVolumeEmitter()
   observing_plane.observe()
   measured_volume_emission = 6 * power.value.mean
   measured_volume_error = 6 * power.value.error()

   print('Expected volume emission => {} W'.format(calculated_volume_emission))
   print('Measured volume emission => {} +/- {} W'.format(measured_volume_emission, measured_volume_error))

For our test case, expected volume emission => 6.57 W, measured volume emission => 6.623 +/- 0.057 W.


