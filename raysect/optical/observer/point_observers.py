
from multiprocessing import Process, cpu_count, Queue

import matplotlib.pyplot as plt

from raysect.optical.ray import Ray
from raysect.optical import Spectrum
from raysect.core import World, AffineMatrix, Point, Vector, Observer


class PointObserver(Observer):
    """
    Base class for observers that observe from a single origin point. Derived observers could include LineOfSight,
    OpticalFibre, etc. Point observers can only return a spectrum, not an image. In future, the ray sampling class will
    will be specified by the user, i.e. single ray along viewing axis, or volumetric sampling of a light cone, etc.


    :param spectral_samples: number of spectral samples by ray
    :param rays: number of rays, each ray will be given a part of the spectrum
    :param parent: scenegraph parent
    :param transform: relative position to the parent
    :param name: a printable name
    """

    def __init__(self, process_count=cpu_count(), sensitivity=1.0, spectral_samples=20, rays=1, parent=None,
                 transform=AffineMatrix(), name=""):

        Observer.__init__(self, parent, transform, name)

        # ray configuration
        self.rays = rays
        self.spectral_samples = spectral_samples
        self.min_wavelength = 375.0
        self.max_wavelength = 740.0
        self.ray_extinction_prob = 0.1
        self.ray_min_depth = 3
        self.ray_max_depth = 15

        # concurrency configuration
        self.process_count = process_count

        self.sensitivity = sensitivity

        # accumulation settings
        self.accumulate = False
        self.accumulated_samples = 0

        # Output from last call to Observe()
        self.spectrum = None
        self.max_radiance = 0.

    def observe(self):
        """ Fire a single ray and fill the 'spectrum' attribute with the
        observed spectrum. If 'display' attribute is True, the spectrum is
        shown at the end.
        """

        if not isinstance(self.root, World):
            raise TypeError("Observer is not connected to a scene graph containing a World object.")

        if self.min_wavelength >= self.max_wavelength:
            raise RuntimeError("Min wavelength is superior to max wavelength!")

        world = self.root

        total_samples = self.rays * self.spectral_samples

        rays = self._generate_rays()

        if not self.accumulate:
            self.spectrum = Spectrum(self.min_wavelength, self.max_wavelength, total_samples)
            self.accumulated_samples = 0

        # trace
        if self.process_count == 1:
            self._observe_single(world, rays)
        else:
            self._observe_parallel(world, rays)

    def _observe_single(self, world, rays):

        lower_index = 0
        for index, ray in enumerate(rays):

            upper_index = self.spectral_samples * (index + 1)

            # convert ray parameters to world space
            ray.origin = Point(0, 0, 0).transform(self.to_root())
            ray.direction = Vector(0., 0., 1.0).transform(self.to_root())

            # sample world
            sample = ray.trace(world)
            self.spectrum.samples[lower_index:upper_index] = sample.samples

            lower_index = upper_index

    def _observe_parallel(self, world, rays):
        pass

    def _generate_rays(self):
        """
        Virtual method - to be implemented by derived classes.

        Called for each pixel in the _worker() observe loop. For a given pixel, this function must return a list of
        vectors to ray trace.
        """
        raise NotImplementedError("Virtual method _generate_rays() has not been implemented for this Observer.")

    def display(self):
        """
        Plot the observed spectrum.
        """

        if self.spectrum is None:
            raise RuntimeError("No spectrum on this observer. Must call observe() before a spectrum can be displayed.")

        plt.plot(self.spectrum.wavelengths, self.spectrum.samples)
        plt.xlabel('wavelength (nm)')
        plt.ylabel('radiance (J/s/m^2/str/nm)')
        plt.show()


class LineOfSight(PointObserver):

    def _generate_rays(self):

        rays = list()
        delta_wavelength = (self.max_wavelength - self.min_wavelength) / self.rays
        lower_wavelength = self.min_wavelength
        for index in range(self.rays):

            upper_wavelength = self.min_wavelength + delta_wavelength * (index + 1)

            rays.append(Ray(min_wavelength=lower_wavelength, max_wavelength=upper_wavelength,
                            num_samples=self.spectral_samples, max_depth=self.ray_max_depth))

            lower_wavelength = upper_wavelength

        return rays


# TODO - implement optical fibre
class OpticalFibre(PointObserver):
    pass

