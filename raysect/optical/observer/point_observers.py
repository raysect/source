
from multiprocessing import Process, cpu_count, Queue

import matplotlib.pyplot as plt
from numpy import pi as PI

from raysect.optical.observer.pixel import VectorSamplerPixel
from raysect.optical.ray import Ray
from raysect.optical import Spectrum
from raysect.core import World, AffineMatrix3D, Point2D, Observer
from raysect.optical.observer.point_generator import SinglePointGenerator, CircularPointGenerator
from raysect.optical.observer.vector_generators import SingleRay, Cone


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

    def __init__(self, sensitivity=1.0, spectral_samples=512, spectral_rays=1, pixel_samples=1, parent=None,
                 transform=AffineMatrix3D(), name=""):

        super().__init__(parent, transform, name)

        # ray configuration
        self.spectral_rays = spectral_rays
        self.spectral_samples = spectral_samples
        self.min_wavelength = 375.0
        self.max_wavelength = 740.0
        self.ray_extinction_prob = 0.1
        self.ray_min_depth = 3
        self.ray_max_depth = 15

        self.sensitivity = sensitivity
        self.pixel_samples = pixel_samples
        self.pixel = None

        # accumulation settings
        self.accumulate = False
        self.accumulated_samples = 0

        # Output from last call to Observe()
        self.spectrum = None
        self.max_radiance = 0.

    def configure_pixel(self):
        """
        Virtual method - to be implemented by derived classes.

        Runs at the start of observe() loop to set up any data needed for calculating pixel vectors
        and super-sampling that shouldn't be calculated at every loop iteration. The result of this
        function should be written to self._pixel_vectors_variables.
        """
        raise NotImplementedError("Virtual method configure_pixel() has not been implemented for this Camera.")

    def observe(self):
        """ Fire a single ray and fill the 'spectrum' attribute with the
        observed spectrum. If 'display' attribute is True, the spectrum is
        shown at the end.
        """

        if not isinstance(self.root, World):
            raise TypeError("Observer is not connected to a scene graph containing a World object.")

        if self.min_wavelength >= self.max_wavelength:
            raise RuntimeError("Min wavelength is superior to max wavelength!")

        total_samples = self.spectral_rays * self.spectral_samples

        if not self.accumulate:
            self.spectrum = Spectrum(self.min_wavelength, self.max_wavelength, total_samples)
            self.accumulated_samples = 0

        # generate spectral data
        wvl_channels = self._calc_wvl_channel_config()

        # rebuild pixels in case camera properties have changed
        self.configure_pixel()

        # Loop over spectral samples and trace rays
        lower_index = 0
        for i_channel, wvl_channel_config in enumerate(wvl_channels):

            upper_index = self.spectral_samples * (i_channel + 1)

            min_wavelength, max_wavelength, spectral_samples = wvl_channel_config

            # trace rays on this pixel
            spectrum, ray_count = self.pixel.sample_pixel(min_wavelength, max_wavelength, spectral_samples, self)

            self.spectrum.samples[lower_index:upper_index] = spectrum.samples

            lower_index = upper_index

    def _calc_wvl_channel_config(self):
        """
        Break the wavelength range up based on the number of required spectral rays. When simulated dispersion effects
        or reflections for example, the overall wavelength range may be broken up into >20 sub regions for individual
        ray sampling.

        :return: list[tuples (min_wavelength, max_wavelength, spectral_samples),...]
        """

        # TODO - spectral_samples needs to be over whole wavelength range, not each sub wavelength range.
        config = []
        delta_wavelength = (self.max_wavelength - self.min_wavelength) / self.spectral_rays
        for index in range(self.spectral_rays):
            config.append((self.min_wavelength + delta_wavelength * index,
                           self.min_wavelength + delta_wavelength * (index + 1),
                           self.spectral_samples))
        return config

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

    def configure_pixel(self):
        self.pixel = VectorSamplerPixel((0, 0), Point2D(0.0, 0.0), AffineMatrix3D(), self.to_root(),
                                        SinglePointGenerator(), SingleRay())


class OpticalFibre(PointObserver):
    def __init__(self, acceptance_angle=PI/16, radius=0.001, sensitivity=1.0, spectral_samples=512,
                 spectral_rays=1, pixel_samples=1, parent=None, transform=AffineMatrix3D(), name=""):

        super().__init__(sensitivity=sensitivity, spectral_samples=spectral_samples, spectral_rays=spectral_rays,
                         pixel_samples=pixel_samples, parent=parent, transform=transform, name=name)

        if not 0 <= acceptance_angle <= PI/2:
            raise RuntimeError("Acceptance angle {} for OpticalFibre must be between 0 and pi/2."
                               "".format(acceptance_angle))

        self.acceptance_angle = acceptance_angle
        self.radius = radius

    def configure_pixel(self):
        self.pixel = VectorSamplerPixel((0, 0), Point2D(0.0, 0.0), AffineMatrix3D(), self.to_root(),
                                        CircularPointGenerator(self.radius), Cone(self.acceptance_angle))
