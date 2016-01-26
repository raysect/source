# cython: language_level=3

# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import matplotlib.pyplot as plt
from numpy import pi as PI

from raysect.optical.observer.point_generator import SinglePoint, Disk
from raysect.optical.observer.vector_generators import SingleRay, ConeUniform
from raysect.optical.ray import Ray
from raysect.optical import Spectrum
from raysect.core import World, AffineMatrix3D, Point2D, Observer, Point3D, Vector3D


class LOSObserver(Observer):
    """
    The abstract base class for Line Of Sight (LOS) observers.

    These observers are different from since they observe from a single origin point and the concept of pixels doesn't
    make sense. Derived observers could include LineOfSight, OpticalFibre, etc. Point observers can only return a
    spectrum, not an image.

    :param float sensitivity: The observers sensitivity coefficient, all samples collected by this LOS will be
    multiplied by this number.
    :param int spectral_samples: The number of wavelength bins to collect over the wavelength range min_wavelength to
    max_wavelength. For example, if the wavelength range is 400nm to 500nm and spectral_samples=100, each wavelength bin
    would be 1nm wide. Default is spectral_samples=20.
    :param int spectral_rays: The number of rays to sample over this wavelength range. For example, if the wavelength
    range is 400nm to 500nm, spectral_samples=100 and spectral_rays=5, their would be five rays launched each with 20
    spectral samples. The first ray would have the range 400nm-420nm, 420nm-440nm for the second ray, etc. This
    behaviour is needed when dispersion effects are important. For example, when light passes through a prism and is
    separated into different paths. For scenes where dispersion effects are important, rays >> 10. The default
    spectral_rays = 1.
    :param int los_samples: The number of rays to launch when sampling this LOS. Real life observers like fibre optics
    collect light over a solid angle. To prevent aliasing effects and compute a more realistic response,
    los_samples >> 10. The default los_samples=100.
    :param parent: The parent node in the scenegraph. All camera observers must be parented to a World object.
    :param AffineMatrix3D transform: A transform matrix describing the location and orientation of this camera in world
    space.
    :param name: An optional name for this camera.
    """

    def __init__(self, sensitivity=1.0, spectral_samples=512, spectral_rays=1, los_samples=100, parent=None,
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
        self.los_samples = los_samples
        self.pixel = None

        # accumulation settings
        self.accumulate = False
        self.accumulated_samples = 0

        # Output from last call to Observe()
        self.spectrum = None
        self.max_radiance = 0.

    def observe(self):
        """ Ask this observer to Observe its world. """

        if not isinstance(self.root, World):
            raise TypeError("Observer is not connected to a scene graph containing a World object.")

        if self.min_wavelength >= self.max_wavelength:
            raise RuntimeError("Min wavelength is superior to max wavelength!")

        if not self.accumulate:
            self.spectrum = Spectrum(self.min_wavelength, self.max_wavelength, self.spectral_samples)
            self.accumulated_samples = 0

        # generate spectral data
        ray_templates = self._generate_ray_templates()

        # Loop over spectral samples and trace rays
        lower_index = 0
        for i_channel, ray_template in enumerate(ray_templates):

            # upper_index = self.spectral_samples * (i_channel + 1)
            upper_index = lower_index + ray_template.num_samples

            # trace rays on this pixel
            spectrum, ray_count = self._sample_los(self.root, ray_template)

            self.spectrum.samples[lower_index:upper_index] += spectrum.samples

            lower_index = upper_index

    def _sample_los(self, world, ray_template):

        # generate rays
        rays = self._generate_rays(ray_template)

        # create spectrum and calculate sample weighting
        spectrum = ray_template.new_spectrum()
        weight = 1 / self.los_samples

        # initialise ray statistics
        ray_count = 0

        # launch rays and accumulate spectral samples
        for ray in rays:

            print("tracing ray {}".format(ray_count))

            # convert ray from local space to world space
            ray.origin = ray.origin.transform(self.to_root())
            ray.direction = ray.direction.transform(self.to_root())

            sample = ray.trace(world)
            spectrum.samples += weight * sample.samples

            # accumulate statistics
            ray_count += ray.ray_count

        # apply camera sensitivity
        spectrum.samples *= self.sensitivity

        return spectrum, ray_count

    def _generate_rays(self, ray_template):
        """
        Virtual method - to be implemented by derived classes.

        Runs during the observe() loop to generate the rays. Allows observers to customise how they launch rays.

        :return list Rays: a list of ray objects for the observer to sample the scene with.
        """
        raise NotImplementedError("Virtual method _generate_rays() has not been implemented for this point observer.")

    def _generate_ray_templates(self):

        # split spectral bins across rays - non-integer division is handled by
        # rounding up or down the non-integer boundaries between the ray ranges,
        # this means that some rays will have more samples than others
        current = 0
        start = 0
        ranges = []
        while start < self.spectral_samples:
            current += self.spectral_samples / self.spectral_rays
            end = round(current)
            ranges.append((start, end))
            start = end

        # build template rays
        rays = []
        delta_wavelength = (self.max_wavelength - self.min_wavelength) / self.spectral_samples
        for start, end in ranges:
            rays.append(
                Ray(min_wavelength=self.min_wavelength + delta_wavelength * start,
                    max_wavelength=self.min_wavelength + delta_wavelength * end,
                    num_samples=end - start,
                    extinction_prob=self.ray_extinction_prob,
                    min_depth=self.ray_min_depth,
                    max_depth=self.ray_max_depth)
            )

        return rays

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


class BasicLineOfSight(LOSObserver):
    """
    The most basic Line Of Sight observer in Raysect.

    Inherits arguments and attributes from the base LOSObserver class. Fires a single ray oriented along the observer's
    z axis in world space.
    """
    def _generate_rays(self, ray_template):

        rays = []
        for n in range(self.los_samples):
            rays.append(ray_template.copy(Point3D(), Vector3D()))
        return rays


class OpticalFibre(LOSObserver):
    """
    An optical fibre Line Of Sight observer.

    Inherits arguments and attributes from the base LOSObserver class. Rays are sampled over a circular area at the fibre tip
    and a conical solid angle defined by the acceptance_angle parameter.

    :param float acceptance_angle: The angle in radians between the z axis and the cone surface which defines the fibres
    soild angle sampling area.
    :param float radius: The radius of the fibre tip in metres. This radius defines a circular area at the fibre tip
    which will be sampled over.
    """
    def __init__(self, acceptance_angle=PI/16, radius=0.001, sensitivity=1.0, spectral_samples=512,
                 spectral_rays=1, los_samples=1, parent=None, transform=AffineMatrix3D(), name=""):

        super().__init__(sensitivity=sensitivity, spectral_samples=spectral_samples, spectral_rays=spectral_rays,
                         los_samples=los_samples, parent=parent, transform=transform, name=name)

        if not 0 <= acceptance_angle <= PI/2:
            raise RuntimeError("Acceptance angle {} for OpticalFibre must be between 0 and pi/2."
                               "".format(acceptance_angle))

        self.acceptance_angle = acceptance_angle
        self.radius = radius

    def _generate_rays(self, ray_template):

        point_generator = Disk(self.radius)
        origins = point_generator(self.los_samples)
        vector_generator = ConeUniform(self.acceptance_angle)
        directions = vector_generator(self.los_samples)

        rays = []
        for n in range(self.los_samples):
            rays.append(ray_template.copy(origins[n], directions[n]))
        return rays
