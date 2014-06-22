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

from time import time
from numpy import array, zeros
from math import sin, cos, tan, atan, pi
from matplotlib.pyplot import imshow, imsave, show, ion, ioff, clf, figure, draw
from raysect.core import World, Observer, AffineMatrix, Point, Vector
from raysect.optical.ray import Ray
from raysect.optical.colour import resample_ciexyz, spectrum_to_ciexyz, ciexyz_to_srgb
from raysect.optical import Spectrum

class PinholeCamera(Observer):

    def __init__(self, parent = None, transform = AffineMatrix(), pixels = (640, 480), fov = 40, spectral_samples = 20, dispersion = False, name = ""):

        super().__init__(parent, transform, name)

        self.pixels = pixels
        self.fov = fov
        self.frame = None

        self.spectral_samples = spectral_samples
        self.dispersion = dispersion

        self.display_progress = True
        self.display_update_time = 5.0

    @property
    def pixels(self):

        return self._pixels

    @pixels.setter
    def pixels(self, pixels):

        if len(pixels) != 2:

            raise ValueError("Pixel dimensions of camera framebuffer must be a tuple containing the x and y pixel counts.")

        self._pixels = pixels

    @property
    def fov(self):

        return self._fov

    @fov.setter
    def fov(self, fov):

        if fov <= 0:

            raise ValueError("Field of view angle can not be less than or equal to 0 degrees.")

        self._fov = fov

    def observe(self):

        self.frame = zeros((self._pixels[1], self._pixels[0], 3))

        if isinstance(self.root, World) == False:

            raise TypeError("Observer is not conected to a scenegraph containing a World object.")

        world = self.root

        max_pixels = max(self._pixels)

        if max_pixels > 0:

            # max width of image plane at 1 meter
            image_max_width = 2 * tan(pi / 180 * 0.5 * self._fov)

            # pixel step and start point in image plane
            image_delta = image_max_width / (max_pixels - 1)

            # start point of scan in image plane
            image_start_x = 0.5 * self._pixels[0] * image_delta
            image_start_y = 0.5 * self._pixels[1] * image_delta

        else:

            # single ray on axis
            image_delta = 0
            image_start_x = 0
            image_start_y = 0

        min_wavelength = 375
        max_wavelength = 785

        resampled_xyz = resample_ciexyz(min_wavelength,
                                        max_wavelength,
                                        self.spectral_samples)

        if self.dispersion:

            # seperate wavelength for each ray
            rays = list()
            delta_wavelength = (max_wavelength - min_wavelength) / self.spectral_samples
            lower_wavelength = min_wavelength
            for index in range(self.spectral_samples):

                upper_wavelength = min_wavelength + delta_wavelength * (index + 1)

                rays.append(Ray(min_wavelength = lower_wavelength,
                                max_wavelength = upper_wavelength,
                                samples = 1))

                print(lower_wavelength, upper_wavelength)

                lower_wavelength = upper_wavelength

        else:

            # single ray for all wavelengths
            ray = Ray(min_wavelength = min_wavelength,
                      max_wavelength = max_wavelength,
                      samples = self.spectral_samples)

        if self.display_progress:

            self.display()
            display_timer = time()

        for y in range(0, self._pixels[1]):

            if self.display_progress:

                print("line " + str(y) + "/" + str(self._pixels[1]))

            for x in range(0, self._pixels[0]):

                # ray angles
                theta = atan(image_start_x - image_delta * x)
                phi = atan(image_start_y - image_delta * y)

                # calculate ray parameters
                origin = Point([0, 0, 0])
                direction = Vector([sin(theta),
                                    cos(theta) * sin(phi),
                                    cos(theta) * cos(phi)])

                # convert to world space
                origin = origin.transform(self.to_root())
                direction = direction.transform(self.to_root())

                if self.dispersion:

                    spectrum = Spectrum(min_wavelength, max_wavelength, self.spectral_samples)

                    for index, ray in enumerate(rays):

                        ray.origin = origin
                        ray.direction = direction

                        sample = ray.trace(world)
                        spectrum.bins[index] = sample.bins[0]

                    xyz = spectrum_to_ciexyz(spectrum, resampled_xyz)
                    rgb = ciexyz_to_srgb(xyz[0], xyz[1], xyz[2])
                    self.frame[y, x, 0] = rgb[0]
                    self.frame[y, x, 1] = rgb[1]
                    self.frame[y, x, 2] = rgb[2]

                else:

                    ray.origin = origin
                    ray.direction = direction

                    # trace and accumulate
                    spectrum = ray.trace(world)
                    xyz = spectrum_to_ciexyz(spectrum, resampled_xyz)
                    rgb = ciexyz_to_srgb(xyz[0], xyz[1], xyz[2])
                    self.frame[y, x, 0] = rgb[0]
                    self.frame[y, x, 1] = rgb[1]
                    self.frame[y, x, 2] = rgb[2]

                if self.display_progress and (time() - display_timer) > self.display_update_time:

                    self.display()
                    display_timer = time()

        if self.display_progress:

            self.display()

    def display(self):

        clf()
        imshow(self.frame, aspect = "equal", origin = "upper")
        draw()

    def save(self, filename):

        imsave(filename, self.frame)
