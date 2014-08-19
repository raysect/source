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

    def __init__(self, pixels = (640, 480), fov = 40, spectral_samples = 20, rays = 1, parent = None, transform = AffineMatrix(), name = ""):

        super().__init__(parent, transform, name)

        self.pixels = pixels
        self.fov = fov
        self.frame = None
        # self.subsampling = 1

        self.rays = rays
        self.spectral_samples = spectral_samples

        self.min_wavelength = 375.0
        self.max_wavelength = 785.0

        self.ray_max_depth = 15

        self.display_progress = True
        self.display_update_time = 10.0


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

            raise TypeError("Observer is not connected to a scene graph containing a World object.")

        world = self.root

        max_pixels = max(self._pixels)

        if max_pixels > 1:

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

        total_samples = self.rays * self.spectral_samples

        resampled_xyz = resample_ciexyz(self.min_wavelength,
                                        self.max_wavelength,
                                        total_samples)

        # generate rays
        rays = list()
        delta_wavelength = (self.max_wavelength - self.min_wavelength) / self.rays
        lower_wavelength = self.min_wavelength
        for index in range(self.rays):

            upper_wavelength = self.min_wavelength + delta_wavelength * (index + 1)

            rays.append(Ray(min_wavelength=lower_wavelength,
                            max_wavelength=upper_wavelength,
                            samples=self.spectral_samples,
                            max_depth=self.ray_max_depth))

            lower_wavelength = upper_wavelength

        total_pixels = self._pixels[0] * self._pixels[1]
        progress_timer = time()

        display_timer = 0
        if self.display_progress:

            self.display()
            display_timer = time()

        for y in range(0, self._pixels[1]):

            for x in range(0, self._pixels[0]):

                if (time() - progress_timer) > 1.0:

                    current_pixel = y * self._pixels[0] + x
                    completion = 100 * current_pixel / total_pixels
                    print("{}% complete (line {}/{}, pixel {}/{})".format(completion, y, self._pixels[1], current_pixel, total_pixels))
                    progress_timer = time()

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

                # sample world
                spectrum = Spectrum(self.min_wavelength, self.max_wavelength, total_samples)
                lower_index = 0
                for index, ray in enumerate(rays):

                    upper_index = self.spectral_samples * (index + 1)

                    ray.origin = origin
                    ray.direction = direction

                    sample = ray.trace(world)
                    spectrum.bins[lower_index:upper_index] = sample.bins

                    lower_index = upper_index

                # convert spectrum to sRGB
                xyz = spectrum_to_ciexyz(spectrum, resampled_xyz)
                rgb = ciexyz_to_srgb(xyz)
                self.frame[y, x, 0] = rgb[0]
                self.frame[y, x, 1] = rgb[1]
                self.frame[y, x, 2] = rgb[2]

                if self.display_progress and (time() - display_timer) > self.display_update_time:

                    print("Refreshing display...")
                    self.display()
                    display_timer = time()

        print("100% complete (line {}/{}, pixel {}/{})".format(self._pixels[1], self._pixels[1], total_pixels, total_pixels))

        if self.display_progress:

            self.display()

    def display(self):

        clf()
        imshow(self.frame, aspect = "equal", origin = "upper")
        draw()
        show()

    def save(self, filename):

        imsave(filename, self.frame)
