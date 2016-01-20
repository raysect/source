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


# TODO: add sensitivity to pixel.sample calls

from math import tan, pi, ceil
from multiprocessing import Process, cpu_count, Queue
from time import time

import numpy as np
from matplotlib.pyplot import imshow, imsave, show, clf, draw, pause
from numpy import zeros

from raysect.core import World, AffineMatrix3D, Point3D, Point2D, Vector3D, Observer, translate
from raysect.core.math import random
from raysect.optical.colour import resample_ciexyz, spectrum_to_ciexyz, ciexyz_to_srgb
from raysect.optical.observer.point_generator import Rectangle
from raysect.optical.observer.vector_generators import SingleRay
from raysect.optical import Ray


# todo: add scheme to evenly subdivide spectral_samples across specified spectral_rays
class Camera(Observer):

    def __init__(self, pixels=(512, 512), sensitivity=1.0, spectral_samples=20, spectral_rays=1, pixel_samples=100,
                 process_count=cpu_count(), parent=None, transform=AffineMatrix3D(), name=None):
        """

        :param pixels:
        :param sensitivity:
        :param spectral_samples: number of wavelength bins between min/max wavelengths.
        :param spectral_rays: number of rays to use when sub-sampling the overall spectral range. This must be ~15 or
        greater to see dispersion effects in materials.
        :param pixel_samples:
        :param process_count:
        :param parent:
        :param transform:
        :param name:
        :return:
        """

        super().__init__(parent, transform, name)

        # ray configuration
        self.spectral_rays = spectral_rays
        self.spectral_samples = spectral_samples
        self.min_wavelength = 375.0
        self.max_wavelength = 740.0
        self.ray_extinction_prob = 0.1
        self.ray_min_depth = 3
        self.ray_max_depth = 100

        # progress information
        self.display_progress = True
        self.display_update_time = 10.0

        # concurrency configuration
        self.process_count = process_count

        # camera configuration
        self._pixels = pixels
        self.sensitivity = sensitivity
        self.pixel_samples = pixel_samples
        self.accumulate = False
        self.accumulated_samples = 0

        # output from last call to Observe()
        self.xyz_frame = zeros((pixels[1], pixels[0], 3))
        self.frame = zeros((pixels[1], pixels[0], 3))

        self._point_generator = None
        self._vector_generator = None

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):

        # TODO: add validation
        self._pixels = pixels

    def observe(self):

        # must be connected to a world node to be able to perform a ray trace
        if not isinstance(self.root, World):
            raise TypeError("Observer is not connected to a scene graph containing a World object.")

        if self.min_wavelength >= self.max_wavelength:
            raise RuntimeError("Min wavelength is superior to max wavelength!")

        # create intermediate and final frame-buffers
        if not self.accumulate:
            self.xyz_frame = zeros((self._pixels[0], self._pixels[1], 3))
            self.frame = zeros((self._pixels[0], self._pixels[1], 3))
            self.accumulated_samples = 0

        # generate spectral data
        ray_templates = self._generate_ray_templates()

        # trace
        if self.process_count == 1:
            self._observe_single(self.xyz_frame, ray_templates)
        else:
            self._observe_parallel(self.xyz_frame, ray_templates)

        # update sample accumulation statistics
        self.accumulated_samples += self.pixel_samples * self.spectral_rays

    def _observe_single(self, xyz_frame, ray_templates):

        # initialise user interface
        display_timer = self._start_display()
        statistics_data = self._start_statistics()

        # generate weightings for accumulation
        total_samples = self.accumulated_samples + self.pixel_samples * self.spectral_rays
        previous_weight = self.accumulated_samples / total_samples
        added_weight = self.spectral_rays * self.pixel_samples / total_samples

        # scale previous state to account for additional samples
        xyz_frame[:, :, :] = previous_weight * xyz_frame[:, :, :]

        # render
        for i_channel, ray_template in enumerate(ray_templates):

            # generate resampled XYZ curves for channel spectral range
            resampled_xyz = resample_ciexyz(
                ray_template.min_wavelength,
                ray_template.max_wavelength,
                ray_template.num_samples
            )

            ny, nx = self._pixels
            for iy in range(ny):
                for ix in range(nx):

                    # generate pixel transform
                    transform = self._generate_pixel_transform(ix, iy)

                    # load selected pixel and trace rays
                    spectrum, ray_count = self._sample_pixel(self.pixel_samples, self.root, ray_template, self.to_root() * transform)

                    # convert spectrum to CIE XYZ
                    xyz = spectrum_to_ciexyz(spectrum, resampled_xyz)

                    xyz_frame[iy, ix, 0] += added_weight * xyz[0]
                    xyz_frame[iy, ix, 1] += added_weight * xyz[1]
                    xyz_frame[iy, ix, 2] += added_weight * xyz[2]

                    # convert to sRGB colour-space
                    self.frame[iy, ix, :] = ciexyz_to_srgb(*xyz_frame[iy, ix, :])

                    # update users
                    pixel = ix + self._pixels[0] * iy
                    statistics_data = self._update_statistics(statistics_data, i_channel, pixel, ray_count)
                    display_timer = self._update_display(display_timer)

        # final update for users
        self._final_statistics(statistics_data)
        self._final_display()

    def _observe_parallel(self, xyz_frame, ray_templates):

        world = self.root

        # initialise user interface
        display_timer = self._start_display()
        statistics_data = self._start_statistics()

        total_pixels = self._pixels[0] * self._pixels[1]

        # generate weightings for accumulation
        total_samples = self.accumulated_samples + self.pixel_samples * self.spectral_rays
        previous_weight = self.accumulated_samples / total_samples
        added_weight = self.spectral_rays * self.pixel_samples / total_samples

        # scale previous state to account for additional samples
        xyz_frame[:, :, :] = previous_weight * xyz_frame[:, :, :]

        # render
        for channel, ray_template in enumerate(ray_templates):

            # generate resampled XYZ curves for channel spectral range
            resampled_xyz = resample_ciexyz(
                ray_template.min_wavelength,
                ray_template.max_wavelength,
                ray_template.num_samples
            )

            # establish ipc queues using a manager process
            task_queue = Queue()
            result_queue = Queue()

            # start process to generate image samples
            producer = Process(target=self._producer, args=(task_queue, ))
            producer.start()

            # start worker processes
            workers = []
            for pid in range(self.process_count):
                p = Process(target=self._worker,
                            args=(world, ray_template,
                                  resampled_xyz, task_queue, result_queue))
                p.start()
                workers.append(p)

            # collect results
            for pixel in range(total_pixels):

                # obtain result
                location, xyz, sample_ray_count = result_queue.get()
                x, y = location

                xyz_frame[y, x, 0] += added_weight * xyz[0]
                xyz_frame[y, x, 1] += added_weight * xyz[1]
                xyz_frame[y, x, 2] += added_weight * xyz[2]

                # convert to sRGB colour-space
                self.frame[y, x, :] = ciexyz_to_srgb(*xyz_frame[y, x, :])

                # update users
                display_timer = self._update_display(display_timer)
                statistics_data = self._update_statistics(statistics_data, channel, pixel, sample_ray_count)

            # shutdown workers
            for _ in workers:
                task_queue.put(None)

        # final update for users
        self._final_statistics(statistics_data)
        self._final_display()

    def _producer(self, task_queue):
        # task is simply the pixel location
        nx, ny = self._pixels
        for y in range(ny):
            for x in range(nx):
                task_queue.put((x, y))

    def _worker(self, world, ray_template, resampled_xyz, task_queue, result_queue):

        # re-seed the random number generator to prevent all workers inheriting the same sequence
        random.seed()

        while True:

            # request next pixel
            pixel_id = task_queue.get()

            # have we been commanded to shutdown?
            if pixel_id is None:
                break

            ix, iy = pixel_id

            # generate pixel transform
            transform = self._generate_pixel_transform(ix, iy)

            # load selected pixel and trace rays
            spectrum, ray_count = self._sample_pixel(self.pixel_samples, self.root, ray_template, self.to_root() * transform)

            # convert spectrum to CIE XYZ
            xyz = spectrum_to_ciexyz(spectrum, resampled_xyz)

            # encode result and send
            result = (pixel_id, xyz, ray_count)
            result_queue.put(result)

    def _generate_ray_templates(self):

        rays = []
        delta_wavelength = (self.max_wavelength - self.min_wavelength) / self.spectral_rays
        for index in range(self.spectral_rays):
            rays.append(
                Ray(min_wavelength=self.min_wavelength + delta_wavelength * index,
                    max_wavelength=self.min_wavelength + delta_wavelength * (index + 1),
                    num_samples=self.spectral_samples,
                    extinction_prob=self.ray_extinction_prob,
                    min_depth=self.ray_min_depth,
                    max_depth=self.ray_max_depth)
            )
        return rays

    def _generate_pixel_transform(self, x, y):
        raise NotImplementedError("To be defined in subclass.")

    def _sample_pixel(self, samples, world, ray_template, transform=None):

        # create spectrum and calculate sample weighting
        spectrum = ray_template.new_spectrum()
        weight = 1 / samples

        # initialise ray statistics
        ray_count = 0

        # generate list of ray origin point and vectors
        origin_points = self._point_generator(samples)
        direction_vectors = self._vector_generator(samples)

        # launch rays and accumulate spectral samples
        for origin, direction in zip(origin_points, direction_vectors):

            if transform is not None:
                origin = origin.transform(transform)
                direction = direction.transform(transform)

            ray = ray_template.copy(origin, direction)
            sample = ray.trace(world)
            spectrum.samples += weight * sample.samples

            # accumulate statistics
            ray_count += ray.ray_count

        return spectrum, ray_count

    def _start_display(self):
        """
        Display live render.
        """

        display_timer = 0
        if self.display_progress:
            self.display()
            display_timer = time()
        return display_timer

    def _update_display(self, display_timer):
        """
        Update live render.
        """

        # update live render display
        if self.display_progress and (time() - display_timer) > self.display_update_time:

            print("Refreshing display...")
            self.display()
            display_timer = time()

        return display_timer

    def _final_display(self):
        """
        Display final frame.
        """

        if self.display_progress:
            self.display()

    def _start_statistics(self):
        """
        Initialise statistics.
        """

        total_pixels = self._pixels[0] * self._pixels[1]
        total_work = total_pixels * self.spectral_rays
        ray_count = 0
        start_time = time()
        progress_timer = time()
        return total_work, total_pixels, ray_count, start_time, progress_timer

    def _update_statistics(self, statistics_data, index, pixel, sample_ray_count):
        """
        Display progress statistics.
        """

        total_work, total_pixels, ray_count, start_time, progress_timer = statistics_data
        ray_count += sample_ray_count

        if (time() - progress_timer) > 1.0:

            current_work = total_pixels * index + pixel
            completion = 100 * current_work / total_work
            print("{:0.2f}% complete (channel {}/{}, line {}/{}, pixel {}/{}, {:0.1f}k rays)".format(
                completion, index + 1, self.spectral_rays,
                ceil((pixel + 1) / self._pixels[0]), self._pixels[1],
                pixel + 1, total_pixels, ray_count / 1000))
            ray_count = 0
            progress_timer = time()

        return total_work, total_pixels, ray_count, start_time, progress_timer

    def _final_statistics(self, statistics_data):
        """
        Final statistics output.
        """
        _, _, _, start_time, _ = statistics_data
        elapsed_time = time() - start_time
        print("Render complete - time elapsed {:0.3f}s".format(elapsed_time))

    def display(self):
        clf()
        imshow(self.frame, aspect="equal", origin="upper")
        draw()
        show()
        # workaround for interactivity for QT backend
        pause(0.1)

    def save(self, filename):
        imsave(filename, self.frame)


class OrthographicCamera(Camera):
    """ A camera observing an orthogonal (orthographic) projection of the scene, avoiding perspective effects.

    :param pixels: tuple containing the number of pixels along horizontal and vertical axis
    :param width: width of the area to observe in meters, the height is deduced from the 'pixels' attribute
    :param spectral_samples: number of spectral samples by ray
    :param rays: number of rays. The total spectrum will be divided over all the rays. The number of rays must be >1 for
     dispersive effects.
    :param parent: the scenegraph node which will be the parent of this observer.
    :param transform: AffineMatrix describing the relative position/rotation of this node to the parent.
    :param name: a printable name.
    """

    def __init__(self, pixels=(512, 512), width=10, sensitivity=1.0, spectral_samples=20, spectral_rays=1,
                 pixel_samples=100, sub_sample=False, process_count=cpu_count(), parent=None,
                 transform=AffineMatrix3D(), name=None):

        super().__init__(pixels=pixels, sensitivity=sensitivity, spectral_samples=spectral_samples,
                         spectral_rays=spectral_rays, pixel_samples=pixel_samples, process_count=process_count,
                         parent=parent, transform=transform, name=name)

        self.sub_sample = sub_sample
        self.width = width

        self._update_image_geometry()

        self._point_generator = Rectangle(self.image_delta, self.image_delta)
        self._vector_generator = SingleRay()

    def _update_image_geometry(self):

        self.image_delta = self._width / self._pixels[1]
        self.image_start_x = 0.5 * self._pixels[1] * self.image_delta
        self.image_start_y = 0.5 * self._pixels[0] * self.image_delta

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):

        # a validation
        self._pixels = pixels
        self._update_image_geometry()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        if width <= 0:
            raise ValueError("width can not be less than or equal to 0 meters.")
        self._width = width
        self._update_image_geometry()

    def _generate_pixel_transform(self, i, j):

        pixel_x = self.image_start_x - self.image_delta * i
        pixel_y = self.image_start_y - self.image_delta * j
        return translate(pixel_x, pixel_y, 0)
