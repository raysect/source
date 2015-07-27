
from time import time
from math import tan, pi, ceil
from multiprocessing import Process, Manager, cpu_count, Queue

from numpy import array, zeros
from matplotlib.pyplot import imshow, imsave, show, ion, ioff, clf, figure, draw, pause

from raysect.optical.ray import Ray
from raysect.optical import Spectrum
from raysect.core import World, AffineMatrix, Point, Vector, Observer
from raysect.optical.colour import resample_ciexyz, spectrum_to_ciexyz, ciexyz_to_srgb


class VectorCamera(Observer):

    def __init__(self, pixel_origins, pixel_directions, name="", sensitivity=1.0, spectral_samples=20, rays=1,
                 super_samples=1, process_count=cpu_count(), parent=None, transform=AffineMatrix()):

        super().__init__(parent, transform, name)

        # camera configuration
        self.pixel_origins = pixel_origins
        self.pixel_directions = pixel_directions
        self._pixels = pixel_directions.shape
        self.sensitivity = sensitivity
        self.super_samples = super_samples

        # ray configuration
        self.rays = rays
        self.spectral_samples = spectral_samples
        self.min_wavelength = 375.0
        self.max_wavelength = 740.0
        self.ray_max_depth = 15

        # progress information
        self.display_progress = True
        self.display_update_time = 10.0

        # concurrency configuration
        self.process_count = process_count

        # Dimensions of 3 are due to RGB samples/colour channels, etc.
        self.frame = zeros((self._pixels[1], self._pixels[0], 3))

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):
        if len(pixels) != 2:
            raise ValueError("Pixel dimensions of camera framebuffer must be a tuple containing the x and y pixel counts.")
        self._pixels = pixels

    def observe(self):

        # TODO - This doesn't look write. This should already have been checked in attribute getter/setter
        # must be connected to a world node to be able to perform a ray trace
        if not isinstance(self.root, World):
            raise TypeError("Observer is not connected to a scene graph containing a World object.")

        # create intermediate and final frame-buffers
        xyz_frame = zeros((self._pixels[1], self._pixels[0], 3))
        self.frame = zeros((self._pixels[1], self._pixels[0], 3))

        world = self.root
        total_pixels = self._pixels[0] * self._pixels[1]

        # generate spectral data
        channel_configs = self._calc_channel_config()

        # display live render
        display_timer = 0
        if self.display_progress:

            self.display()
            display_timer = time()

        # initialise statistics
        total_work = total_pixels * self.rays
        ray_count = 0
        start_time = time()
        progress_timer = time()

        # render
        for index, channel_config in enumerate(channel_configs):

            min_wavelength, max_wavelength, spectral_samples = channel_config

            # generate resampled XYZ curves for channel spectral range
            resampled_xyz = resample_ciexyz(min_wavelength, max_wavelength, spectral_samples)

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
                            args=(world, min_wavelength, max_wavelength, spectral_samples,
                                  resampled_xyz, task_queue, result_queue))
                p.start()
                workers.append(p)

            # collect results
            for pixel in range(total_pixels):

                # obtain result
                location, xyz, sample_ray_count = result_queue.get()
                x, y = location

                # collect ray statistics
                ray_count += sample_ray_count

                # accumulate colour
                xyz_frame[y, x, 0] += xyz[0]
                xyz_frame[y, x, 1] += xyz[1]
                xyz_frame[y, x, 2] += xyz[2]

                # convert to sRGB colourspace
                self.frame[y, x, :] = ciexyz_to_srgb(*xyz_frame[y, x, :])

                # display progress statistics
                if (time() - progress_timer) > 1.0:

                    current_work = total_pixels * index + pixel
                    completion = 100 * current_work / total_work
                    print("{:0.2f}% complete (channel {}/{}, line {}/{}, pixel {}/{}, {:0.1f}k rays)".format(
                        completion, index + 1, self.rays,
                        ceil((pixel + 1) / self._pixels[0]), self._pixels[1],
                        pixel + 1, total_pixels, ray_count / 1000))
                    ray_count = 0
                    progress_timer = time()

                # update live render display
                if self.display_progress and (time() - display_timer) > self.display_update_time:

                    print("Refreshing display...")
                    self.display()
                    display_timer = time()

            # shutdown workers
            for _ in workers:
                task_queue.put(None)

        # close statistics
        elapsed_time = time() - start_time
        print("Render complete - time elapsed {:0.3f}s".format(elapsed_time))

        # display final frame
        if self.display_progress:

            self.display()

    def _calc_channel_config(self):

        config = []
        delta_wavelength = (self.max_wavelength - self.min_wavelength) / self.rays
        for index in range(self.rays):
            config.append((self.min_wavelength + delta_wavelength * index,
                           self.min_wavelength + delta_wavelength * (index + 1),
                           self.spectral_samples))

        return config

    def display(self):

        clf()
        imshow(self.frame, aspect="equal", origin="upper")
        draw()
        show()

        # workaround for interactivity for QT backend
        pause(0.1)

    def save(self, filename):

        imsave(filename, self.frame)

    def _producer(self, task_queue):

        # task is simply the pixel location
        nx, ny = self._pixels
        for y in range(ny):
            for x in range(nx):
                task_queue.put((x, y))

    def _worker(self, world, min_wavelength, max_wavelength, spectral_samples, resampled_xyz, task_queue, result_queue):

        while True:

            # request next pixel
            pixel = task_queue.get()

            # have we been commanded to shutdown?
            if pixel is None:
                break

            # extract pixel coordinate
            x, y = pixel

            origin = self.pixel_origins[x, y]
            direction = self.pixel_directions[x, y]

            spectrum = Spectrum(min_wavelength, max_wavelength, spectral_samples)

            # generate ray
            ray = Ray(origin, direction, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                      num_samples=spectral_samples, max_depth=self.ray_max_depth)

            # trace
            sample = ray.trace(world)

            # camera sensitivity
            spectrum.samples += self.sensitivity * sample.samples

            # convert spectrum to CIE XYZ
            xyz = spectrum_to_ciexyz(spectrum, resampled_xyz)

            # encode result and send
            result = (pixel, xyz, 1)
            result_queue.put(result)

