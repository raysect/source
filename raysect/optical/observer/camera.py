from time import time
from math import tan, pi, ceil
from multiprocessing import Process, cpu_count, Queue

from numpy import zeros
from matplotlib.pyplot import imshow, imsave, show, clf, draw, pause

from raysect.optical.ray import Ray
from raysect.optical import Spectrum
from raysect.core import World, AffineMatrix, Point, Vector, Observer
from raysect.optical.colour import resample_ciexyz, spectrum_to_ciexyz, ciexyz_to_srgb


class Camera(Observer):

    def __init__(self, pixels=(512, 512), sensitivity=1.0, spectral_samples=20, rays=1, super_samples=1,
                 process_count=cpu_count(), parent=None, transform=AffineMatrix(), name=""):

        super().__init__(parent, transform, name)

        # TODO: Add sanity checks

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

        # camera configuration
        self._pixels = pixels
        self.sensitivity = sensitivity
        self.super_samples = super_samples

        # output from last call to Observe()
        self.frame = zeros((pixels[1], pixels[0], 3))

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):
        if len(pixels) != 2:
            raise ValueError("Pixel dimensions of camera frame-buffer must be a tuple "
                             "containing the x and y pixel counts.")
        self._pixels = pixels

    def observe(self):

        # must be connected to a world node to be able to perform a ray trace
        if not isinstance(self.root, World):
            raise TypeError("Observer is not connected to a scene graph containing a World object.")
        world = self.root

        # create intermediate and final frame-buffers
        xyz_frame = zeros((self._pixels[1], self._pixels[0], 3))
        self.frame = zeros((self._pixels[1], self._pixels[0], 3))

        # generate spectral data
        channel_configs = self._calc_channel_config()

        # setup pixel vectors in advance of main loop
        pixel_config = self._setup_pixel_config()

        if self.process_count == 1:
            self._observe_single(world, xyz_frame, channel_configs, pixel_config)
        else:
            self._observe_parallel(world, xyz_frame, channel_configs, pixel_config)

    def _observe_single(self, world, xyz_frame, channel_configs, pixel_config):

        # initialise user interface
        display_timer = self._start_display()
        statistics_data = self._start_statistics()

        # render
        for channel, channel_config in enumerate(channel_configs):

            min_wavelength, max_wavelength, spectral_samples = channel_config

            # generate resampled XYZ curves for channel spectral range
            resampled_xyz = resample_ciexyz(min_wavelength, max_wavelength, spectral_samples)

            nx, ny = self._pixels
            for y in range(ny):

                for x in range(nx):

                    spectrum, ray_count = self._sample_pixel(x, y, min_wavelength, max_wavelength, spectral_samples,
                                                             pixel_config, world)

                    # convert spectrum to CIE XYZ
                    xyz = spectrum_to_ciexyz(spectrum, resampled_xyz)

                    # accumulate colour
                    xyz_frame[y, x, 0] += xyz[0]
                    xyz_frame[y, x, 1] += xyz[1]
                    xyz_frame[y, x, 2] += xyz[2]

                    # convert to sRGB colour-space
                    self.frame[y, x, :] = ciexyz_to_srgb(*xyz_frame[y, x, :])

                    # update users
                    pixel = x + self._pixels[0] * y
                    statistics_data = self._update_statistics(statistics_data, channel, pixel, ray_count)
                    display_timer = self._update_display(display_timer)

        # final update for users
        self._final_statistics(statistics_data)
        self._final_display()

    def _observe_parallel(self, world, xyz_frame, channel_configs, pixel_config):

        # initialise user interface
        display_timer = self._start_display()
        statistics_data = self._start_statistics()

        total_pixels = self._pixels[0] * self._pixels[1]

        # render
        for channel, channel_config in enumerate(channel_configs):

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
                                  resampled_xyz, pixel_config, task_queue, result_queue))
                p.start()
                workers.append(p)

            # collect results
            for pixel in range(total_pixels):

                # obtain result
                location, xyz, sample_ray_count = result_queue.get()
                x, y = location

                # accumulate colour
                xyz_frame[y, x, 0] += xyz[0]
                xyz_frame[y, x, 1] += xyz[1]
                xyz_frame[y, x, 2] += xyz[2]

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

    def _worker(self, world, min_wavelength, max_wavelength, spectral_samples, resampled_xyz, pixel_config, task_queue, result_queue):

        while True:

            # request next pixel
            pixel = task_queue.get()

            # have we been commanded to shutdown?
            if pixel is None:
                break

            x, y = pixel
            spectrum, ray_count = self._sample_pixel(x, y, min_wavelength, max_wavelength, spectral_samples,
                                                     pixel_config, world)

            # convert spectrum to CIE XYZ
            xyz = spectrum_to_ciexyz(spectrum, resampled_xyz)

            # encode result and send
            result = (pixel, xyz, ray_count)
            result_queue.put(result)

    def _calc_channel_config(self):

        config = []
        delta_wavelength = (self.max_wavelength - self.min_wavelength) / self.rays
        for index in range(self.rays):
            config.append((self.min_wavelength + delta_wavelength * index,
                           self.min_wavelength + delta_wavelength * (index + 1),
                           self.spectral_samples))
        return config

    def _sample_pixel(self, x, y, min_wavelength, max_wavelength, spectral_samples, pixel_config, world):

        # obtain rays for this pixel
        rays = self._get_pixel_rays(x, y, min_wavelength, max_wavelength, spectral_samples, pixel_config)

        weight = 1 / len(rays)
        ray_count = 0
        spectrum = Spectrum(min_wavelength, max_wavelength, spectral_samples)

        for ray in rays:
            # trace
            sample = ray.trace(world)

            # camera sensitivity
            spectrum.samples += weight * self.sensitivity * sample.samples

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
        total_work = total_pixels * self.rays
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
                completion, index + 1, self.rays,
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

    def _setup_pixel_config(self):
        """
        Virtual method - to be implemented by derived classes.

        Runs at the start of observe() loop to set up any data needed for calculating pixel vectors
        and supersampling that shouldn't be calculated at every loop iteration. The result of this
        functon should be written to self._pixel_vectors_variables.
        """
        raise NotImplementedError("Virtual method _setup_pixel_vectors() has not been implemented for this Camera.")

    def _get_pixel_rays(self, x, y, min_wavelength, max_wavelength, spectral_samples, pixel_configuration):
        """
        Virtual method - to be implemented by derived classes.

        Called for each pixel in the _worker() observe loop. For a given pixel, this function must return a list of
        vectors to ray trace.
        """
        raise NotImplementedError("Virtual method _get_pixel_vectors() has not been implemented for this Camera.")

    def display(self):
        clf()
        imshow(self.frame, aspect="equal", origin="upper")
        draw()
        show()
        # workaround for interactivity for QT backend
        pause(0.1)

    def save(self, filename):
        imsave(filename, self.frame)


class PinholeCamera(Camera):

    def __init__(self, pixels=(512, 512), fov=45, sensitivity=1.0, spectral_samples=20, rays=1, super_samples=1,
                 process_count=cpu_count(), parent=None, transform=AffineMatrix(), name=""):

        super().__init__(pixels=pixels, sensitivity=sensitivity, spectral_samples=spectral_samples, rays=rays,
                         super_samples=super_samples, process_count=process_count, parent=parent,
                         transform=transform, name=name)

        self._fov = fov

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, fov):
        if fov <= 0:
            raise ValueError("Field of view angle can not be less than or equal to 0 degrees.")
        self._fov = fov

    def _setup_pixel_config(self):

        max_pixels = max(self._pixels)

        if max_pixels > 1:
            # generate ray directions by simulating an image plane 1m from pinhole "aperture"
            # max width of image plane at 1 meter for given field of view
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

        origin = Point(0, 0, 0).transform(self.to_root())

        return origin, image_delta, image_start_x, image_start_y

    def _get_pixel_rays(self, x, y, min_wavelength, max_wavelength, spectral_samples, pixel_configuration):

        origin, image_delta, image_start_x, image_start_y = pixel_configuration

        # subsample AA
        super_samples = self.super_samples
        delta = 1 / super_samples
        offset = delta / 2 - 0.5

        rays = []

        for i in range(super_samples):
            for j in range(super_samples):

                dx = delta * i + offset
                dy = delta * j + offset

                # calculate ray parameters
                direction = Vector(image_start_x - image_delta * (x + dx), image_start_y - image_delta * (y + dy), 1.0).normalise()
                direction = direction.transform(self.to_root())

                # generate ray and add to array to return
                rays.append(
                    Ray(origin, direction, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                        num_samples=spectral_samples, max_depth=self.ray_max_depth)
                )

        return rays


class VectorCamera(Camera):

    def __init__(self, pixel_origins, pixel_directions, name="", sensitivity=1.0, spectral_samples=20, rays=1,
                 super_samples=1, process_count=cpu_count(), parent=None, transform=AffineMatrix()):

        if super_samples > 1:
            raise NotImplementedError("The VectorCamera does not yet implement supersampling.")

        super().__init__(pixels=pixel_directions.shape, sensitivity=sensitivity, spectral_samples=spectral_samples,
                         rays=rays, super_samples=super_samples, process_count=process_count, parent=parent,
                         transform=transform, name=name)

        # camera configuration
        self.pixel_origins = pixel_origins
        self.pixel_directions = pixel_directions

    def _setup_pixel_config(self):
        pass

    def _get_pixel_rays(self, x, y, min_wavelength, max_wavelength, spectral_samples, pixel_configuration):
        # TODO - support super_samples > 1
        origin = self.pixel_origins[x, y]
        direction = self.pixel_directions[x, y]

        return [
            Ray(origin, direction, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                num_samples=spectral_samples, max_depth=self.ray_max_depth)
        ]
