from time import time
from math import tan, pi, ceil
from multiprocessing import Process, Manager, cpu_count, Queue

from numpy import zeros, empty, ndenumerate, nditer
from matplotlib.pyplot import imshow, imsave, show, ion, ioff, clf, figure, draw, pause

from raysect.optical.ray import Ray
from raysect.optical import Spectrum
from raysect.core import World, AffineMatrix, Point, Vector, Observer
from raysect.optical.colour import resample_ciexyz, spectrum_to_ciexyz, ciexyz_to_srgb
from collections import namedtuple

# Variables
_pinhole_camera_pixel_vector_variables = namedtuple("Pixel_Vector_Variables", "origin image_delta image_start_x image_start_y")


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
        self._max_pixels = max(self._pixels)
        self.sensitivity = sensitivity
        self.super_samples = super_samples
        self._pixel_vectors_variables = None

        # output from last call to Observe()
        self.frame = zeros((pixels[1], pixels[0], 3))

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):
        if len(pixels) != 2:
            raise ValueError("Pixel dimensions of camera framebuffer must be a tuple "
                             "containing the x and y pixel counts.")
        self._pixels = pixels

    def observe(self):
        # TODO - this test should move to a getter/setter on Node class???
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

        # Setup pixel Vectors in advance of main loop
        pixel_configuration = self._setup_pixel_vectors_variables()

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
                                  resampled_xyz, pixel_configuration, task_queue, result_queue))
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

    def _producer(self, task_queue):
        # task is simply the pixel location
        nx, ny = self._pixels
        for y in range(ny):
            for x in range(nx):
                task_queue.put((x, y))

    def _worker(self, world, min_wavelength, max_wavelength, spectral_samples, resampled_xyz, pixel_configuration, task_queue, result_queue):

        while True:

            # request next pixel
            pixel = task_queue.get()

            # have we been commanded to shutdown?
            if pixel is None:
                break

            # get all direction vectors for this pixel
            x, y = pixel
            rays = self._get_pixel_rays(x, y, min_wavelength, max_wavelength, spectral_samples, pixel_configuration)

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

            # convert spectrum to CIE XYZ
            xyz = spectrum_to_ciexyz(spectrum, resampled_xyz)

            # encode result and send
            result = (pixel, xyz, ray_count)
            result_queue.put(result)

    def _setup_pixel_vectors_variables(self):
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

    def _setup_pixel_vectors_variables(self):

        if self._max_pixels > 1:
            # generate ray directions by simulating an image plane 1m from pinhole "aperture"
            # max width of image plane at 1 meter for given field of view
            image_max_width = 2 * tan(pi / 180 * 0.5 * self._fov)

            # pixel step and start point in image plane
            image_delta = image_max_width / (self._max_pixels - 1)

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

    def _setup_pixel_vectors_variables(self):
        pass

    def _get_pixel_rays(self, x, y, min_wavelength, max_wavelength, spectral_samples, pixel_configuration):
        # TODO - support super_samples > 1
        origin = self.pixel_origins[x, y]
        direction = self.pixel_directions[x, y]

        return [
            Ray(origin, direction, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                num_samples=spectral_samples, max_depth=self.ray_max_depth)
        ]
