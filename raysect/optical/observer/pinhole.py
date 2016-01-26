
from multiprocessing import cpu_count
from math import tan, pi as PI
import numpy as np

from raysect.core import Point3D, Vector3D
from raysect.optical.observer.point_generator import Rectangle
from raysect.optical.observer.camera import Camera


class PinholeCamera(Camera):

    def __init__(self, pixels=(512, 512), fov=45, sensitivity=1.0, spectral_samples=21, spectral_rays=1,
                 pixel_samples=100, process_count=0, parent=None, transform=None, name=None):

        super().__init__(pixels=pixels, sensitivity=sensitivity, spectral_samples=spectral_samples,
                         spectral_rays=spectral_rays, pixel_samples=pixel_samples, process_count=process_count,
                         parent=parent, transform=transform, name=name)

        self._fov = fov
        self._update_image_geometry()
        self.point_generator = Rectangle(self.image_delta, self.image_delta)

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):
        if len(pixels) != 2:
            raise ValueError("Pixel dimensions of camera frame-buffer must be a tuple "
                             "containing the x and y pixel counts.")
        self._pixels = pixels

        # reset frames
        self.xyz_frame = np.zeros((self._pixels[1], self._pixels[0], 3))
        self.rgb_frame = np.zeros((self._pixels[1], self._pixels[0], 3))
        self.accumulated_samples = 0

        # update pixel geometry
        self._update_image_geometry()

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, fov):
        if fov <= 0:
            raise ValueError("Field of view angle can not be less than or equal to 0 degrees.")
        self._fov = fov

    def _update_image_geometry(self):

        max_pixels = max(self._pixels)

        if max_pixels > 1:

            # Get width of image plane at a distance of 1m from aperture.
            image_max_width = 2 * tan(PI / 180 * 0.5 * self._fov)

            # set pixel step size in image plane
            self.image_delta = image_delta = image_max_width / max_pixels

            self.image_start_x = 0.5 * self._pixels[1] * image_delta
            self.image_start_y = 0.5 * self._pixels[0] * image_delta

        else:
            raise RuntimeError("Number of Pinhole camera Pixels must be > 1.")

    def _generate_rays(self, ix, iy, ray_template):

        # generate pixel transform
        pixel_x = self.image_start_x - self.image_delta * ix
        pixel_y = self.image_start_y - self.image_delta * iy
        pixel_origin = Point3D(pixel_x, pixel_y, 1)

        points = self.point_generator(self.pixel_samples)

        # assemble rays
        rays = []
        for point in points:
            # calculate point in image plane to be used for ray direction.
            img_point = Point3D(point.x + pixel_origin.x, point.y + pixel_origin.y, point.z + pixel_origin.z)
            rays.append(ray_template.copy(Point3D(), Vector3D(img_point.x, img_point.y, img_point.z)))

        return rays
