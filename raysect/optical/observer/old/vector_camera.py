from raysect.optical.observer.old.sensor import Imaging


class VectorCamera(Imaging):

    """
    An observer that uses a specified set of pixel vectors.

    A simple camera that uses calibrated vectors for each pixel to sample the scene.

    Arguments and attributes are inherited from the base Imaging sensor class.

    :param double fov: The field of view of the camera in degrees (default is 90 degrees).
    """

    def __init__(self, pixel_origins, pixel_directions, pixels=(512, 512), exposure_handler=None, spectral_samples=21,
                 spectral_rays=1, pixel_samples=100, parent=None, transform=None, name=None):

        super().__init__(pixels=pixels, exposure_handler=exposure_handler, spectral_samples=spectral_samples,
                         spectral_rays=spectral_rays, pixel_samples=pixel_samples,
                         parent=parent, transform=transform, name=name)

        # camera configuration
        self.pixel_origins = pixel_origins
        self.pixel_directions = pixel_directions

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):
        raise RuntimeError('Pixel shape cannot be changed for a VectorCamera.')

    def _generate_rays(self, ix, iy, ray_template):

        # assemble rays
        origin = self.pixel_origins[ix, iy]
        direction = self.pixel_directions[ix, iy]

        # assemble rays
        rays = []
        for i in range(self.pixel_samples):

            ray = ray_template.copy(origin, direction)

            # projected area weight is normal.incident which simplifies
            # to incident.z here as the normal is (0, 0 ,1)
            rays.append((ray, 1.0))

        return rays



