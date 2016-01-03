
"""
All pixels have a common base class, the Pixel class, which defines common properties such as multiprocessor sampling,
etc. Actual used Pixels are dived into two types:
- TransferFunctionPixel
- VectorSamplerPixel
"""

from raysect.optical import Spectrum
from raysect.optical.ray import Ray


class Pixel:
    """
    Base class for all Pixel observers.

    All pixel types have:
     - a pixel transform which defines their offset/location
     - a point generator which effectively defines their sampling surface area.
    """
    def __init__(self, pixel_transform, point_generator):
        self.pixel_transform = pixel_transform
        self.point_generator = point_generator

    def sample_pixel(self, min_wavelength, max_wavelength, spectral_samples, camera):
        """
        :param spectrum_instructions:
        :param world:
        :return:
        """
        world = camera.root
        num_pixel_samples = camera.pixel_samples
        sensitivity = camera.sensitivity

        spectrum = Spectrum(min_wavelength, max_wavelength, spectral_samples)

        # obtain rays for this pixel
        rays = self.build_rays(num_pixel_samples, min_wavelength, max_wavelength, spectral_samples)

        weight = 1 / len(rays)
        ray_count = 0

        for ray in rays:
            # trace
            sample = ray.trace(world)

            # camera sensitivity
            spectrum.samples += weight * sensitivity * sample.samples

            # accumulate statistics
            ray_count += ray.ray_count

        return spectrum, ray_count

    def build_rays(self, num_pixel_samples, min_wavelength, max_wavelength, spectral_samples):
        raise NotImplementedError("Function build_rays must be implemented on this Pixel class.")


class VectorSamplerPixel(Pixel):

    def __init__(self, pixel_transform, point_generator, vector_generator):
        """

        :param AffineMatrix3D pixel_transform: Transform matrix with respect to parent.
        :param point_generator: Class that generates points on pixel surface for sampling over pixel area.
        :param vector_generator: Class that generates vectors for sampling.
        :return:
        """

        super().__init__(pixel_transform, point_generator)

        self.vector_generator = vector_generator

    def build_rays(self, num_pixel_samples, min_wavelength, max_wavelength, spectral_samples):

        rays = []
        # TODO - these need to be in the correct space
        directions = self.vector_generator.sample(num_pixel_samples)
        origins = self.point_generator.sample(num_pixel_samples)

        for i in range(num_pixel_samples):

            # generate ray and add to array to return
            rays.append(
                Ray(origins[i], directions[i],
                    min_wavelength=min_wavelength,
                    max_wavelength=max_wavelength,
                    num_samples=spectral_samples,
                    # TODO - parse in these parameters
                    # extinction_prob=self.ray_extinction_prob,
                    # min_depth=self.ray_min_depth,
                    # max_depth=self.ray_max_depth
                )
            )

        return rays


class TransferFunctionPixel(Pixel):

    def __init__(self, pixel_transform, point_generator, transfer_function):

        super().__init__(pixel_transform, point_generator)

        # TODO - work out logic and structure for the transfer equation
        self.vector_generator = transfer_function

    def build_rays(self, pixel_samples, min_wavelength, max_wavelength, spectral_samples):
        pass

