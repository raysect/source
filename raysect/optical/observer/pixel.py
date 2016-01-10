
"""
All pixels have a common base class, the Pixel class, which defines common properties such as multiprocessor sampling,
etc. Actual used Pixels are dived into two types:
- TransferFunctionPixel
- VectorSamplerPixel
"""

from raysect.optical import Spectrum, Ray


class Pixel:
    """
    Base class for all Pixel observers.

    All pixel types have:
     - a pixel transform which defines their offset/location
     - a point generator which effectively defines their sampling surface area.
    """
    def __init__(self, pixel_index, pixel_origin, to_pixel_origin, to_world, point_generator, manipulation_func=None):
        self.pixel_index = pixel_index
        self.pixel_origin = pixel_origin
        self.to_pixel_origin = to_pixel_origin
        self.to_world = to_world
        self.point_generator = point_generator
        self.manipulation_func = manipulation_func

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

        # print("#############")
        # print("Building rays")
        # print()
        # print('ix - {}, ij - {}'.format(self.pixel_index[0], self.pixel_index[0]))
        # print('pixel origin - {}'.format(self.pixel_origin))

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


# TODO - this name is terrible. Need to rename.
class VectorSamplerPixel(Pixel):

    def __init__(self, pixel_index, pixel_origin, to_pixel_origin, to_world, point_generator, vector_generator):
        """

        :param AffineMatrix3D to_world: Transform matrix from camera space to world space.
        :param point_generator: Class that generates points on pixel surface for sampling over pixel area.
        :param vector_generator: Class that generates vectors for sampling.
        :return:
        """

        super().__init__(pixel_index, pixel_origin, to_pixel_origin, to_world, point_generator)

        self.vector_generator = vector_generator

    def build_rays(self, num_pixel_samples, min_wavelength, max_wavelength, spectral_samples):

        rays = []
        points = self.point_generator.sample(num_pixel_samples)
        directions = self.vector_generator.sample(num_pixel_samples)

        # print(points)
        # print(directions)

        if self.manipulation_func:
            points, directions = self.manipulation_func(points, directions)

        # print(points)
        # print(directions)
        # input("...")

        for i in range(num_pixel_samples):

            point_in_world = points[i].transform(self.to_world)
            direction_in_world = directions[i].transform(self.to_world)
            # print(point_in_world)
            # print(direction_in_world)
            # input("...")
            # generate ray and add to array to return
            rays.append(
                Ray(point_in_world, direction_in_world, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                    num_samples=spectral_samples,
                    # TODO - parse in these parameters
                    # extinction_prob=self.ray_extinction_prob,
                    # min_depth=self.ray_min_depth,
                    # max_depth=self.ray_max_depth
                    )
            )

        return rays


class TransferFunctionPixel(Pixel):

    def __init__(self, pixel_index, pixel_origin, to_pixel_origin, to_world, point_generator, transfer_function):

        super().__init__(pixel_index, pixel_origin, to_pixel_origin, to_world, point_generator)

        # TODO - work out logic and structure for the transfer equation
        self.vector_generator = transfer_function

    def build_rays(self, pixel_samples, min_wavelength, max_wavelength, spectral_samples):
        pass
