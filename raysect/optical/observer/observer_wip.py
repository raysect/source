# Copyright (c) 2016, Dr Alex Meakins, Raysect Project
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

import numpy as np
from random import shuffle

from raysect.core.workflow import MulticoreEngine
from raysect.optical.scenegraph import Observer, World
from raysect.optical.observer.frame import Pixel, Frame2D
from raysect.optical.colour import resample_ciexyz, ciexyz_to_srgb, spectrum_to_ciexyz


class SpectralChunk(Pixel):

    def __init__(self):
        super().__init__()



class Pipeline2D:
    """
    base class defining the core interfaces to define an image processing pipeline
    """

    def initialise(self, pixels, ray_templates):
        """
        setup internal buffers (e.g. frames)
        reset internal statistics as appropriate
        etc..

        :return:
        """
        pass


class PixelProcessor:

    def add_sample(self, spectrum):
        pass

    def pack_results(self):
        pass


class RGBPipeline2D(Pipeline2D):

    def initialise(self, pixels, ray_templates):

        # create intermediate and final frame-buffers
        if not self.accumulate:
            self.xyz_frame = Frame2D(pixels, channels=3)
            self.rgb_frame = np.zeros((pixels[0], pixels[1], 3))

        # generate resampled XYZ curves for ray spectral ranges
        self._resampled_xyz = [resample_ciexyz(ray.min_wavelength, ray.max_wavelength, ray.num_samples) for ray in ray_templates]

        # TODO - add statistics and display initialisation

    def pixel_processor(self, channel):
        return XYZPixelProcessor(self._resampled_xyz[channel])

    def update(self, packed_result, channel):
        pass

    def finalise(self):
        pass


class XYZPixelProcessor(PixelProcessor):

    def __init__(self, resampled_xyz):
        self._resampled_xyz = resampled_xyz
        self._xyz = Pixel(channels=3)

    def add_sample(self, spectrum):
        # convert spectrum to CIE XYZ and add sample to pixel buffer
        x, y, z = spectrum_to_ciexyz(spectrum, self._resampled_xyz)
        self._xyz.add_sample(0, x)
        self._xyz.add_sample(1, y)
        self._xyz.add_sample(2, z)

    def pack_results(self):

        mean = (self._xyz.value[0], self._xyz.value[1], self._xyz.value[2])
        variance = (self._xyz.variance[0], self._xyz.variance[1], self._xyz.variance[2])
        samples = (self._xyz.samples[0], self._xyz.samples[1], self._xyz.samples[2])

        return mean, variance, samples


class FrameSampler:

    def generate_tasks(self, pixels):
        pass


class FullFrameSampler(FrameSampler):

    def generate_tasks(self, pixels):

        tasks = []
        nx, ny = pixels
        for iy in range(ny):
            for ix in range(nx):
                tasks.append((ix, iy))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


# TODO - add Adaptive Sampler
# class AdaptiveSampler(FrameSampler):
#
#     def generate_tasks(self, pixels):
#
#         # build task list -> task is simply the pixel location
#         tasks = []
#         nx, ny = pixels
#         f = self.xyz_frame
#         i = f.value > 0
#         norm_frame_var = np.zeros((pixels[0], pixels[1], 3))
#         norm_frame_var[i] = f.variance[i]
#         # norm_frame_var[i] = np.minimum(f.variance[i], f.variance[i] / f.value[i]**2)
#         max_frame_samples = f.samples.max()
#         percentile_frame_variance = np.percentile(norm_frame_var, 80)
#         for iy in range(ny):
#             for ix in range(nx):
#                 min_pixel_samples = self.xyz_frame.samples[ix, iy, :].min()
#                 max_pixel_variance = norm_frame_var[ix, iy, :].max()
#                 if min_pixel_samples < 1000*self.spectral_rays or \
#                     min_pixel_samples <= 0.1 * max_frame_samples or \
#                     max_pixel_variance >= percentile_frame_variance:
#                     tasks.append((ix, iy))




class Observer2D(Observer):
    """
    - Needs to know about mean, max, min wavelength, number of samples, rays.
    - Things it will do:
        - initialise statistics between each pipeline and this object
        - generate the pixel rendering tasks using the frame sampler to select
        which pixels to render and how.
        - implement render_pixel() and update_frame() of the render engine. Then
        start the render engine.
        - finally closes the stats and makes final display.

    - inside render_pixel()
        - passed in are ray_template and pipeline object references
        - unpack task ID (pixel id)
        - launch the rays and sample pixel using pixel_sampler() and ray_template,
        returns a frame 1D object representing a spectrum with variance and sample count
        for each bin.
        - passes frame 1D to each pipeline object for pixel processing. Each of those
        returns a custum tuple of data tied to details of that pipeline.
        - All results packaged together with pixel ID and returned to consumer.

    - inside update_frame()
        - unpack pixel ID and pipeline result tuples.
        - pass results to each pipeline to update pipelines internal state
        - print workflow statistics and any statistics for each pipeline.
        - display visual imagery for each pipeline as required
        - save state for each pipeline as required.
    """

    def __init__(self, parent=None, transform=None, name=None):
        super().__init__(parent, transform, name)

    def construct(self, frame_sampler, processing_pipelines, render_engine=None):

        # TODO - add validation
        self.frame_sampler = frame_sampler
        self.processing_pipelines = processing_pipelines
        self.render_engine = render_engine or MulticoreEngine()

        self.spectral_rays = spectral_rays
        self.spectral_samples = spectral_samples
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.ray_extinction_prob = ray_extinction_prob
        self.ray_min_depth = ray_min_depth
        self.ray_max_depth = ray_max_depth
        self.ray_importance_sampling = ray_importance_sampling
        self.ray_important_path_weight = ray_important_path_weight

        # progress information
        self.display_progress = display_progress
        self.display_update_time = display_update_time

        # camera configuration
        self.pixels = pixels
        self.pixel_samples = pixel_samples

    def observe(self):
        """ Ask this Camera to Observe its world. """

        # must be connected to a world node to be able to perform a ray trace
        if not isinstance(self.root, World):
            raise TypeError("Observer is not connected to a scene graph containing a World object.")

        if self.min_wavelength >= self.max_wavelength:
            raise RuntimeError("Min wavelength is superior to max wavelength!")

        # TODO - initialise workflow statistics

        # generate ray templates
        ray_templates = self._generate_ray_templates()

        # initialise pipelines for rendering
        for pipeline in self.pipelines:
            pipeline.initialise(self.pixels, ray_templates)

        tasks = self.frame_sampler.generate_tasks()

        # render
        for channel, ray_template in enumerate(ray_templates):

            self.render_engine.run(
                tasks, self._render_pixel, self._update_pipelines,
                render_args=(self.root, ray_template, channel),
                update_args=(channel, )
            )

        # close pipelines
        for pipeline in self.pipelines:
            pipeline.finalise()

    #################
    # WORKER THREAD #
    #################

    def _render_pixel(self, pixel_id, ray_template, channel):
        """
        - passed in are ray_template and pipeline object references
        - unpack task ID (pixel id)
        - launch the rays and sample pixel using pixel_sampler() and ray_template,
        returns a frame 1D object representing a spectrum with variance and sample count
        for each bin.
        - passes frame 1D to each pipeline object for pixel processing. Each of those
        returns a custum tuple of data tied to details of that pipeline.
        - All results packaged together with pixel ID and returned to consumer.
        :return:
        """

        # obtain reference to world
        world = self.root

        # unpack pixel coords from task
        ix, iy = pixel_id

        # generate rays
        rays = self._generate_rays(ix, iy, ray_template)

        pixel_processors = [pipeline.pixel_processor(channel) for pipeline in self.processing_pipelines]

        # initialise ray statistics
        ray_count = 0

        # launch rays and accumulate spectral samples
        for ray, projection_weight in rays:

            # convert ray from local space to world space
            ray.origin = ray.origin.transform(self.to_root())
            ray.direction = ray.direction.transform(self.to_root())

            # sample and apply projection weight
            spectrum = ray.trace(world)
            spectrum.mul_scalar(projection_weight)

            for processor in pixel_processors:
                processor.add_sample(spectrum)

            # accumulate statistics
            ray_count += ray.ray_count

        # acquire results from pixel processors
        results = [processor.pack_results() for processor in pixel_processors]

        return pixel_id, results, ray_count

    def _generate_rays(self, ix, iy, ray_template):
        """
        Generate a list of Rays and their respective area weightings for pixel (ix, iy).

        This is a virtual method to be implemented by derived classes.

        Runs during the observe() loop to generate the rays. Allows observers to customise how they launch rays.

        This method must return a list of tuples, with each tuple containing
        a ray object and a corresponding projected area weight (direction cosine).

        If the projected area weight is not required (due to the ray sampling
        algorithm taking the weighting into account in the distribution e.g.
        cosine weighted) then the weight should be set to 1.0.

        :param int ix: Index of this pixel along the images' x axis.
        :param int iy: Index of this pixel along the images' y axis.
        :param Ray ray_template: A Ray object from which spectral settings can be propagated.
        :return list Rays: A list of tuples.
        """

        raise NotImplementedError("To be defined in subclass.")

    ###################
    # CONSUMER THREAD #
    ###################

    def _update_pipelines(self, packed_result, channel):
        """
        - unpack pixel ID and pipeline result tuples.
        - pass results to each pipeline to update pipelines internal state
        - print workflow statistics and any statistics for each pipeline.
        - display visual imagery for each pipeline as required
        - save state for each pipeline as required.

        :return:
        """

        # unpack worker results
        pixel_id, results, ray_count = packed_result

        for result, pipeline in zip(results, self.processing_pipelines):
            pipeline.update(result, channel)

        # update users
        # TODO - add statistics

    # def _start_statistics(self):
    #     """
    #     Initialise statistics.
    #     """
    #
    #     total_pixels = self._pixels[0] * self._pixels[1]
    #     total_work = total_pixels * self.spectral_rays
    #     ray_count = 0
    #     start_time = time()
    #     progress_timer = time()
    #     self.__statistics_data = (total_work, total_pixels, ray_count, start_time, progress_timer)
    #
    #     for pipeline in self.processing_pipelines:
    #         pipeline.start_statistics()
    #
    # def _update_statistics(self, channel, x, y, sample_ray_count):
    #     """
    #     Display progress statistics.
    #     """
    #
    #     total_work, total_pixels, ray_count, start_time, progress_timer = self.__statistics_data
    #     ray_count += sample_ray_count
    #     nx, ny = self._pixels
    #     pixel = y*ny + x
    #
    #     if (time() - progress_timer) > 1.0:
    #
    #         current_work = total_pixels * channel + pixel
    #         completion = 100 * current_work / total_work
    #         print("{:0.2f}% complete (channel {}/{}, line {}/{}, pixel {}/{}, {:0.1f}k rays)".format(
    #             completion,
    #             channel + 1, self.spectral_rays,
    #             ceil((pixel + 1) / nx), ny,
    #             pixel + 1, total_pixels, ray_count / 1000
    #
    #         # update pipeline statistics
    #         for pipeline in self.processing_pipelines:
    #             pipeline.update_statistics()
    #
    #         ray_count = 0
    #         progress_timer = time()
    #
    #     self.__statistics_data = (total_work, total_pixels, ray_count, start_time, progress_timer)
    #
    # def _final_statistics(self):
    #     """
    #     Final statistics output.
    #     """
    #
    #     for pipeline in self.processing_pipelines:
    #         pipeline.final_statistics()
    #
    #     _, _, _, start_time, _ = self.__statistics_data
    #     elapsed_time = time() - start_time
    #     print("Render complete - time elapsed {:0.3f}s".format(elapsed_time))
