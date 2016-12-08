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

from raysect.core.workflow import MulticoreEngine
from raysect.optical import Ray
from raysect.optical import World
from raysect.core import Observer as CoreObserver
from time import time

# TODO: cythonise me!


class Observer(CoreObserver):
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

    def __init__(self, frame_sampler, processing_pipelines, render_engine=None, parent=None, transform=None, name=None):

        super().__init__(parent, transform, name)

        # TODO - add validation
        self.frame_sampler = frame_sampler
        self.pipelines = processing_pipelines
        self.render_engine = render_engine or MulticoreEngine()

        # ray configuration
        self.spectral_rays = 1
        self.spectral_samples = 15
        self.min_wavelength = 375.0
        self.max_wavelength = 740.0
        self.ray_extinction_prob = 0.01
        self.ray_min_depth = 3
        self.ray_max_depth = 500
        self.ray_importance_sampling = True
        self.ray_important_path_weight = 0.2

        # self.spectral_rays = spectral_rays
        # self.spectral_samples = spectral_samples
        # self.min_wavelength = min_wavelength
        # self.max_wavelength = max_wavelength
        # self.ray_extinction_prob = ray_extinction_prob
        # self.ray_min_depth = ray_min_depth
        # self.ray_max_depth = ray_max_depth
        # self.ray_importance_sampling = ray_importance_sampling
        # self.ray_important_path_weight = ray_important_path_weight

        # progress information
        # self.display_progress = display_progress
        # self.display_update_time = display_update_time

        # camera configuration
        self.pixels = (64, 64)
        self.pixel_samples = 250
        # self.pixels = pixels
        # self.pixel_samples = pixel_samples

    def observe(self):
        """ Ask this Camera to Observe its world. """

        # must be connected to a world node to be able to perform a ray trace
        if not isinstance(self.root, World):
            raise TypeError("Observer is not connected to a scene graph containing a World object.")

        # generate ray templates
        ray_templates = self._generate_ray_templates()

        # initialise pipelines for rendering
        for pipeline in self.pipelines:
            pipeline.initialise(self.pixels, ray_templates)

        tasks = self.frame_sampler.generate_tasks(self.pixels)

        # initialise statistics with total task count
        self._initialise_statistics(tasks)

        # render
        for channel, ray_template in enumerate(ray_templates):

            self.render_engine.run(
                tasks, self._render_pixel, self._update_state,
                render_args=(ray_template, channel),
                update_args=(channel, )
            )

        # close pipelines
        for pipeline in self.pipelines:
            pipeline.finalise()

        # close statistics
        self._finalise_statistics()

    def _generate_ray_templates(self):

        # split spectral bins across rays - non-integer division is handled by
        # rounding up or down the non-integer boundaries between the ray ranges,
        # this means that some rays will have more samples than others
        current = 0
        start = 0
        ranges = []
        while start < self.spectral_samples:
            current += self.spectral_samples / self.spectral_rays
            end = round(current)
            ranges.append((start, end))
            start = end

        # build template rays
        rays = []
        delta_wavelength = (self.max_wavelength - self.min_wavelength) / self.spectral_samples
        for start, end in ranges:
            rays.append(
                Ray(min_wavelength=self.min_wavelength + delta_wavelength * start,
                    max_wavelength=self.min_wavelength + delta_wavelength * end,
                    num_samples=end - start,
                    extinction_prob=self.ray_extinction_prob,
                    min_depth=self.ray_min_depth,
                    max_depth=self.ray_max_depth,
                    importance_sampling=self.ray_importance_sampling,
                    important_path_weight=self.ray_important_path_weight)
            )

        return rays

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
        returns a custom tuple of data tied to details of that pipeline.
        - All results packaged together with pixel ID and returned to consumer.
        :return:
        """

        # obtain reference to world
        world = self.root

        # generate rays
        rays = self._generate_rays(pixel_id, ray_template)  # todo: ray_template -> channel_config

        pixel_processors = [pipeline.pixel_processor(channel) for pipeline in self.pipelines]

        # initialise ray statistics
        ray_count = 0

        # obtain pixel etendue to convert spectral radiance to spectral power
        etendue = self._pixel_etendue(pixel_id)

        # launch rays and accumulate spectral samples
        for ray, projection_weight in rays:

            # convert ray from local space to world space
            ray.origin = ray.origin.transform(self.to_root())
            ray.direction = ray.direction.transform(self.to_root())

            # sample, apply projection weight and convert to power
            spectrum = ray.trace(world)
            spectrum.samples *= projection_weight * etendue

            for processor in pixel_processors:
                processor.add_sample(spectrum)

            # accumulate statistics
            ray_count += ray.ray_count

        # acquire results from pixel processors
        results = [processor.pack_results() for processor in pixel_processors]

        return pixel_id, results, ray_count

    def _generate_rays(self, pixel_id, ray_template):
        """
        Generate a list of Rays that sample over the etendue of the pixel.

        This is a virtual method to be implemented by derived classes.

        Runs during the observe() loop to generate the rays. Allows observers to customise how they launch rays.

        This method must return a list of tuples, with each tuple containing
        a Ray object and a corresponding projected area weight (direction cosine).

        If the projected area weight is not required (due to the ray sampling
        algorithm taking the weighting into account in the distribution e.g.
        cosine weighted) then the weight should be set to 1.0.

        :param tuple pixel_id:
        :param Ray ray_template: A Ray object from which spectral settings can be propagated.
        :return list Rays: A list of tuples.
        """

        raise NotImplementedError("To be defined in subclass.")

    def _pixel_etendue(self, pixel_id):
        """

        :param pixel_id:
        :return:
        """

        raise NotImplementedError("To be defined in subclass.")

    ###################
    # CONSUMER THREAD #
    ###################

    def _update_state(self, packed_result, channel):
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

        for result, pipeline in zip(results, self.pipelines):
            pipeline.update(pixel_id, result, channel)

        # update users
        self._update_statistics(ray_count)

    def _initialise_statistics(self, tasks):
        """
        Initialise statistics.
        """

        self._stats_ray_count = 0
        self._stats_total_rays = 0
        self._stats_start_time = time()
        self._stats_progress_timer = time()
        self._stats_total_tasks = len(tasks) * self.spectral_rays
        self._stats_completed_tasks = 0

    def _update_statistics(self, sample_ray_count):
        """
        Display progress statistics.
        """

        self._stats_completed_tasks += 1
        self._stats_ray_count += sample_ray_count
        self._stats_total_rays += sample_ray_count

        if (time() - self._stats_progress_timer) > 1.0:

            current_time = time() - self._stats_start_time
            completion = 100 * self._stats_completed_tasks / self._stats_total_tasks
            print("Render time: {:0.3f}s ({:0.2f}% complete, {:0.1f}k rays)".format(
                current_time, completion, self._stats_ray_count / 1000))

            self._stats_ray_count = 0
            self._stats_progress_timer = time()

    def _finalise_statistics(self):
        """
        Final statistics output.
        """

        elapsed_time = time() - self._stats_start_time
        mean_rays_per_sec = self._stats_total_rays / elapsed_time
        print("Render complete - time elapsed {:0.3f}s - {:0.1f}k rays/s".format(
            elapsed_time, mean_rays_per_sec / 1000))

