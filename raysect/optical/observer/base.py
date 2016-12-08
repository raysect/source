# Copyright (c) 2014-2016, Dr Alex Meakins, Raysect Project
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
from raysect.optical.ray import Ray
from raysect.optical.scenegraph.world import World
from raysect.optical.scenegraph import Observer
from time import time

# TODO: cythonise me!
# TODO: convert fragment tuple to a cdef class - SpectralFragment?
# TODO: also... create a SpectralConfig to hold the commonly used min, max lambda and num_samples? This would tidy up a lot of material code too...
# TODO: relabel SpectralFragment as SpectralSlice


# TODO - raise not implemented exceptions
class _FrameSamplerBase:

    def generate_tasks(self, pixels):
        pass


# TODO - raise not implemented exceptions
class _PipelineBase:
    """
    base class defining the core interfaces to define an image processing pipeline
    """

    def initialise(self, pixel_config, spectral_fragments):
        """
        setup internal buffers (e.g. frames)
        reset internal statistics as appropriate
        etc..

        :return:
        """
        pass

    def pixel_processor(self, fragment):
        pass

    def update(self, pixel, packed_result, fragment):
        pass

    def finalise(self):
        pass


class _ObserverBase(Observer):
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

    def __init__(self, render_engine=None, parent=None,
                 transform=None, name=None, pixel_samples=None, spectral_rays=None, spectral_samples=None,
                 min_wavelength=None, max_wavelength=None, ray_extinction_prob=None, ray_min_depth=None,
                 ray_max_depth=None, ray_importance_sampling=None, ray_important_path_weight=None):

        super().__init__(parent, transform, name)

        self.pixel_samples = pixel_samples or 100
        self.render_engine = render_engine or MulticoreEngine()

        # preset internal values to satisfy property dependencies
        self._min_wavelength = 0
        self._ray_min_depth = 0

        # ray configuration (order matters due to property dependencies)
        self.spectral_samples = spectral_samples or 15
        self.spectral_rays = spectral_rays or 1
        self.max_wavelength = max_wavelength or 740.0
        self.min_wavelength = min_wavelength or 375.0
        self.ray_extinction_prob = ray_extinction_prob or 0.01
        self.ray_max_depth = ray_max_depth or 500
        self.ray_min_depth = ray_min_depth or 3
        self.ray_importance_sampling = ray_importance_sampling or True
        self.ray_important_path_weight = ray_important_path_weight or 0.2

    @property
    def pixel_samples(self):
        return self._pixel_samples

    @pixel_samples.setter
    def pixel_samples(self, value):
        if value <= 0:
            raise ValueError("The number of pixel samples must be greater than 0.")
        self._pixel_samples = value

    @property
    def spectral_samples(self):
        return self._spectral_samples

    @spectral_samples.setter
    def spectral_samples(self, value):
        if value <= 0:
            raise ValueError("The number of spectral samples must be greater than 0.")
        self._spectral_samples = value

    @property
    def spectral_rays(self):
        return self._spectral_rays

    @spectral_rays.setter
    def spectral_rays(self, value):
        if not 0 < value <= self.spectral_samples:
            raise ValueError("The number of spectral rays must be in the range [1, spectral_samples].")
        self._spectral_rays = value

    @property
    def min_wavelength(self):
        return self._min_wavelength

    @min_wavelength.setter
    def min_wavelength(self, value):
        if value <= 0:
            raise ValueError("The minimum wavelength must be greater than 0.")
        if value >= self._max_wavelength:
            raise ValueError("The minimum wavelength must be less than the maximum wavelength.")
        self._min_wavelength = value

    @property
    def max_wavelength(self):
        return self._max_wavelength

    @max_wavelength.setter
    def max_wavelength(self, value):
        if value <= 0:
            raise ValueError("The maximum wavelength must be greater than 0.")
        if value <= self._min_wavelength:
            raise ValueError("The maximum wavelength must be greater than the minimum wavelength.")
        self._max_wavelength = value

    @property
    def ray_extinction_prob(self):
        return self._ray_extinction_prob

    @ray_extinction_prob.setter
    def ray_extinction_prob(self, value):
        if not 0 <= value <= 1:
            raise ValueError("The ray extinction probability must be in the range [0, 1].")
        self._ray_extinction_prob = value

    @property
    def ray_min_depth(self):
        return self._ray_min_depth

    @ray_min_depth.setter
    def ray_min_depth(self, value):
        if value < 0:
            raise ValueError("Minimum ray depth cannot be less than 0.")
        if value > self._max_ray_depth:
            raise ValueError("Minimum ray depth cannot be greater than maximum ray depth.")
        self._ray_min_depth = value

    @property
    def ray_max_depth(self):
        return self._ray_max_depth

    @ray_max_depth.setter
    def ray_max_depth(self, value):
        if value < 0:
            raise ValueError("Maximum ray depth cannot be less than 0.")
        if value < self._min_ray_depth:
            raise ValueError("Maximum ray depth cannot be less than minimum ray depth.")
        self._ray_max_depth = value

    @property
    def ray_important_path_weight(self):
        return self._ray_important_path_weight

    @ray_important_path_weight.setter
    def ray_important_path_weight(self, value):
        if not 0 <= value <= 1:
            raise ValueError("The ray important path weight must be in the range [0, 1].")
        self._ray_important_path_weight = value

    def observe(self):
        """ Ask this Camera to Observe its world. """

        # must be connected to a world node to be able to perform a ray trace
        if not isinstance(self.root, World):
            raise TypeError("Observer is not connected to a scene graph containing a World object.")

        # generate spectral fragment configuration
        fragments = self._generate_spectral_fragments()

        # initialise pipelines for rendering
        for pipeline in self.pipelines:
            pipeline.initialise(self._pixel_config, fragments)

        tasks = self.frame_sampler.generate_tasks(self._pixel_config)

        # initialise statistics with total task count
        self._initialise_statistics(tasks)

        # render each spectral fragment
        for fragment in fragments:

            self.render_engine.run(
                tasks, self._render_pixel, self._update_state,
                render_args=(fragment, ),
                update_args=(fragment, )
            )

        # close pipelines
        for pipeline in self.pipelines:
            pipeline.finalise()

        # close statistics
        self._finalise_statistics()

    def _generate_spectral_fragments(self):
        """
        Sub-divides the spectral range into smaller wavelength fragments.

        In dispersive rendering, where multiple rays are launched across the full spectral range, each ray samples a small
        portion of the spectrum. A fragment defines a sub region of the spectral range that is sampled
        by launching a ray.

        A spectral fragment is a tuple containing the following information:

            fragment = (
                id,             # numerical index of fragment
                ray_template,   # ray template configured for fragment
                offset,         # offset into full spectral sample array
                num_samples,    # number of samples covered by the fragment
                min_wavelength, # lower wavelength bound of fragment
                max_wavelength, # upper wavelength bound of fragment
            )

        :return: A list of fragment tuples.
        """

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

        # build fragments
        fragments = []
        delta_wavelength = (self.max_wavelength - self.min_wavelength) / self.spectral_samples
        for id, range in enumerate(ranges):

            start, end = range

            # calculate fragment parameters
            min_wavelength = self.min_wavelength + delta_wavelength * start
            max_wavelength = self.min_wavelength + delta_wavelength * end
            num_samples = end - start
            offset = start
            ray_template = Ray(
                min_wavelength=min_wavelength,
                max_wavelength=max_wavelength,
                num_samples=num_samples,
                extinction_prob=self.ray_extinction_prob,
                min_depth=self.ray_min_depth,
                max_depth=self.ray_max_depth,
                importance_sampling=self.ray_importance_sampling,
                important_path_weight=self.ray_important_path_weight
            )

            fragment = (id, ray_template, offset, num_samples, min_wavelength, max_wavelength)
            fragments.append(fragment)

        return fragments

    #################
    # WORKER THREAD #
    #################

    def _render_pixel(self, pixel_id, fragment):
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
        rays = self._generate_rays(pixel_id, fragment)

        pixel_processors = [pipeline.pixel_processor(fragment) for pipeline in self.pipelines]

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

    def _generate_rays(self, pixel_id, fragment):
        """
        Generate a list of Rays that sample over the etendue of the pixel.

        This is a virtual method to be implemented by derived classes.

        Runs during the observe() loop to generate the rays. Allows observers to customise how they launch rays.

        This method must return a list of tuples, with each tuple containing
        a Ray object and a corresponding weighting, typically the projected area/direction cosine.

        If the projected area weight is not required (due to the ray sampling
        algorithm taking the weighting into account in the distribution e.g.
        cosine weighted) then the weight should be set to 1.0.

        :param tuple pixel_id: The pixel id.
        :param tuple fragment: The spectral fragment configuration.
        :return list: A list of tuples of (ray, weight)
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

    def _update_state(self, packed_result, fragment):
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
            pipeline.update(pixel_id, result, fragment)

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

