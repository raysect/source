# cython: language_level=3

# Copyright (c) 2014-2017, Dr Alex Meakins, Raysect Project
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

from time import time
from raysect.core.workflow import RenderEngine, MulticoreEngine

cimport cython
from raysect.optical cimport World, Spectrum
from raysect.optical.observer.base.sampler cimport FrameSampler1D, FrameSampler2D
from raysect.optical.observer.base.pipeline cimport Pipeline0D, Pipeline1D, Pipeline2D
from raysect.optical.observer.base.processor cimport PixelProcessor
from raysect.optical.observer.base.slice cimport SpectralSlice


cdef class _ObserverBase(Observer):
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

    def __init__(self, parent=None, transform=None, name=None, render_engine=None, spectral_rays=None, spectral_bins=None,
                 min_wavelength=None, max_wavelength=None, ray_extinction_prob=None, ray_extinction_min_depth=None,
                 ray_max_depth=None, ray_importance_sampling=None, ray_important_path_weight=None):

        super().__init__(parent, transform, name)

        self.render_engine = render_engine or MulticoreEngine()

        # preset internal values to satisfy property dependencies
        self._spectral_rays = 1
        self._min_wavelength = 0
        self._ray_extinction_min_depth = 0

        # ray configuration (order matters due to property dependencies)
        self.spectral_bins = spectral_bins or 15
        self.spectral_rays = spectral_rays or 1
        self.max_wavelength = max_wavelength or 740.0
        self.min_wavelength = min_wavelength or 375.0
        self.ray_extinction_prob = ray_extinction_prob or 0.01
        self.ray_max_depth = ray_max_depth or 500
        self.ray_extinction_min_depth = ray_extinction_min_depth or 3
        self.ray_importance_sampling = ray_importance_sampling or True
        self.ray_important_path_weight = ray_important_path_weight or 0.2

        # flag indicating if the frame sampler is not supplying any tasks (in which case the rendering process is over)
        self.render_complete = False

    @property
    def spectral_bins(self):
        return self._spectral_bins

    @spectral_bins.setter
    def spectral_bins(self, value):
        if value <= 0:
            raise ValueError("The number of spectral bins must be greater than 0.")
        if value < self.spectral_rays:
            raise ValueError("The number of spectral bins cannot be less than the number of spectral rays (currently {}).".format(self.spectral_rays))
        self._spectral_bins = value

    @property
    def spectral_rays(self):
        return self._spectral_rays

    @spectral_rays.setter
    def spectral_rays(self, value):
        if not 0 < value <= self.spectral_bins:
            raise ValueError("The number of spectral rays cannot be greater than the number of spectral bins (currently {}).".format(self.spectral_bins))
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
    def ray_extinction_min_depth(self):
        return self._ray_extinction_min_depth

    @ray_extinction_min_depth.setter
    def ray_extinction_min_depth(self, value):
        if value < 0:
            raise ValueError("Minimum ray depth cannot be less than 0.")
        self._ray_extinction_min_depth = value

    @property
    def ray_max_depth(self):
        return self._ray_max_depth

    @ray_max_depth.setter
    def ray_max_depth(self, value):
        if value < 0:
            raise ValueError("Maximum ray depth cannot be less than 0.")
        self._ray_max_depth = value

    @property
    def ray_important_path_weight(self):
        return self._ray_important_path_weight

    @ray_important_path_weight.setter
    def ray_important_path_weight(self, value):
        if not 0 <= value <= 1:
            raise ValueError("The ray important path weight must be in the range [0, 1].")
        self._ray_important_path_weight = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef observe(self):
        """ Ask this Camera to Observe its world. """

        cdef:
            list slices, templates, tasks
            int slice_id
            Ray template

        self.render_complete = False

        # must be connected to a world node to be able to perform a ray trace
        if not isinstance(self.root, World):
            raise TypeError("Observer is not connected to a scene graph containing a World object.")

        # generate spectral configuration and ray templates
        slices = self._slice_spectrum()
        templates = self._generate_templates(slices)

        # initialise pipelines for rendering
        self._initialise_pipelines(self._min_wavelength, self._max_wavelength, self._spectral_bins, slices)

        # request render tasks and escape early if there is no work to perform
        # if there is no work to perform then the render is considered "complete"
        tasks = self._generate_tasks()
        if not tasks:
            self.render_complete = True
            print("Render complete - No render tasks were generated.")
            return

        # initialise statistics with total task count
        self._initialise_statistics(tasks)

        # render each spectral slice
        for slice_id, template in enumerate(templates):

            self.render_engine.run(
                tasks, self._render_pixel, self._update_state,
                render_args=(slice_id, template),
                update_args=(slice_id, )
            )

        # close pipelines and statistics
        self._finalise_pipelines()
        self._finalise_statistics()

    cpdef list _slice_spectrum(self):
        """
        Sub-divides the spectral range into smaller wavelength slices.

        In dispersive rendering, where multiple rays are launched across the full spectral range, each ray samples a small
        portion of the spectrum. A slice defines a sub region of the spectral range that is sampled
        by launching a ray.

        :return: A list of SpectralSlice objects.
        """

        cdef:
            double current
            int start, end
            list ranges

        # split spectral bins across rays - non-integer division is handled by
        # rounding up or down the non-integer boundaries between the ray ranges,
        # this means that some rays will have more samples than others
        current = 0
        start = 0
        ranges = []
        while start < self._spectral_bins:
            current += self._spectral_bins / self._spectral_rays
            end = round(current)
            ranges.append((start, end))
            start = end

        # build slices
        return [SpectralSlice(self._min_wavelength, self._max_wavelength, self._spectral_bins, end - start, start) for start, end in ranges]

    cpdef list _generate_templates(self, list slices):

        return [
            Ray(
                min_wavelength=slice.min_wavelength,
                max_wavelength=slice.max_wavelength,
                bins=slice.bins,
                extinction_prob=self.ray_extinction_prob,
                extinction_min_depth=self.ray_extinction_min_depth,
                max_depth=self.ray_max_depth,
                importance_sampling=self.ray_importance_sampling,
                important_path_weight=self.ray_important_path_weight
            ) for slice in slices
        ]

    #################
    # WORKER THREAD #
    #################

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object _render_pixel(self, tuple task, int slice_id, Ray template):
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

        cdef:
            World world
            list rays, pixel_processors
            PixelProcessor processor
            int ray_count
            double etendue, projection_weight
            Ray ray
            Spectrum spectrum
            list results

        # obtain reference to world
        world = self.root

        # generate rays and obtain pixel processors from each pipeline
        rays = self._obtain_rays(task, template)
        pixel_processors = self._obtain_pixel_processors(task, slice_id)

        # initialise ray statistics
        ray_count = 0

        # obtain pixel etendue to convert spectral radiance to spectral power
        etendue = self._obtain_etendue(task)

        # launch rays and accumulate spectral samples
        for ray, projection_weight in rays:

            # convert ray from local space to world space
            ray.origin = ray.origin.transform(self.to_root())
            ray.direction = ray.direction.transform(self.to_root())

            # sample, apply projection weight
            spectrum = ray.trace(world)
            spectrum.mul_scalar(projection_weight)

            for processor in pixel_processors:
                processor.add_sample(spectrum, etendue)

            # accumulate statistics
            ray_count += ray.ray_count

        # acquire results from pixel processors
        results = [processor.pack_results() for processor in pixel_processors]

        return task, results, ray_count

    ###################
    # CONSUMER THREAD #
    ###################

    cpdef object _update_state(self, tuple packed_result, int slice_id):
        """
        - unpack task configuration and pipeline result tuples.
        - pass results to each pipeline to update pipelines internal state
        - print workflow statistics and any statistics for each pipeline.
        - display visual imagery for each pipeline as required
        - save state for each pipeline as required.

        :return:
        """

        cdef:
            tuple task
            list results
            int ray_count

        # unpack worker results
        task, results, ray_count = packed_result

        # update pipelines and statistics
        self._update_pipelines(task, results, slice_id)
        self._update_statistics(ray_count)

    cpdef list _generate_tasks(self):
        raise NotImplementedError("To be defined in subclass.")

    cpdef list _obtain_pixel_processors(self, tuple task, int slice_id):
        raise NotImplementedError("To be defined in subclass.")

    cpdef object _initialise_pipelines(self, double min_wavelength, double max_wavelength, int spectral_bins, list slices):
        raise NotImplementedError("To be defined in subclass.")

    cpdef object _update_pipelines(self, tuple task, list results, int slice_id):
        raise NotImplementedError("To be defined in subclass.")

    cpdef object _finalise_pipelines(self):
        raise NotImplementedError("To be defined in subclass.")

    cpdef object _initialise_statistics(self, list tasks):
        """
        Initialise statistics.
        """

        self._stats_ray_count = 0
        self._stats_total_rays = 0
        self._stats_start_time = time()
        self._stats_progress_timer = time()
        self._stats_total_tasks = len(tasks) * self.spectral_rays
        self._stats_completed_tasks = 0

    cpdef object _update_statistics(self, int sample_ray_count):
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

    cpdef object _finalise_statistics(self):
        """
        Final statistics output.
        """

        elapsed_time = time() - self._stats_start_time
        mean_rays_per_sec = self._stats_total_rays / elapsed_time
        print("Render complete - time elapsed {:0.3f}s - {:0.1f}k rays/s".format(
            elapsed_time, mean_rays_per_sec / 1000))

    cpdef list _obtain_rays(self, tuple task, Ray template):
        """
        Returns a list of Rays that sample over the etendue of the pixel.

        This is a virtual method to be implemented by derived classes.

        Runs during the observe() loop to generate the rays. Allows observers
        to customise how they launch rays.

        This method must return a list of tuples, with each tuple containing
        a Ray object and a corresponding weighting, typically the projected
        area/direction cosine.

        If the projected area weight is not required (due to the ray sampling
        algorithm taking the weighting into account in the distribution e.g.
        cosine weighted) then the weight should be set to 1.0.

        :param tuple task: The render task configuration.
        :param Ray template: The template ray from which all rays should be generated.
        :return list: A list of tuples of (ray, weight)
        """

        raise NotImplementedError("To be defined in subclass.")

    cpdef double _obtain_etendue(self, tuple task):
        """

        :param pixel_id:
        :return:
        """

        raise NotImplementedError("To be defined in subclass.")


cdef class Observer0D(_ObserverBase):
    """

    """

    def __init__(self, pipelines, parent=None, transform=None, name=None,
                 render_engine=None, pixel_samples=None, samples_per_task=None, spectral_rays=None, spectral_bins=None,
                 min_wavelength=None, max_wavelength=None, ray_extinction_prob=None, ray_extinction_min_depth=None,
                 ray_max_depth=None, ray_importance_sampling=None, ray_important_path_weight=None):

        self.pixel_samples = pixel_samples or 1000
        self.samples_per_task = samples_per_task or 250
        self.pipelines = pipelines

        super().__init__(
            parent, transform, name, render_engine, spectral_rays, spectral_bins,
            min_wavelength, max_wavelength, ray_extinction_prob, ray_extinction_min_depth,
            ray_max_depth, ray_importance_sampling, ray_important_path_weight
        )

    @property
    def pixel_samples(self):
        return self._pixel_samples

    @pixel_samples.setter
    def pixel_samples(self, value):
        if value <= 0:
            raise ValueError("The number of pixel samples must be greater than 0.")
        self._pixel_samples = value

    @property
    def samples_per_task(self):
        return self._samples_per_task

    @samples_per_task.setter
    def samples_per_task(self, value):
        if value <= 0:
            raise ValueError("The number of pixel samples per task must be greater than 0.")
        self._samples_per_task = value

    @property
    def pipelines(self):
        return self._pipelines

    @pipelines.setter
    def pipelines(self, value):
        pipelines = tuple(value)
        if len(pipelines) < 1:
            raise ValueError("At least one processing pipeline must be provided.")
        for pipeline in pipelines:
            if not isinstance(pipeline, Pipeline0D):
                raise TypeError("Processing pipelines for a 0d observer must be a subclass of Pipeline0D.")
        self._pipelines = pipelines

    cpdef list _generate_tasks(self):

        cdef:
            list tasks
            int samples

        # subdivide pixel samples into discrete chunks to enable work to be processed in parallel
        tasks = []
        samples = self._pixel_samples
        while samples > 0:
            if samples >= self._samples_per_task:
                tasks.append((self._samples_per_task, ))
            else:
                tasks.append((samples, ))
            samples -= self._samples_per_task
        return tasks

    cpdef list _obtain_pixel_processors(self, tuple task, int slice_id):
        return [pipeline.pixel_processor(slice_id) for pipeline in self._pipelines]

    cpdef object _initialise_pipelines(self, double min_wavelength, double max_wavelength, int spectral_bins, list slices):
        cdef Pipeline0D pipeline
        for pipeline in self._pipelines:
            pipeline.initialise(self._min_wavelength, self._max_wavelength, self._spectral_bins, slices)

    cpdef object _update_pipelines(self, tuple task, list results, int slice_id):

        cdef:
            int samples
            tuple result
            Pipeline0D pipeline

        samples, = task
        for result, pipeline in zip(results, self._pipelines):
            pipeline.update(slice_id, result, samples)

    cpdef object _finalise_pipelines(self):
        cdef Pipeline0D pipeline
        for pipeline in self._pipelines:
            pipeline.finalise()

    cpdef list _obtain_rays(self, tuple task, Ray template):
        cdef int samples
        samples, = task
        return self._generate_rays(template, samples)

    cpdef double _obtain_etendue(self, tuple task):
        return self._pixel_etendue()

    cpdef list _generate_rays(self, Ray template, int ray_count):
        """
        Generate a list of Rays that sample over the etendue of the pixel.

        This is a virtual method to be implemented by derived classes.

        Runs during the observe() loop to generate the rays. Allows observers
        to customise how they launch rays.

        This method must return a list of tuples, with each tuple containing
        a Ray object and a corresponding weighting, typically the projected
        area/direction cosine. The number of rays returned must be equal to
        ray_count otherwise pipeline statistics will be incorrectly calculated.

        If the projected area weight is not required (due to the ray sampling
        algorithm taking the weighting into account in the distribution e.g.
        cosine weighted) then the weight should be set to 1.0.

        :param Ray template: The template ray from which all rays should be generated.
        :param int ray_count: The number of rays to be generated.
        :return list: A list of tuples of (ray, weight)
        """

        raise NotImplementedError("To be defined in subclass.")

    cpdef double _pixel_etendue(self):
        """

        :return:
        """

        raise NotImplementedError("To be defined in subclass.")


cdef class Observer1D(_ObserverBase):
    """

    """

    def __init__(self, pixels, frame_sampler, pipelines, parent=None, transform=None, name=None,
                 render_engine=None, pixel_samples=None, spectral_rays=None, spectral_bins=None,
                 min_wavelength=None, max_wavelength=None, ray_extinction_prob=None, ray_extinction_min_depth=None,
                 ray_max_depth=None, ray_importance_sampling=None, ray_important_path_weight=None):

        self.pixel_samples = pixel_samples or 100
        self.pixels = pixels
        self.frame_sampler = frame_sampler
        self.pipelines = pipelines

        super().__init__(
            parent, transform, name, render_engine, spectral_rays, spectral_bins,
            min_wavelength, max_wavelength, ray_extinction_prob, ray_extinction_min_depth,
            ray_max_depth, ray_importance_sampling, ray_important_path_weight
        )

    @property
    def pixel_samples(self):
        return self._pixel_samples

    @pixel_samples.setter
    def pixel_samples(self, value):
        if value <= 0:
            raise ValueError("The number of pixel samples must be greater than 0.")
        self._pixel_samples = value

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, value):
        if value <= 0:
            raise ValueError("Number of pixels must be greater than 0.")
        self._pixels = value

    @property
    def frame_sampler(self):
        return self._frame_sampler

    @frame_sampler.setter
    def frame_sampler(self, value):
        if not isinstance(value, FrameSampler1D):
            raise TypeError("The frame sampler for a 1d observer must be a subclass of FrameSampler1D.")
        self._frame_sampler = value

    @property
    def pipelines(self):
        return self._pipelines

    @pipelines.setter
    def pipelines(self, value):
        pipelines = tuple(value)
        if len(pipelines) < 1:
            raise ValueError("At least one processing pipeline must be provided.")
        for pipeline in pipelines:
            if not isinstance(pipeline, Pipeline1D):
                raise TypeError("Processing pipelines for a 1d observer must be a subclass of Pipeline1D.")
        self._pipelines = pipelines

    cpdef list _generate_tasks(self):
        return self._frame_sampler.generate_tasks(self._pixels)

    cpdef list _obtain_pixel_processors(self, tuple task, int slice_id):

        cdef:
            int pixel
            Pipeline1D pipeline

        pixel, = task
        return [pipeline.pixel_processor(pixel, slice_id) for pipeline in self._pipelines]

    cpdef object _initialise_pipelines(self, double min_wavelength, double max_wavelength, int spectral_bins, list slices):
        cdef Pipeline1D pipeline
        for pipeline in self._pipelines:
            pipeline.initialise(self._pixels, self._pixel_samples, self._min_wavelength, self._max_wavelength, self._spectral_bins, slices)

    cpdef object _update_pipelines(self, tuple task, list results, int slice_id):

        cdef:
            int pixel
            tuple result
            Pipeline1D pipeline

        pixel, = task
        for result, pipeline in zip(results, self._pipelines):
            pipeline.update(pixel, slice_id, result)

    cpdef object _finalise_pipelines(self):
        cdef Pipeline1D pipeline
        for pipeline in self._pipelines:
            pipeline.finalise()

    cpdef list _obtain_rays(self, tuple task, Ray template):
        cdef int pixel
        pixel, = task
        return self._generate_rays(pixel, template, self._pixel_samples)

    cpdef double _obtain_etendue(self, tuple task):
        cdef int pixel
        pixel, = task
        return self._pixel_etendue(pixel)

    cpdef list _generate_rays(self, int pixel, Ray template, int ray_count):
        """
        Generate a list of Rays that sample over the etendue of the pixel.

        This is a virtual method to be implemented by derived classes.

        Runs during the observe() loop to generate the rays. Allows observers
        to customise how they launch rays.

        This method must return a list of tuples, with each tuple containing
        a Ray object and a corresponding weighting, typically the projected
        area/direction cosine. The number of rays returned must be equal to
        ray_count otherwise pipeline statistics will be incorrectly calculated.

        If the projected area weight is not required (due to the ray sampling
        algorithm taking the weighting into account in the distribution e.g.
        cosine weighted) then the weight should be set to 1.0.

        :param int pixel: Pixel index.
        :param Ray template: The template ray from which all rays should be generated.
        :param int ray_count: The number of rays to be generated.
        :return list: A list of tuples of (ray, weight)
        """

        raise NotImplementedError("To be defined in subclass.")

    cpdef double _pixel_etendue(self, int pixel):
        """

        :param int pixel: Pixel index.
        :return:
        """

        raise NotImplementedError("To be defined in subclass.")


cdef class Observer2D(_ObserverBase):
    """

    """

    def __init__(self, pixels, frame_sampler, pipelines, parent=None, transform=None, name=None,
                 render_engine=None, pixel_samples=None, spectral_rays=None, spectral_bins=None,
                 min_wavelength=None, max_wavelength=None, ray_extinction_prob=None, ray_extinction_min_depth=None,
                 ray_max_depth=None, ray_importance_sampling=None, ray_important_path_weight=None):

        self.pixel_samples = pixel_samples or 100
        self.pixels = pixels
        self.frame_sampler = frame_sampler
        self.pipelines = pipelines

        super().__init__(
            parent, transform, name, render_engine, spectral_rays, spectral_bins,
            min_wavelength, max_wavelength, ray_extinction_prob, ray_extinction_min_depth,
            ray_max_depth, ray_importance_sampling, ray_important_path_weight
        )

    @property
    def pixel_samples(self):
        return self._pixel_samples

    @pixel_samples.setter
    def pixel_samples(self, value):
        if value <= 0:
            raise ValueError("The number of pixel samples must be greater than 0.")
        self._pixel_samples = value

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, value):
        pixels = tuple(value)
        if len(pixels) != 2:
            raise ValueError("Pixels must be a 2 element tuple defining the x and y resolution.")
        x, y = pixels
        if x <= 0:
            raise ValueError("Number of x pixels must be greater than 0.")
        if y <= 0:
            raise ValueError("Number of y pixels must be greater than 0.")
        self._pixels = pixels

    @property
    def frame_sampler(self):
        return self._frame_sampler

    @frame_sampler.setter
    def frame_sampler(self, value):
        if not isinstance(value, FrameSampler2D):
            raise TypeError("The frame sampler for a 2d observer must be a subclass of FrameSampler2D.")
        self._frame_sampler = value

    @property
    def pipelines(self):
        return self._pipelines

    @pipelines.setter
    def pipelines(self, value):
        pipelines = tuple(value)
        if len(pipelines) < 1:
            raise ValueError("At least one processing pipeline must be provided.")
        for pipeline in pipelines:
            if not isinstance(pipeline, Pipeline2D):
                raise TypeError("Processing pipelines for a 2d observer must be a subclass of Pipeline2D.")
        self._pipelines = pipelines

    cpdef list _generate_tasks(self):
        return self._frame_sampler.generate_tasks(self._pixels)

    cpdef list _obtain_pixel_processors(self, tuple task, int slice_id):

        cdef:
            int x, y
            Pipeline2D pipeline

        x, y = task
        return [pipeline.pixel_processor(x, y, slice_id) for pipeline in self._pipelines]

    cpdef object _initialise_pipelines(self, double min_wavelength, double max_wavelength, int spectral_bins, list slices):
        cdef Pipeline2D pipeline
        for pipeline in self._pipelines:
            pipeline.initialise(self._pixels, self._pixel_samples, self._min_wavelength, self._max_wavelength, self._spectral_bins, slices)

    cpdef object _update_pipelines(self, tuple task, list results, int slice_id):

        cdef:
            int x, y
            tuple result
            Pipeline2D pipeline

        x, y = task
        for result, pipeline in zip(results, self._pipelines):
            pipeline.update(x, y, slice_id, result)

    cpdef object _finalise_pipelines(self):
        cdef Pipeline2D pipeline
        for pipeline in self._pipelines:
            pipeline.finalise()

    cpdef list _obtain_rays(self, tuple task, Ray template):
        cdef int x, y
        x, y = task
        return self._generate_rays(x, y, template, self._pixel_samples)

    cpdef double _obtain_etendue(self, tuple task):
        cdef int x, y
        x, y = task
        return self._pixel_etendue(x, y)

    cpdef list _generate_rays(self, int x, int y, Ray template, int ray_count):
        """
        Generate a list of Rays that sample over the etendue of the pixel.

        This is a virtual method to be implemented by derived classes.

        Runs during the observe() loop to generate the rays. Allows observers
        to customise how they launch rays.

        This method must return a list of tuples, with each tuple containing
        a Ray object and a corresponding weighting, typically the projected
        area/direction cosine. The number of rays returned must be equal to
        ray_count otherwise pipeline statistics will be incorrectly calculated.

        If the projected area weight is not required (due to the ray sampling
        algorithm taking the weighting into account in the distribution e.g.
        cosine weighted) then the weight should be set to 1.0.

        :param int x: Pixel x index.
        :param int y: Pixel y index.
        :param Ray template: The template ray from which all rays should be generated.
        :param int ray_count: The number of rays to be generated.
        :return list: A list of tuples of (ray, weight)
        """

        raise NotImplementedError("To be defined in subclass.")

    cpdef double _pixel_etendue(self, int x, int y):
        """

        :param int x: Pixel x index.
        :param int y: Pixel y index.
        :return:
        """

        raise NotImplementedError("To be defined in subclass.")