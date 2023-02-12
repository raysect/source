# Copyright (c) 2014-2023, Dr Alex Meakins, Raysect Project
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
import matplotlib.pyplot as plt
import numpy as np

cimport cython
cimport numpy as np
from raysect.core.math.cython cimport clamp
from raysect.optical.spectralfunction cimport SpectralFunction
from raysect.optical.observer.base cimport PixelProcessor, Pipeline2D
from raysect.core.math cimport StatsArray2D
from raysect.optical.observer.pipeline.mono.power cimport PowerPixelProcessor
from libc.math cimport pow


_DEFAULT_PIPELINE_NAME = "Bayer Pipeline"
_DISPLAY_DPI = 100
_DISPLAY_SIZE = (512 / _DISPLAY_DPI, 512 / _DISPLAY_DPI)


cdef class BayerPipeline2D(Pipeline2D):
    """
    A 2D pipeline simulating a Bayer filter.

    Many commercial cameras use a Bayer filter for converting measured spectra into
    a 2D image of RGB values. The 2D sensor pixel array is covered with a mosaic of
    alternating red, green and blue filters. Thus each pixel in the array is only
    responsive to one of the colour filters simulating the response of the human eye.
    The final image is represented by a 2D grid of only red, green and blue values. The
    eye interpolates these values to create other colours. See
    `Wikipedia <https://en.wikipedia.org/wiki/Bayer_filter>`_ for more information.

    :param SpectralFunction red_filter: The spectral function representing the red pixel filter.
    :param SpectralFunction green_filter: The spectral function representing the green pixel filter.
    :param SpectralFunction blue_filter: The spectral function representing the blue pixel filter.
    :param bool display_progress: Toggles the display of live render progress (default=True).
    :param float display_update_time: Time in seconds between preview display
      updates (default=15 seconds).
    :param bool accumulate: Whether to accumulate samples with subsequent calls
      to observe() (default=True).
    :param bool display_auto_exposure: Toggles the use of automatic exposure of
      final images (default=True).
    :param float display_black_point: Lower clamp point for pixel to appear black
      (default=0.0).
    :param float display_white_point: Upper clamp point for pixel saturation
      (default=1.0).
    :param float display_unsaturated_fraction:  Fraction of pixels that must not
      be saturated. Display values will be scaled to satisfy this value
      (default=1.0).
    :param float display_gamma: Gamma exponent to account for non-linear response of
      display screens (default=2.2).
    :param str name: User friendly name for this pipeline (default="Bayer Pipeline").

    .. code-block:: pycon

        >>> from raysect.optical import InterpolatedSF
        >>> from raysect.optical.observer import BayerPipeline2D
        >>>
        >>> filter_red = InterpolatedSF([100, 650, 660, 670, 680, 800], [0, 0, 1, 1, 0, 0])
        >>> filter_green = InterpolatedSF([100, 530, 540, 550, 560, 800], [0, 0, 1, 1, 0, 0])
        >>> filter_blue = InterpolatedSF([100, 480, 490, 500, 510, 800], [0, 0, 1, 1, 0, 0])
        >>>
        >>> bayer = BayerPipeline2D(filter_red, filter_green, filter_blue,
                                    display_unsaturated_fraction=0.96, name="Bayer Filter")
    """

    def __init__(self, SpectralFunction red_filter, SpectralFunction green_filter,
                 SpectralFunction blue_filter, bint display_progress=True,
                 double display_update_time=15, bint accumulate=True,
                 bint display_auto_exposure=True, double display_black_point=0.0, double display_white_point=1.0,
                 double display_unsaturated_fraction=1.0, display_gamma=2.2, str name=None):

        self.name = name or _DEFAULT_PIPELINE_NAME

        self.red_filter = red_filter
        self.green_filter = green_filter
        self.blue_filter = blue_filter

        # TODO - expose bayer mosaic as a property
        # also support converting 'R', 'G', 'B' strings to 0, 1, 2
        self._bayer_mosaic = (0, 1, 1, 2)

        if display_black_point < 0:
            raise ValueError('Black point cannot be less than zero.')

        if display_white_point < display_black_point:
            display_white_point = display_black_point

        self.display_progress = display_progress
        self.display_update_time = display_update_time
        self.display_persist_figure = True

        self.display_gamma = display_gamma
        self._display_black_point = display_black_point
        self._display_white_point = display_white_point
        self._display_auto_exposure = display_auto_exposure
        self.display_unsaturated_fraction = display_unsaturated_fraction

        self.accumulate = accumulate

        self.frame = None

        self._working_mean = None
        self._working_variance = None
        self._working_touched = None

        self._display_frame = None
        self._display_timer = 0
        self._display_figure = None

        self._processors = None

        self._pixels = None
        self._samples = 0

        self._quiet = False

    def __getstate__(self):

        return (
            self.name,
            self.red_filter,
            self.green_filter,
            self.blue_filter,
            self._bayer_mosaic,
            self.display_progress,
            self.display_update_time,
            self.display_persist_figure,
            self.display_gamma,
            self._display_black_point,
            self._display_white_point,
            self._display_auto_exposure,
            self.display_unsaturated_fraction,
            self.accumulate,
            self.frame
        )

    def __setstate__(self, state):

        (
            self.name,
            self.red_filter,
            self.green_filter,
            self.blue_filter,
            self._bayer_mosaic,
            self.display_progress,
            self.display_update_time,
            self.display_persist_figure,
            self.display_gamma,
            self._display_black_point,
            self._display_white_point,
            self._display_auto_exposure,
            self.display_unsaturated_fraction,
            self.accumulate,
            self.frame
        ) = state

        # initialise internal state
        self._working_mean = None
        self._working_variance = None
        self._working_touched = None
        self._display_frame = None
        self._display_timer = 0
        self._display_figure = None
        self._processors = None
        self._pixels = None
        self._samples = 0
        self._quiet = False

    # must override automatic __reduce__ method generated by cython for the base class
    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @property
    def display_white_point(self):
        """
        Upper clamp point for pixel colour saturation.

        :rtype: float
        """
        return self._display_white_point

    @display_white_point.setter
    def display_white_point(self, value):
        if value < self._display_black_point:
            raise ValueError("White point cannot be less than black point.")
        self._display_auto_exposure = False
        self._display_white_point = value
        self._refresh_display()

    @property
    def display_black_point(self):
        """
        Lower clamp point for pixel to appear black.

        :rtype: float
        """
        return self._display_black_point

    @display_black_point.setter
    def display_black_point(self, value):
        if value < 0:
            raise ValueError('Black point cannot be less than zero.')
        if value > self._display_white_point:
            raise ValueError("Black point cannot be greater than white point.")
        self._display_black_point = value
        self._refresh_display()

    @property
    def display_gamma(self):
        """
        Power law exponent to approximate non-linear human eye response.

        Each pixel value will be raised to power gamma:

        .. math::

            V_{out} = V_{in}^{\\gamma}

        For more information see `Wikipedia <https://en.wikipedia.org/wiki/Gamma_correction>`_.

        :rtype: float
        """
        return self._display_gamma

    @display_gamma.setter
    def display_gamma(self, value):
        if value <= 0.0:
            raise ValueError('Gamma correction value must be greater that 0.')
        self._display_gamma = value
        self._refresh_display()

    @property
    def display_auto_exposure(self):
        """
        Toggles the use of automatic exposure on final image.

        :rtype: bool
        """
        return self._display_auto_exposure

    @display_auto_exposure.setter
    def display_auto_exposure(self, value):
        self._display_auto_exposure = value
        self._refresh_display()

    @property
    def display_unsaturated_fraction(self):
        """
        Fraction of pixels that must not be saturated. Display values will
        be scaled to satisfy this value.

        :rtype: float
        """
        return self._display_unsaturated_fraction

    @display_unsaturated_fraction.setter
    def display_unsaturated_fraction(self, value):
        if not (0 < value <= 1):
            raise ValueError('Auto exposure unsaturated fraction must lie in range (0, 1].')
        self._display_unsaturated_fraction = value
        self._refresh_display()

    @property
    def display_update_time(self):
        """
        Time in seconds between preview display updates.

        :rtype: float
        """
        return self._display_update_time

    @display_update_time.setter
    def display_update_time(self, value):
        if value <= 0:
            raise ValueError('Display update time must be greater than zero seconds.')
        self._display_update_time = value

    cpdef object initialise(self, tuple pixels, int pixel_samples, double min_wavelength, double max_wavelength, int spectral_bins, list spectral_slices, bint quiet):

        nx, ny = pixels
        self._pixels = pixels
        self._samples = pixel_samples

        # create intermediate and final frame-buffers
        if not self.accumulate or self.frame is None or self.frame.shape != (nx, ny):
            self.frame = StatsArray2D(nx, ny)

        self._working_mean = np.zeros((nx, ny))
        self._working_variance = np.zeros((nx, ny))
        self._working_touched = np.zeros((nx, ny), dtype=np.int8)

        # generate pixel processor configurations for each spectral slice
        resampled_red_filter = [self.red_filter.sample_mv(slice.min_wavelength, slice.max_wavelength, slice.bins) for slice in spectral_slices]
        resampled_green_filter = [self.green_filter.sample_mv(slice.min_wavelength, slice.max_wavelength, slice.bins) for slice in spectral_slices]
        resampled_blue_filter = [self.blue_filter.sample_mv(slice.min_wavelength, slice.max_wavelength, slice.bins) for slice in spectral_slices]

        # build pixel processors
        red_processors = [PowerPixelProcessor(filter) for filter in resampled_red_filter]
        green_processors = [PowerPixelProcessor(filter) for filter in resampled_green_filter]
        blue_processors = [PowerPixelProcessor(filter) for filter in resampled_blue_filter]
        self._processors = [red_processors, green_processors, blue_processors]

        self._quiet = quiet

        if self.display_progress:
            self._start_display()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int x, int y, int slice_id):

        cdef:
            int index, filter_id
            PowerPixelProcessor processor

        index = (x % 2) + 2 * (y % 2)
        filter_id = self._bayer_mosaic[index]
        processor = self._processors[filter_id][slice_id]
        processor.reset()
        return processor

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object update(self, int x, int y, int slice_id, tuple packed_result):

        cdef:
            double mean, variance

        # obtain result
        mean, variance = packed_result

        # accumulate sub-samples
        self._working_mean[x, y] += mean
        self._working_variance[x, y] += variance

        # mark pixel as modified
        self._working_touched[x, y] = 1

        # update users
        if self.display_progress:
            self._update_display(x, y)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object finalise(self):

        cdef int x, y

        # update final frame with working frame results
        for x in range(self.frame.nx):
            for y in range(self.frame.ny):
                if self._working_touched[x, y] == 1:
                    self.frame.combine_samples(x, y, self._working_mean[x, y], self._working_variance[x, y], self._samples)

        if self.display_progress:
            self._render_display(self.frame)

    cpdef object _start_display(self):
        """
        Display live render.
        """

        # reset figure handle if we are not persisting across observation runs
        if not self.display_persist_figure:
            self._display_figure = None

        # populate live frame with current frame state
        self._display_frame = self.frame.copy()

        # display initial frame
        self._render_display(self._display_frame, 'rendering...')

        # workaround for interactivity for QT backend
        try:
            plt.pause(0.1)
        except NotImplementedError:
            pass

        self._display_timer = time()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object _update_display(self, int x, int y):
        """
        Update live render.
        """

        # update display pixel by combining existing frame data with working data
        self._display_frame.mean_mv[x, y] = self.frame.mean_mv[x, y]
        self._display_frame.variance_mv[x, y] = self.frame.variance_mv[x, y]
        self._display_frame.samples_mv[x, y] = self.frame.samples_mv[x, y]

        self._display_frame.combine_samples(x, y, self._working_mean[x, y], self._working_variance[x, y], self._samples)

        # update live render display
        if (time() - self._display_timer) > self.display_update_time:

            if not self._quiet:
                print("{} - updating display...".format(self.name))

            self._render_display(self._display_frame, 'rendering...')

            # workaround for interactivity for QT backend
            try:
                plt.pause(0.1)
            except NotImplementedError:
                pass

            self._display_timer = time()

    cpdef object _refresh_display(self):
        """
        Refreshes the display window (if active) and frame data is present.

        This method is called when display attributes are changed to refresh
        the display according to the new settings.
        """

        # there must be frame data present
        if not self.frame:
            return

        # is there a figure present (only present if display() called or display progress was on during render)?
        if not self._display_figure:
            return

        # does the figure have an active window?
        if not plt.fignum_exists(self._display_figure.number):
            return

        self._render_display(self.frame)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object _render_display(self, StatsArray2D frame, str status=None):

        INTERPOLATION = 'nearest'

        # generate display image
        image = self._generate_display_image(frame)

        # create a fresh figure if the existing figure window has gone missing
        if not self._display_figure or not plt.fignum_exists(self._display_figure.number):
            self._display_figure = plt.figure(facecolor=(0.5, 0.5, 0.5), figsize=_DISPLAY_SIZE, dpi=_DISPLAY_DPI)
        fig = self._display_figure

        # set window title
        if fig.canvas.manager is not None:
            if status:
                fig.canvas.manager.set_window_title("{} - {}".format(self.name, status))
            else:
                fig.canvas.manager.set_window_title(self.name)

        # populate figure
        fig.clf()
        ax = fig.add_axes([0,0,1,1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(np.transpose(image), aspect="equal", origin="upper", interpolation=INTERPOLATION, cmap=plt.get_cmap('gray'), vmin=0.0)
        fig.canvas.draw_idle()
        plt.show()

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray _generate_display_image(self, StatsArray2D frame):

        cdef:
            int nx, ny, x, y
            np.ndarray image
            double[:,::1] image_mv
            double gamma_exponent

        if self._display_auto_exposure:
            self._display_white_point = self._calculate_white_point(frame.mean)

        image = frame.mean.copy()
        image_mv = image

        # clamp data to within black and white point range, and shift zero to blackpoint, apply gamma correction
        nx = frame.shape[0]
        ny = frame.shape[1]
        gamma_exponent = 1.0 / self._display_gamma
        for x in range(nx):
            for y in range(ny):
                image_mv[x, y] = clamp(image_mv[x, y], self._display_black_point, self._display_white_point) - self._display_black_point
                image_mv[x, y] = pow(image_mv[x, y], gamma_exponent)

        return image

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double _calculate_white_point(self, np.ndarray image):

        cdef:
            int nx, ny, pixels, x, y, i
            double peak_luminance
            np.ndarray luminance
            double[:,::1] imv
            double[::1] lmv

        nx = image.shape[0]
        ny = image.shape[1]
        imv = image  # memory view

        pixels = nx * ny
        luminance = np.zeros(pixels)
        lmv = luminance  # memory view

        # calculate luminance values for frame
        for x in range(nx):
            for y in range(ny):
                lmv[y*nx + x] = max(imv[x, y] - self._display_black_point, 0)

        # sort by luminance
        luminance.sort()

        # if all pixels black, return default sensitivity
        for i in range(pixels):
            if lmv[i] > 0:
                break

        if i == pixels:
            return self._display_black_point

        # identify luminance at threshold
        peak_luminance = lmv[<int> min(pixels - 1, pixels * self._display_unsaturated_fraction)]

        if peak_luminance == 0:
            return self._display_black_point

        return peak_luminance + self._display_black_point

    cpdef object display(self):
        """
        Plot the RGB frame.
        """
        if not self.frame:
            raise ValueError("There is no frame to display.")
        self._render_display(self.frame)

    cpdef object save(self, str filename):
        """
        Saves the display image to a png file.

        The current display settings (exposure, gamma, etc..) are used to
        process the image prior saving.

        :param str filename: Image path and filename.
        """

        if not self.frame:
            raise ValueError("There is no frame to save.")

        image = self._generate_display_image(self.frame)
        plt.imsave(filename, np.transpose(image), cmap='gray', vmin=0.0)

