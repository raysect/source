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
from raysect.optical.observer.base cimport PixelProcessor, Pipeline2D
from raysect.core.math cimport StatsArray3D, StatsArray1D
from raysect.optical.colour cimport resample_ciexyz, spectrum_to_ciexyz, ciexyz_to_srgb

from raysect.optical.spectrum cimport Spectrum


_DEFAULT_PIPELINE_NAME = "RGBPipeline Pipeline"
_DISPLAY_DPI = 100
_DISPLAY_SIZE = (512 / _DISPLAY_DPI, 512 / _DISPLAY_DPI)


cdef class RGBPipeline2D(Pipeline2D):
    """
    2D pipeline of sRGB colour values.

    Converts the measured spectrum from each pixel into sRGB
    colour space values. See the colour module for more
    information. The RGBPipeline2D class is the workhorse
    pipeline for visualisation of scenes with Raysect and the
    default pipeline for most 2D observers.

    :param bool display_progress: Toggles the display of live render progress
      (default=True).
    :param float display_update_time: Time in seconds between preview display
      updates (default=15 seconds).
    :param bool accumulate: Whether to accumulate samples with subsequent calls
      to observe() (default=True).
    :param bool display_auto_exposure: Toggles the use of automatic exposure of
      final images (default=True).
    :param float display_sensitivity: The sensitivity of the camera, effectively
      inverse of the exposure time (default=1.0).
    :param float display_unsaturated_fraction: Fraction of pixels that must not
      be saturated. Display values will be scaled to satisfy this value
      (default=1.0).
    :param str name: User friendly name for this pipeline.
    """

    def __init__(self, bint display_progress=True,
                 double display_update_time=15, bint accumulate=True,
                 bint display_auto_exposure=True, double display_sensitivity=1.0,
                 double display_unsaturated_fraction=1.0, str name=None):

        self.name = name or _DEFAULT_PIPELINE_NAME

        self.display_progress = display_progress
        self.display_update_time = display_update_time
        self.display_persist_figure = True

        if display_sensitivity <= 0:
            raise ValueError("Sensitivity must be greater than 0.")

        self._display_sensitivity = display_sensitivity
        self._display_auto_exposure = display_auto_exposure
        self.display_unsaturated_fraction = display_unsaturated_fraction

        self.accumulate = accumulate

        self.xyz_frame = None

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
            self.display_progress,
            self.display_update_time,
            self.display_persist_figure,
            self._display_sensitivity,
            self._display_auto_exposure,
            self.display_unsaturated_fraction,
            self.accumulate,
            self.xyz_frame
        )

    def __setstate__(self, state):

        (
            self.name,
            self.display_progress,
            self.display_update_time,
            self.display_persist_figure,
            self._display_sensitivity,
            self._display_auto_exposure,
            self.display_unsaturated_fraction,
            self.accumulate,
            self.xyz_frame
        ) = state

        self._working_mean = None
        self._working_variance = None
        self._working_touched = None
        self._display_frame = None
        self._display_timer = 0
        self._display_figure = None
        self._pixels = None
        self._samples = 0
        self._quiet = False

    # must override automatic __reduce__ method generated by cython for the base class
    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @property
    def display_sensitivity(self):
        """
        The sensitivity of the camera, effectively inverse of the exposure time.

        :rtype: float
        """
        return self._display_sensitivity

    @display_sensitivity.setter
    def display_sensitivity(self, value):
        if value <= 0:
            raise ValueError("Sensitivity must be greater than 0.")
        self._display_auto_exposure = False
        self._display_sensitivity = value
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
        if not self.accumulate or self.xyz_frame is None or self.xyz_frame.shape != (nx, ny, 3):
            self.xyz_frame = StatsArray3D(nx, ny, 3)

        self._working_mean = np.zeros((nx, ny, 3))
        self._working_variance = np.zeros((nx, ny, 3))
        self._working_touched = np.zeros((nx, ny), dtype=np.int8)

        # generate pixel processor configurations for each spectral slice
        resampled_xyz = [resample_ciexyz(slice.min_wavelength, slice.max_wavelength, slice.bins) for slice in spectral_slices]
        self._processors = [XYZPixelProcessor(xyz) for xyz in resampled_xyz]

        self._quiet = quiet

        if self.display_progress:
            self._start_display()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int x, int y, int slice_id):
        cdef XYZPixelProcessor processor = self._processors[slice_id]
        processor.reset()
        return processor

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object update(self, int x, int y, int slice_id, tuple packed_result):

        cdef:
            double[::1] mean, variance

        # unpack results
        mean, variance = packed_result

        # accumulate sub-samples
        self._working_mean[x, y, 0] += mean[0]
        self._working_mean[x, y, 1] += mean[1]
        self._working_mean[x, y, 2] += mean[2]

        self._working_variance[x, y, 0] += variance[0]
        self._working_variance[x, y, 1] += variance[1]
        self._working_variance[x, y, 2] += variance[2]

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
        for x in range(self.xyz_frame.nx):
            for y in range(self.xyz_frame.ny):
                if self._working_touched[x, y] == 1:
                    self.xyz_frame.combine_samples(x, y, 0, self._working_mean[x, y, 0], self._working_variance[x, y, 0], self._samples)
                    self.xyz_frame.combine_samples(x, y, 1, self._working_mean[x, y, 1], self._working_variance[x, y, 1], self._samples)
                    self.xyz_frame.combine_samples(x, y, 2, self._working_mean[x, y, 2], self._working_variance[x, y, 2], self._samples)

        if self.display_progress:
            self._render_display(self.xyz_frame)

    cpdef object _start_display(self):
        """
        Display live render.
        """

        # reset figure handle if we are not persisting across observation runs
        if not self.display_persist_figure:
            self._display_figure = None

        # populate live frame with current frame state
        self._display_frame = self.xyz_frame.copy()

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
        self._display_frame.mean_mv[x, y, :] = self.xyz_frame.mean_mv[x, y, :]
        self._display_frame.variance_mv[x, y, :] = self.xyz_frame.variance_mv[x, y, :]
        self._display_frame.samples_mv[x, y, :] = self.xyz_frame.samples_mv[x, y, :]

        self._display_frame.combine_samples(x, y, 0, self._working_mean[x, y, 0], self._working_variance[x, y, 0], self._samples)
        self._display_frame.combine_samples(x, y, 1, self._working_mean[x, y, 1], self._working_variance[x, y, 1], self._samples)
        self._display_frame.combine_samples(x, y, 2, self._working_mean[x, y, 2], self._working_variance[x, y, 2], self._samples)

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
        if not self.xyz_frame:
            return

        # is there a figure present (only present if display() called or display progress was on during render)?
        if not self._display_figure:
            return

        # does the figure have an active window?
        if not plt.fignum_exists(self._display_figure.number):
            return

        self._render_display(self.xyz_frame)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object _render_display(self, StatsArray3D frame, str status=None):

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
        ax.imshow(np.transpose(image, (1, 0, 2)), aspect="equal", origin="upper", interpolation=INTERPOLATION)
        fig.canvas.draw_idle()
        plt.show()

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray _generate_display_image(self, StatsArray3D frame):

        cdef:
            int x, y, c
            np.ndarray xyz_image, rgb_image
            double[:,:,::1] xyz_image_mv
            double gamma_exponent

        if self._display_auto_exposure:
            self._display_sensitivity = self._calculate_sensitivity(frame.mean)

        xyz_image = frame.mean.copy()
        xyz_image_mv = xyz_image

        # apply sensitivity
        for x in range(frame.nx):
            for y in range(frame.ny):
                for c in range(frame.nz):
                    xyz_image_mv[x, y, c] *= self._display_sensitivity

        # convert XYZ to sRGB
        rgb_image = self._generate_srgb_image(xyz_image_mv)

        return rgb_image

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double _calculate_sensitivity(self, np.ndarray image):

        cdef:
            int nx, ny, pixels, x, y, i
            double peak_luminance
            np.ndarray luminance
            double[:,:,::1] imv
            double[::1] lmv

        nx = image.shape[0]
        ny = image.shape[1]
        imv = image  # memory view

        pixels = nx * ny
        luminance = np.zeros(pixels)
        lmv = luminance  # memory view

        # TODO - should really consider X and Z when working out brightness
        # calculate luminance values for frame (XYZ Y component is luminance)
        for x in range(nx):
            for y in range(ny):
                lmv[y*nx + x] = imv[x, y, 1]

        # sort by luminance
        luminance.sort()

        # if all pixels black, return default sensitivity
        for i in range(pixels):
            if lmv[i] > 0:
                break

        # return default sensitivity
        if i == pixels:
            return 1.0

        # identify luminance at threshold
        peak_luminance = lmv[<int> min(pixels - 1, pixels * self._display_unsaturated_fraction)]

        # return default sensitivity
        if peak_luminance == 0:
            return 1.0

        return 1 / peak_luminance

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray _generate_srgb_image(self, double[:,:,::1] xyz_image_mv):

        cdef:
            int nx, ny, ix, iy
            np.ndarray rgb_image
            double[:,:,::1] rgb_image_mv
            (double, double, double) rgb_pixel

        nx = xyz_image_mv.shape[0]
        ny = xyz_image_mv.shape[1]
        rgb_image = np.zeros((nx, ny, 3))
        rgb_image_mv = rgb_image

        # convert to sRGB colour space
        for ix in range(nx):
            for iy in range(ny):

                rgb_pixel = ciexyz_to_srgb(
                    xyz_image_mv[ix, iy, 0],
                    xyz_image_mv[ix, iy, 1],
                    xyz_image_mv[ix, iy, 2]
                )

                rgb_image_mv[ix, iy, 0] = rgb_pixel[0]
                rgb_image_mv[ix, iy, 1] = rgb_pixel[1]
                rgb_image_mv[ix, iy, 2] = rgb_pixel[2]

        return rgb_image

    cpdef object display(self):
        """
        Plot the RGB frame.
        """
        if not self.xyz_frame:
            raise ValueError("There is no frame to display.")
        self._render_display(self.xyz_frame)

    cpdef object save(self, str filename):
        """
        Saves the display image to a png file.

        The current display settings (exposure, gamma, etc..) are used to
        process the image prior saving.

        :param str filename: Image path and filename.
        """

        if not self.xyz_frame:
            raise ValueError("There is no frame to save.")

        image = self._generate_display_image(self.xyz_frame)
        plt.imsave(filename, np.transpose(image, (1, 0, 2)))


cdef class XYZPixelProcessor(PixelProcessor):
    """
    PixelProcessor that converts each pixel's spectrum into three
    XYZ colourspace values.
    """

    def __init__(self, double[:,::1] resampled_xyz not None):
        self.resampled_xyz = resampled_xyz
        self.xyz = StatsArray1D(3)

    cpdef object reset(self):
        self.xyz.clear()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object add_sample(self, Spectrum spectrum, double sensitivity):

        cdef double x, y, z

        # convert spectrum to CIE XYZ and add sample to pixel buffer
        x, y, z = spectrum_to_ciexyz(spectrum, self.resampled_xyz)
        self.xyz.add_sample(0, x * sensitivity)
        self.xyz.add_sample(1, y * sensitivity)
        self.xyz.add_sample(2, z * sensitivity)

    cpdef tuple pack_results(self):
        return self.xyz.mean, self.xyz.variance

