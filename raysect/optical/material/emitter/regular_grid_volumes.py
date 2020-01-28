# -*- coding: utf-8 -*-
#
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

"""
Regular Grid Volumes accelerate integration through inhomogeneous emitting volume
as they use pre-calculated values of spectral emissivity on a regular grid.

Use `RegularGridBox` class for Cartesian grids and `RegularGridCylinder` class for cylindrical
grids (3D or axisymmetrical).
"""

from raysect.optical import Point3D
from raysect.primitive import Cylinder, Subtract, Box
from .regular_grid_emitters import CylindricalRegularIntegrator, CartesianRegularIntegrator
from .regular_grid_emitters import CylindricalRegularEmitter, CartesianRegularEmitter


class RegularGridVolume:
    """
    Basic class for regular grid volumes.

    :ivar Node parent: Scene-graph parent node.
    :ivar AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system
        relative to the scene-graph parent.
    :ivar RegularGridEmitter material: RegularGridEmitter material.
    """

    def __init__(self, primitive):
        self._primitive = primitive

    @property
    def parent(self):
        return self._primitive.parent

    @parent.setter
    def parent(self, value):
        self._primitive.parent = value

    @property
    def transform(self):
        return self._primitive.transform

    @transform.setter
    def transform(self, value):
        self._primitive.transform = value

    @property
    def material(self):
        return self._primitive.material


class RegularGridCylinder(RegularGridVolume):
    """
    Regular Grid Volume for cylindrical emitter defined on a regular 3D :math:`(R, \phi, Z)` grid.
    The emitter is periodic in :math:`\phi` direction.
    The base of the cylinder is located at `Z = 0` plane. Use `transform`
    parameter to move it.

    :param object ~.emission: The 2D or 4D array or scipy sparse matrix containing the
        emission defined on a regular :math:`(R, \phi, Z)` grid in :math:`W/(str\,m^3\,nm)`
        (contineous spectrum) or in :math:`W/(str\,m^3)` (discrete spectrum).
        Spectral emission can be provided either for selected cells of the regular
        grid (2D array or sparse matrix) or for all grid cells (4D array).
        Note that if provided as a 2D array (or sparse matrix), the argument `grid_shape`
        must be provided and the spatial index `(ir, iphi, iz)` must be flattened in a row-major
        order: `iflat = grid_shape[1] * grid_shape[2] * ir + grid_shape[2] * iphi + iz`.
        Regardless of the form in which the emission is provided, the last axis is the
        spectral one.  The emission will be stored as a сompressed sparse column matrix
        (`scipy.sparse.csc_matrix`). To reduce memory consumption, provide it as a `csc_matrix`.
    :param ndarray wavelengths: The 1D array of wavelengths corresponding to the last axis of
        provided emission array. The size of this array must be equal to `emission.shape[-1]`.
        Initialisation will be faster if this array contains monotonically increasing values.
    :param float radius_outer: Radius of the outer cylinder and the upper bound of grid in
        `R` direction (in meters).
    :param float height: Height of the cylinder and the length of grid in `Z` direction
        (in meters).
    :param float radius_inner: Radius of the inner cylinder and the lower bound of grid in
        `R` direction (in meters), defaults to `radius_inner=0`.
    :param float period: A period in :math:`\phi` direction (in degree), defaults to `period=360`.
    :param tuple grid_shape: The number of grid cells along each direction,
        defaults to `grid_shape=None`. Ignored if emission is a 4D array.
    :param float step: The step of integration along the ray (in meters), defaults to
        `0.25*min((radius_outer - radius_inner) / n_r, height / n_z)`, where n_r and n_z are
        the grid resolutions in `R` and `Z` directions.
    :param bool contineous: Defines whether the emission is porvided as a contineous spectrum
        (in :math:`W/(str\,m^3\,nm)`) or as a discrete spectrum (in :math:`W/(str\,m^3)`).
        Defaults to `contineous=True`.
    :param bool extrapolate: If True, the emission outside the provided spectral range
        will be equal to the emission at the borders of this range (nearest-neighbour
        extrapolation), otherwise it will be zero. Defaults to `extrapolate=True`.
        This parameter is ignored if `contineous=False`.
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system
        relative to the scene-graph parent (default = identity matrix).

    :ivar Node parent: Scene-graph parent node.
    :ivar AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system
        relative to the scene-graph parent.
    :ivar CylindricalRegularmitter material: CylindricalRegularEmitter material.

    Continoues spectrum example:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import World, translate, rotate
        >>> from raysect.primitive import Cylinder, Subtract
        >>> from raysect.optical.material import RegularGridCylinder
        >>> ### Contineous case ###
        >>> # grid parameters
        >>> rmin = 0
        >>> rmax = 2.
        >>> zmin = -0.25
        >>> zmax = 0.25
        >>> r, dr = np.linspace(rmin, rmax, 201, retstep=True)
        >>> r = r[:-1] + 0.5 * dr  # moving to the grid cell centers
        >>> integration_step = 0.05
        >>> # spectral emission profile
        >>> min_wavelength = 375.
        >>> max_wavelength = 740.
        >>> wavelengths, delta_wavelength = np.linspace(min_wavelength, max_wavelength, 50,
                                                        retstep=True)
        >>> wvl_centre = 0.5 * (max_wavelength + min_wavelength)
        >>> wvl_range = min_wavelength - max_wavelength
        >>> shift = 2 * (wavelengths - wvl_centre) / wvl_range + 5.
        >>> emission = np.cos(shift[None, None, None, :] * radius[:, None, None, None])**4
        >>> # scene
        >>> world = World()
        >>> emitter = RegularGridCylinder(emission, wavelengths, radius_outer=rmax,
                                          height=zmax - zmin, radius_inner=rmin,
                                          period=phi_period, parent=world)
        >>> emitter.material.transform = translate(-rmax, -rmax + 1., zmin) * rotate(30, 0, 0)
        ...
        >>> # if ray spectral properties do not change during the rendering,
        >>> # build the cache before the first camera.observe() call to reduce memory consumptions
        >>> emitter.material.cache_build(camera.min_wavelength, camera.max_wavelength,
                                         camera.spectral_bins)

    Discrete spectrum example:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import World, translate
        >>> from raysect.optical.observer import SpectralRadiancePipeline2D
        >>> from raysect.primitive import Cylinder, Subtract
        >>> from raysect.optical.material import RegularGridCylinder
        >>> # Assume that the files 'Be_4574A.npy' and 'Be_527A.npy' contain the emissions
        >>> # (in W / m^3) of Be I (3d1 1D2 -> 2p1 1P1) and Be II (4s1 2S0.5 -> 3p1 2P2.5)
        >>> # spectral lines defined on a regular cylindrical grid: 3.5 m < R < 9 m,
        >>> # 0 < phi < 20 deg, -5 m < Z < 5 m, and periodic in phi direction.
        >>> emission_4574 = np.load('Be_4574A.npy')
        >>> emission_5272 = np.load('Be_5272A.npy')
        >>> wavelengths = np.array([457.4, 527.2])
        >>> # Grid properties
        >>> rmin = 3.5
        >>> rmax = 9.
        >>> phi_period = 20.
        >>> zmin = -5.
        >>> zmax = 5.
        >>> grid_shape = emission_4574.shape
        >>> emission = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], 2))
        >>> emission[:, :, :, 0] = emission_4574 / (4. * np.pi)  # to W/(m^3 str)
        >>> emission[:, :, :, 1] = emission_5272 / (4. * np.pi)
        >>> # Creating the scene
        >>> world = World()
        >>> pipeline = SpectralRadiancePipeline2D()
        >>> emitter = RegularGridCylinder(emission, wavelengths, radius_outer=rmax,
                                          height=zmax - zmin, radius_inner=rmin,
                                          period=phi_period, parent=world,
                                          transform = translate(0, 0, zmin),
                                          contineous=False)
        ...
        >>> camera.spectral_bins = 15
        >>> camera.min_wavelength = 457.
        >>> camera.max_wavelength = 528.
        >>> delta_wavelength = (camera.max_wavelength - camera.min_wavelength)/camera.spectral_bins
        >>> # if ray spectral properties do not change during the rendering,
        >>> # build the cache before the first camera.observe() call to reduce memory consumptions
        >>> emitter.material.cache_build(camera.min_wavelength, camera.max_wavelength,
                                         camera.spectral_bins)
        ...
        >>> # If reflections do not change the wavelength, the results for each spectral line
        >>> # can be obtained in W/(m^2 str) in the following way.
        >>> radiance_4574 = pipeline.frame.mean[:, :, 0] * delta_wavelength
        >>> radiance_5272 = pipeline.frame.mean[:, :, -1] * delta_wavelength
    """

    def __init__(self, emission, wavelengths, radius_outer, height, radius_inner=0, period=360., grid_shape=None,
                 step=None, contineous=True, extrapolate=True, parent=None, transform=None):
        if 360. % period > 1.e-3:
            raise ValueError("The period %.3f is not a multiple of 360." % period)
        if emission.ndim == 2:
            if grid_shape is None:
                raise ValueError("If 'emission' is a 2D array, 'grid_shape' parameter must be specified.")
            if len(grid_shape) != 3:
                raise ValueError("Argument 'grid_shape' must contain 3 elements.")
            if grid_shape[0] <= 0 or grid_shape[1] <= 0 or grid_shape[2] <= 0:
                raise ValueError('Grid sizes must be > 0.')

        elif emission.ndim == 4:
            grid_shape = (emission.shape[0], emission.shape[1], emission.shape[2])

        else:
            raise ValueError("Argument 'emission' must be a 4D or 2D array.")

        dr = (radius_outer - radius_inner) / grid_shape[0]
        dphi = period / grid_shape[1]
        dz = height / grid_shape[2]
        grid_steps = (dr, dphi, dz)
        step = step or 0.25 * min(dr, dz)
        material = CylindricalRegularEmitter(grid_shape, grid_steps, emission, wavelengths, contineous=contineous, extrapolate=extrapolate,
                                             integrator=CylindricalRegularIntegrator(step), rmin=radius_inner)
        primitive = Subtract(Cylinder(radius_outer, height), Cylinder(radius_inner, height),
                             material=material, parent=parent, transform=transform)
        super().__init__(primitive)


class RegularGridBox(RegularGridVolume):
    """
    Regular Grid Volume for rectangular emitter defined on a regular 3D :math:`(X, Y, Z)` grid.
    The grid starts at (0, 0, 0). Use `transform` parameter to move it.

    :param object ~.emission: The 2D or 4D array or scipy sparse matrix containing the
        emission defined on a regular :math:`(X, Y, Z)` grid in :math:`W/(str\,m^3\,nm)`
        (contineous spectrum) or in :math:`W/(str\,m^3)` (discrete spectrum).
        Spectral emission can be provided either for selected cells of the regular
        grid (2D array or sparse matrix) or for all grid cells (4D array).
        Note that if provided as a 2D array (or sparse matrix), the argument `grid_shape`
        must be provided and the spatial index `(ix, iy, iz)` must be flattened in a row-major
        order: `iflat = grid_shape[1] * grid_shape[2] * ix + grid_shape[2] * iy + iz`.
        Regardless of the form in which the emission is provided, the last axis is the
        spectral one.  The emission will be stored as a сompressed sparse column matrix
        (`scipy.sparse.csc_matrix`). To reduce memory consumption, provide it as a `csc_matrix`.
    :param ndarray wavelengths: The 1D array of wavelengths corresponding to the last axis of
        provided emission array. The size of this array must be equal to `emission.shape[-1]`.
        Initialisation will be faster if this array contains monotonically increasing values.
    :param float xmax: Upper bound of grid in `X` direction (in meters).
    :param float ymax: Upper bound of grid in `Y` direction (in meters).
    :param float zmax: Upper bound of grid in `Z` direction (in meters).
    :param tuple grid_shape: The number of grid cells along each direction,
        defaults to `grid_shape=None`. Ignored if emission is a 4D array.
    :param float step: The step of integration along the ray (in meters), defaults to
        `step = 0.25 * min(xmax / n_x, ymax / n_y, zmax / n_z)`, where (n_x, n_y, n_z) is
        the grid resolution.
    :param bool contineous: Defines whether the emission is porvided as a contineous spectrum
        (in :math:`W/(str\,m^3\,nm)`) or as a discrete spectrum (in :math:`W/(str\,m^3)`).
        Defaults to `contineous=True`.
    :param bool extrapolate: If True, the emission outside the provided spectral range
        will be equal to the emission at the borders of this range (nearest-neighbour
        extrapolation), otherwise it will be zero. Defaults to `extrapolate=True`.
        This parameter is ignored if `contineous=False`.
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system
        relative to the scene-graph parent (default = identity matrix).

    :ivar Node parent: Scene-graph parent node.
    :ivar AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system
        relative to the scene-graph parent.
    :ivar CartesianRegularEmitter material: CartesianRegularEmitter material.

    Continoues spectrum example:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import World, translate, rotate
        >>> from raysect.primitive import Cylinder, Subtract
        >>> from raysect.optical.material import RegularGridBox
        >>> # grid parameters
        >>> xmin = ymin = -1.
        >>> xmax = ymax = 1.
        >>> zmin = -0.25
        >>> zmax = 0.25
        >>> x, dx = np.linspace(xmin, xmax, 101, retstep=True)
        >>> y, dy = np.linspace(ymin, ymax, 101, retstep=True)
        >>> x = x[:-1] + 0.5 * dx  # moving to the grid cell centers
        >>> y = y[:-1] + 0.5 * dy
        >>> # spectral emission profile
        >>> min_wavelength = 375.
        >>> max_wavelength = 740.
        >>> wavelengths, delta_wavelength = np.linspace(min_wavelength, max_wavelength, 50,
                                                        retstep=True)
        >>> wvl_centre = 0.5 * (max_wavelength + min_wavelength)
        >>> wvl_range = min_wavelength - max_wavelength
        >>> shift = 2 * (wavelengths - wvl_centre) / wvl_range + 5.
        >>> radius = np.sqrt((x * x)[:, None] + (y * y)[None, :])
        >>> emission = np.cos(shift[None, None, None, :] * radius[:, :, None, None])**4
        >>> # scene
        >>> world = World()
        >>> emitter = RegularGridBox(emission, wavelengths, xmax - xmin, ymax - ymin, zmax - zmin,
                                     parent=world)
        >>> emitter.material.transform = (translate(0, 1., 0) * rotate(30, 0, 0) *
                                          translate(xmin, ymin, zmin))
        ...
        >>> # if ray spectral properties do not change during the rendering,
        >>> # build the cache before the first camera.observe() call to reduce memory consumptions
        >>> emitter.material.cache_build(camera.min_wavelength, camera.max_wavelength,
                                         camera.spectral_bins)

    Discrete spectrum example:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from raysect.optical import World, translate, Point3D
        >>> from raysect.primitive import Box
        >>> from raysect.optical.observer import SpectralRadiancePipeline2D
        >>> from raysect.optical.material import RegularGridBox
        >>> # Assume that the files 'Be_4574A.npy' and 'Be_527A.npy' contain the emissions
        >>> # (in W / m^3) of Be I (3d1 1D2 -> 2p1 1P1) and Be II (4s1 2S0.5 -> 3p1 2P2.5)
        >>> # spectral lines defined on a regular Cartesian grid: -3 m < X < 3 m,
        >>> # -3 m < Y < 3 m and -6 m < Z < 6 m.
        >>> emission_4574 = np.load('Be_4574A.npy')
        >>> emission_5272 = np.load('Be_5272A.npy')
        >>> # Grid properties
        >>> xmin = ymin = -3.
        >>> xmax = ymax = 3.
        >>> zmin = -6.
        >>> zmax = 6.
        >>> grid_shape = emission_4574.shape
        >>> emission = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], 2))
        >>> emission[:, :, :, 0] = emission_4574 / (4. * np.pi)  # to W/(m^3 str)
        >>> emission[:, :, :, 1] = emission_5272 / (4. * np.pi)
        >>> # Creating the scene
        >>> world = World()
        >>> pipeline = SpectralRadiancePipeline2D()
        >>> material = RegularGridBox(emission, wavelengths, xmax - xmin, ymax - ymin, zmax - zmin,
                                      contineous=False, transform=translate(xmin, ymin, zmin),
                                      parent=world)
        ...
        >>> camera.spectral_bins = 15
        >>> camera.min_wavelength = 457
        >>> camera.max_wavelength = 528
        >>> delta_wavelength = (camera.max_wavelength - camera.min_wavelength)/camera.spectral_bins
        >>> # if ray spectral properties do not change during the rendering,
        >>> # build the cache before the first camera.observe() call to reduce memory consumptions
        >>> material.cache_build(camera.min_wavelength, camera.max_wavelength,
                                 camera.spectral_bins)
        ...
        >>> # If reflections do not change the wavelength, the results for each spectral line
        >>> # can be obtained in W/(m^2 sr) in the following way.
        >>> radiance_4574 = pipeline.frame.mean[:, :, 0] * delta_wavelength
        >>> radiance_5272 = pipeline.frame.mean[:, :, -1] * delta_wavelength
    """

    def __init__(self, emission, wavelengths, xmax, ymax, zmax, grid_shape=None, step=None,
                 contineous=True, extrapolate=True, parent=None, transform=None):

        if emission.ndim == 2:
            if grid_shape is None:
                raise ValueError("If 'emission' is a 2D array, 'grid_shape' parameter must be specified.")
            if len(grid_shape) != 3:
                raise ValueError("Argument 'grid_shape' must contain 3 elements.")
            if grid_shape[0] <= 0 or grid_shape[1] <= 0 or grid_shape[2] <= 0:
                raise ValueError('Grid sizes must be > 0.')

        elif emission.ndim == 4:
            grid_shape = (emission.shape[0], emission.shape[1], emission.shape[2])

        else:
            raise ValueError("Argument 'emission' must be a 4D or 2D array.")

        dx = xmax / grid_shape[0]
        dy = ymax / grid_shape[1]
        dz = zmax / grid_shape[2]
        grid_steps = (dx, dy, dz)
        step = step or 0.25 * min(dx, dy, dz)
        material = CartesianRegularEmitter(grid_shape, grid_steps, emission, wavelengths, contineous=contineous,
                                           extrapolate=extrapolate, integrator=CartesianRegularIntegrator(step))
        primitive = Box(lower=Point3D(0, 0, 0), upper=Point3D(xmax, ymax, zmax), material=material, parent=parent, transform=transform)
        super().__init__(primitive)
