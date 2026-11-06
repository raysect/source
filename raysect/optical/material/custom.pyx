# cython: language_level=3
# SPDX-License-Identifier: BSD-3-Clause
"""Spectrally tabulated isotropic BRDF material."""
import numpy as np
cimport cython
from libc.math cimport acos, atan2, sqrt, M_PI, isfinite
from raysect.core.math.sampler cimport HemisphereCosineSampler
from raysect.optical cimport Point3D, Vector3D, AffineMatrix3D, Primitive, World, Ray, Spectrum, Intersection
from raysect.optical.material cimport ContinuousBSDF

cdef HemisphereCosineSampler sampler = HemisphereCosineSampler()

cdef class CustomBRDF(ContinuousBSDF):
    """Isotropic BRDF tabulated over wavelength and surface angles.

    ``table[w, i, j, k]`` contains BRDF values in sr^-1. ``wavelengths``
    gives the strictly increasing, positive wavelength coordinates in nm.
    The angular axes are uniformly spaced: incidence and outgoing polar
    angles include 0 and pi/2; relative azimuth is (phi_out - phi_in) modulo
    2*pi and covers [0, 2*pi), excluding the periodic endpoint. Both
    directions point away from the surface.
    At least two points are required on each angular axis. Values must be
    finite and non-negative. Reciprocity and energy conservation are the
    caller's responsibility.

    Wavelengths are linearly interpolated; outside the supplied range the
    nearest endpoint value is used. A single wavelength is constant at all
    wavelengths. For compatibility, a three-dimensional table without a
    wavelengths argument is also treated as wavelength independent.
    Sampling uses a wavelength-independent cosine-weighted hemisphere.
    The table is copied into contiguous float64 storage at construction.
    """
    cdef double[:, :, :, ::1] table
    cdef double[::1] wavelengths
    cdef Py_ssize_t nw, ni, np_, no

    def __init__(self, table, wavelengths=None):
        super().__init__()
        values = np.array(table, dtype=np.float64, order='C', copy=True)
        if values.ndim == 3 and wavelengths is None:
            values = values[None, ...]
            wavelengths = [1.0]
        if values.ndim != 4 or values.shape[0] < 1 or any(n < 2 for n in values.shape[1:]):
            raise ValueError('table must have shape (wavelength, incidence, azimuth, outgoing) with at least two samples per angular axis')
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError('table values must be finite and non-negative')
        if wavelengths is None:
            raise ValueError('wavelength coordinates in nm are required for a four-dimensional table')
        wave = np.array(wavelengths, dtype=np.float64, order='C', copy=True)
        if wave.ndim != 1 or len(wave) != values.shape[0] or not np.all(np.isfinite(wave)) or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
            raise ValueError('wavelengths must be positive, finite, strictly increasing and match the table')
        self.table = values
        self.wavelengths = wave
        self.nw, self.ni, self.np_, self.no = values.shape

    cpdef double pdf(self, Vector3D s_incoming, Vector3D s_outgoing, bint back_face) noexcept:
        return sampler.pdf(s_outgoing)

    cpdef Vector3D sample(self, Vector3D s_incoming, bint back_face):
        return sampler.sample()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double bsdf(self, Vector3D s_incident, Vector3D s_reflected, double wavelength) noexcept:
        cdef double li, lo, ti, to, phi, a, b, c, result = 0
        cdef Py_ssize_t i, j, k, jj, di, dj, dk, w, upper, mid
        cdef double blend, v0, v1
        cdef double wi, wj, wk
        if not isfinite(wavelength):
            return 0.0
        w = 0
        blend = 0
        if self.nw > 1:
            if wavelength >= self.wavelengths[self.nw - 1]:
                w = self.nw - 1
            elif wavelength > self.wavelengths[0]:
                upper = self.nw - 1
                while upper - w > 1:
                    mid = (w + upper) // 2
                    if wavelength < self.wavelengths[mid]:
                        upper = mid
                    else:
                        w = mid
                blend = (wavelength - self.wavelengths[w]) / (self.wavelengths[w + 1] - self.wavelengths[w])
        li = sqrt(s_incident.x**2 + s_incident.y**2 + s_incident.z**2)
        lo = sqrt(s_reflected.x**2 + s_reflected.y**2 + s_reflected.z**2)
        if li == 0 or lo == 0:
            return 0.0
        if s_incident.z < 0 or s_reflected.z < 0:
            return 0
        ti = acos(min(1.0, max(0.0, s_incident.z / li))) * (self.ni - 1) / (M_PI / 2)
        to = acos(min(1.0, max(0.0, s_reflected.z / lo))) * (self.no - 1) / (M_PI / 2)
        phi = atan2(s_reflected.y, s_reflected.x) - atan2(s_incident.y, s_incident.x)
        if phi < 0:
            phi += 2 * M_PI
        if phi >= 2 * M_PI:
            phi -= 2 * M_PI
        phi *= self.np_ / (2 * M_PI)
        i = min(<Py_ssize_t>ti, self.ni - 2)
        k = min(<Py_ssize_t>to, self.no - 2)
        j = min(<Py_ssize_t>phi, self.np_ - 1)
        a, b, c = ti - i, phi - j, to - k
        for di in range(2):
            wi = a if di else 1 - a
            for dj in range(2):
                wj = b if dj else 1 - b
                jj = (j + dj) % self.np_
                for dk in range(2):
                    wk = c if dk else 1 - c
                    v0 = self.table[w, i + di, jj, k + dk]
                    if blend:
                        v1 = self.table[w + 1, i + di, jj, k + dk]
                        v0 += blend * (v1 - v0)
                    result += wi * wj * wk * v0
        return result

    cpdef Spectrum evaluate_shading(self, World world, Ray ray, Vector3D s_incoming, Vector3D s_outgoing,
                                    Point3D w_reflection_origin, Point3D w_transmission_origin, bint back_face,
                                    AffineMatrix3D world_to_surface, AffineMatrix3D surface_to_world,
                                    Intersection intersection):
        cdef Spectrum spectrum
        cdef Ray reflected
        cdef int index
        cdef double wavelength
        if s_incoming.z <= 0 or s_outgoing.z <= 0:
            return ray.new_spectrum()
        reflected = ray.spawn_daughter(w_reflection_origin, s_outgoing.transform(surface_to_world))
        spectrum = reflected.trace(world)
        for index in range(spectrum.bins):
            wavelength = spectrum.min_wavelength + (index + 0.5) * spectrum.delta_wavelength
            spectrum.samples_mv[index] *= self.bsdf(s_incoming, s_outgoing, wavelength) * s_outgoing.z
        return spectrum

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):
        return spectrum
