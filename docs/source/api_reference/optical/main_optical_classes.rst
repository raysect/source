
Main Optical Classes
====================

Optical Rays
------------

.. autoclass:: raysect.optical.ray.Ray
   :members:
   :show-inheritance:

Spectral Functions
------------------

.. autoclass:: raysect.optical.spectralfunction.SpectralFunction
   :members:

.. autoclass:: raysect.optical.spectrum.Spectrum
   :members: clear, copy, is_compatible, is_zero, new_spectrum, to_photons, total, wavelengths
   :show-inheritance:

.. autoclass:: raysect.optical.spectralfunction.NumericallyIntegratedSF
   :show-inheritance:

.. autoclass:: raysect.optical.spectralfunction.InterpolatedSF
   :show-inheritance:

.. autoclass:: raysect.optical.spectralfunction.ConstantSF
   :show-inheritance:

.. autofunction:: raysect.optical.spectrum.photon_energy


Colours
-------

The CIE 1931 colour spaces define quantitatively the link between pure physical colours
(i.e. wavelengths in the visible spectrum) and human perceivable colours. The mathematical
relationships between the spectrum and perceivable colours are based on the sensitivity
curves for the three different cone cells in the human eye. Raysect implements three X,
Y, Z normalised spectral functions from the CIE 1931 Standard Colorimetric Observer. For
more information see `Wikipedia <https://en.wikipedia.org/wiki/CIE_1931_color_space>`_.

.. data:: raysect.optical.colour.ciexyz_x

   X spectral function from CIE 1931 Standard Colorimetric Observer (normalised)

.. data:: raysect.optical.colour.ciexyz_y

   Y spectral function from CIE 1931 Standard Colorimetric Observer (normalised)

.. data:: raysect.optical.colour.ciexyz_z

   Z spectral function from CIE 1931 Standard Colorimetric Observer (normalised)

.. data:: raysect.optical.colour.d65_white

   CIE D65 standard illuminant, normalised to 1.0 over visual range 375-785 nm

.. autofunction:: raysect.optical.colour.resample_ciexyz

.. autofunction:: raysect.optical.colour.spectrum_to_ciexyz

.. autofunction:: raysect.optical.colour.ciexyy_to_ciexyz

.. autofunction:: raysect.optical.colour.ciexyz_to_ciexyy


Raysect also supports conversion of CIE colour space values to standard sRGB colour space
as defined by HP and Microsoft in 1996 as per IEC 61966-2-1:1999. For more information
see `Wikipedia <https://en.wikipedia.org/wiki/SRGB>`_.

.. autofunction:: raysect.optical.colour.ciexyz_to_srgb

.. autofunction:: raysect.optical.colour.srgb_to_ciexyz


Optical Scenegraph
------------------

.. autoclass:: raysect.optical.scenegraph.world.World
   :members:
   :show-inheritance:
