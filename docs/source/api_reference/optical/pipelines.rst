
Frame Samples and Pipelines
===========================


Base Classes
------------


Frame Sampler
^^^^^^^^^^^^^

For the higher dimensional observers (>0D) frame samplers determine how the individual
pixels that make up the observers frame are sampled. For example, sample the full frame
or adaptively select pixels to sample based on their noise level.

.. autoclass:: raysect.optical.observer.base.sampler.FrameSampler1D
   :members:

.. autoclass:: raysect.optical.observer.base.sampler.FrameSampler2D
   :members:


Pipeline
^^^^^^^^

Pipelines define how spectra are processed by observers into the form desired by the user.
For example, the power pipelines define how the measured spectra is integrated over the
spectral range to give the overall power in W arriving at the observing surfaces. The also
control the display and visualisation of results.

.. autoclass:: raysect.optical.observer.base.pipeline.Pipeline0D
   :members:

.. autoclass:: raysect.optical.observer.base.pipeline.Pipeline1D
   :members:

.. autoclass:: raysect.optical.observer.base.pipeline.Pipeline2D
   :members:


RGB
---

.. autoclass:: raysect.optical.observer.pipeline.rgb.RGBPipeline2D
   :members: display, display_auto_exposure, display_sensitivity, display_unsaturated_fraction, display_update_time, save
   :show-inheritance:


Bayer
-----

.. autoclass:: raysect.optical.observer.pipeline.bayer.BayerPipeline2D
   :members: display, display_auto_exposure, display_black_point, display_gamma, display_unsaturated_fraction,display_update_time, display_white_point, save
   :show-inheritance:


Power
-----

.. autoclass:: raysect.optical.observer.pipeline.mono.PowerPipeline0D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.mono.PowerPipeline1D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.mono.PowerPipeline2D
   :members: display_auto_exposure, display_unsaturated_fraction, display_update_time, save
   :show-inheritance:


Radiance
--------

.. autoclass:: raysect.optical.observer.pipeline.mono.RadiancePipeline0D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.mono.RadiancePipeline1D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.mono.RadiancePipeline2D
   :show-inheritance:


Spectral
--------

.. autoclass:: raysect.optical.observer.pipeline.spectral.SpectralPowerPipeline0D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.spectral.SpectralRadiancePipeline0D
   :members: to_spectrum
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.spectral.SpectralPowerPipeline1D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.spectral.SpectralRadiancePipeline1D
   :members: to_spectrum
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.spectral.SpectralPowerPipeline2D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.spectral.SpectralRadiancePipeline2D
   :members: to_spectrum
   :show-inheritance:


Frame Samplers
--------------

.. autoclass:: raysect.optical.observer.sampler1d.FullFrameSampler1D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.sampler1d.MonoAdaptiveSampler1D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.sampler1d.SpectralAdaptiveSampler1D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.sampler2d.FullFrameSampler2D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.sampler2d.MonoAdaptiveSampler2D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.sampler2d.SpectralAdaptiveSampler2D
   :show-inheritance:

.. autoclass:: raysect.optical.observer.sampler2d.RGBAdaptiveSampler2D
   :show-inheritance:
