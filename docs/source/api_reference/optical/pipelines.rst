
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


Pixel Processor
^^^^^^^^^^^^^^^

A pixel processor handles the processing of the spectra for each pixel sampled by the
observer.

.. autoclass:: raysect.optical.observer.base.processor.PixelProcessor
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
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.rgb.XYZPixelProcessor
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.rgb.RGBAdaptiveSampler2D
   :members:
   :show-inheritance:


Bayer
-----

.. autoclass:: raysect.optical.observer.pipeline.bayer.BayerPipeline2D
   :members:
   :show-inheritance:


Power
-----

.. autoclass:: raysect.optical.observer.pipeline.power.PowerPipeline0D
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.power.PowerPipeline2D
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.power.PowerPixelProcessor
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.power.PowerAdaptiveSampler2D
   :members:
   :show-inheritance:


Spectral
--------

.. autoclass:: raysect.optical.observer.pipeline.spectral.SpectralPipeline0D
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.spectral.SpectralPipeline2D
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.pipeline.spectral.SpectralPixelProcessor
   :members:
   :show-inheritance:




