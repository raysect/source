
.. _observers-page:

Observers
=========

Observers describe the geometry of how rays will sample a scene, such as where
rays are generated on the image plane and their path through a camera before
sampling the rest of the scene. They also act as a manager class controlling
other important settings such as the sampling properties of the camera (how many
samples to obtain per pixel) and the wavelength range of the camera response.

All observers are derrived from a common base class which describes the common
properties of all observers and the overall observing workflow.

.. autoclass:: raysect.optical.observer.base.observer._ObserverBase
   :members:


0D Observers
------------

.. autoclass:: raysect.optical.observer.base.observer.Observer0D
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.nonimaging.sightline.SightLine
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.nonimaging.fibreoptic.FibreOptic
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.nonimaging.pixel.Pixel
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.nonimaging.targetted_pixel.TargettedPixel
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.nonimaging.mesh_pixel.MeshPixel
   :members:
   :show-inheritance:


1D Observers
------------

.. autoclass:: raysect.optical.observer.base.observer.Observer1D
   :members: _generate_rays, pixels, pixel_samples, frame_sampler, pipelines
   :show-inheritance:

.. autoclass:: raysect.optical.observer.nonimaging.mesh_camera.MeshCamera
   :members:
   :show-inheritance:


2D Observers
------------

.. autoclass:: raysect.optical.observer.base.observer.Observer2D
   :members: _generate_rays, pixels, pixel_samples, frame_sampler, pipelines
   :show-inheritance:

.. autoclass:: raysect.optical.observer.imaging.pinhole.PinholeCamera
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.imaging.orthographic.OrthographicCamera
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.imaging.ccd.CCDArray
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.observer.imaging.vector.VectorCamera
   :members:
   :show-inheritance:


