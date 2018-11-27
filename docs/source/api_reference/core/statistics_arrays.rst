
Statistics Arrays
=================

StatsArrays are objects used in Raysect to store arrays of statistical
results with appropriate dimension. Consider for example a 2D pixel
array at a camera's image plane. For each pixel in the camera, Raysect
will trace a large number of rays, perhaps many 1000s of individual Ray
samples. A StatsArray therefore stores the overall statistical results
at each pixel index location. Note that these objects never store the
complete set of samples taken by Raysect, as the dataset would be too
large for suitable storage. Instead, Raysect stores the mean and variance
of all samples collected at each array index location.

Users are unlikely to ever instantiate these classes directly, it is more
likely users will encounter these objects in the pipeline results of their
observations.

.. autoclass:: raysect.core.math.statsarray.StatsBin
   :members:

.. autoclass:: raysect.core.math.statsarray.StatsArray1D
   :members:

.. autoclass:: raysect.core.math.statsarray.StatsArray2D
   :members:

.. autoclass:: raysect.core.math.statsarray.StatsArray3D
   :members:
