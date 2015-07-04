Conventions
-----------

In raysect the following conventions apply:

* the coordinate system is right-handed
    - the x-axis (-ve, +ve) maps to (right, left)
    - the y-axis (-ve, +ve) maps to (down, up)
    - the z-axis (-ve, +ve) maps to (back, forward)

* object orientation in local space (for general consistency)
    - objects with a clear up and forward direction must be aligned such that
      their forward direction is along the +ve z-axis and their up direction
      is aligned to point along the +ve y direction
    - where objects have an obvious axis of rotational symmetry (e.g. a
      cylinder or cone) that axis should be aligned with the Z-axis
    - where objects have an obvious plane of symmetry, that plane should be
      aligned with the y-z plane

In raysect.optical:

* dimensions are in meters
* angles are in degrees
* solid angles are in steradians
* wavelength is in nanometers
* power is in Watts
* spectral radiance is in W/m^2/str/nm