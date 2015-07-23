from numpy import empty, ndenumerate
from scipy.io.netcdf import netcdf_file
from raysect.core.math import Vector, Point
from raysect.optical.observer import VectoredCamera

camera_config = netcdf_file('kl11-p1db-87203.nc')

ray_start_coords = camera_config.variables['RayStartCoords'].shape
ray_end_coords = camera_config.variables['RayEndCoords'].shape

pixel_origins = empty(shape=ray_start_coords.shape[0:2]).tolist()
pixel_directions = empty(shape=ray_start_coords.shape[0:2]).tolist()

for (x, y), _ in ndenumerate(pixel_origins):
    xi, yi, zi = ray_start_coords[x, y]
    pixel_origins[x, y] = Point(xi, yi, zi)

for (x, y), _ in ndenumerate(pixel_directions):
    xi, yi, zi = ray_start_coords[x, y]
    xj, yj, zj = ray_end_coords[x, y]
    pixel_directions[x, y] = Vector(xj-xi, yj-yi, zj-zi)

vectorcam = VectoredCamera(pixel_origins, pixel_directions)