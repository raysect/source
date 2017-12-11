
import matplotlib.pyplot as plt
import numpy as np

from raysect.core.math.polygon import triangulate2d

# test polygon
poly = np.array([
    (-0.7, -1),
    (0.1, 0.1),
    (-0.3, 0.4),
    (-0.25, 0.5),
    (0, 0.9),
    (0.2, 1.15),
    (0.4, 1.2),
    (0.6, 1.0),
    (0.1, 0.4),
    (0.4, -0.3),
    (-0.1, -0.35)
], dtype=np.float64)

# reverse polygon (uncomment to test with an anti-clockwise ordered polygon)
# poly = poly[::-1, :].copy()

# subdivide into triangles
triangles = triangulate2d(poly)

# display
for t in triangles:
    v1x, v1y = poly[t[0], 0], poly[t[0], 1]
    v2x, v2y = poly[t[1], 0], poly[t[1], 1]
    v3x, v3y = poly[t[2], 0], poly[t[2], 1]
    plt.plot([v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 'r')

px = np.append(poly[:,0], [poly[0,0]])
py = np.append(poly[:,1], [poly[0,1]])

plt.plot(px, py, 'b')
plt.show()
