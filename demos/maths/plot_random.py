import numpy as np
import matplotlib.pyplot as plt
from raysect.core import Vector3D
from raysect.core.math.random import point_disk, point_square, vector_sphere, vector_hemisphere_uniform, vector_hemisphere_cosine, vector_cone

samples = 1000000


def vector_list_to_numpy(vectors):

    a = np.empty((len(vectors), 3))
    for i, v in enumerate(vectors):
        a[i, :] = v.x, v.y, v.z
    return a


# hemisphere sampling, cosine weighted
vectors = [vector_hemisphere_uniform() for i in range(samples)]
a = vector_list_to_numpy(vectors)
# w = [vector.dot(Vector3D(0, 0, 1)) for vector in vectors]

plt.figure()
plt.plot(a[:, 0], a[:, 1], '.')
#
plt.figure()
plt.plot(a[:, 0], a[:, 2], '.')

plt.figure()
plt.hist2d(a[:, 0], a[:, 1], bins=25)
# plt.hist2d(a[:, 0], a[:, 1], bins=25, weights=w)

# hemisphere sampling, cosine weighted
a = vector_list_to_numpy([vector_hemisphere_cosine() for i in range(samples)])

plt.figure()
plt.plot(a[:, 0], a[:, 1], '.')

plt.figure()
plt.plot(a[:, 0], a[:, 2], '.')

plt.figure()
plt.hist2d(a[:, 0], a[:, 1], bins=25)

plt.show()





