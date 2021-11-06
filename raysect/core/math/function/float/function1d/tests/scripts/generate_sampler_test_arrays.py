from raysect.core.math.function.float.function1d.samplers import sample1d
import numpy as np
np.set_printoptions(12, 30000, linewidth=100)


print('sampling')
sampling = np.linspace(0, 1, 20, dtype=np.float64)
print(sampling)

print('power series')
print(sampling ** sampling)