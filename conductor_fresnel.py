import numpy as np
import matplotlib.pyplot as plt
from math import *
#
#         ci2 = ci * ci
#         k0 = n * n + k * k
#         k1 = k0 * ci2 + 1
#         k2 = 2 * n * ci
#         k3 = k0 + ci2
#
#
#
# ((n * n + k * k) * ci2 + 1 - 2 * n * ci) / ((n * n + k * k) * ci2 + 1 + 2 * n * ci)
# ((n * n + k * k) + ci2 - 2 * n * ci) / ((n * n + k * k) + ci2 + 2 * n * ci))
#


# parallel
def rp(angle, n, k):

    ci = fabs(cos(radians(angle)))

    n2 = n*n
    k2 = k*k
    c2i = ci*ci

    a = (n2 + k2) + c2i
    b = 2*n*ci
    return (a - b) / (a + b)


# perpendicular
def rs(angle, n, k):

    ci = fabs(cos(radians(angle)))

    n2 = n*n
    k2 = k*k
    c2i = ci*ci

    a = (n2 + k2)*c2i + 1
    b = 2*n*ci
    return (a - b) / (a + b)


def phase(angle, n, k):

    ci = fabs(cos(radians(angle)))
    si = sqrt(1 - ci*ci)
    ti = si / ci

    n2 = n*n
    k2 = k*k
    s2i = si*si
    t2i = ti*ti

    w = sqrt((n2 - k2 - s2i)**2 + 4*n2*k2)
    f = 0.5*(n2 + k2 - s2i + w)
    g = 0.5*(k2 - n2 + s2i + w)

    return pi - atan2(2*sqrt(g)*si*ti, s2i * t2i - (f + g))



def plot_fresnel(ni, ki):

    n = 1000
    a = np.linspace(0, 90, n)
    Rp = np.zeros(n)
    Rs = np.zeros(n)
    P = np.zeros(n)
    for i in range(n):
        Rp[i] = rp(a[i], ni, ki)
        Rs[i] = rs(a[i], ni, ki)
        P[i] = degrees(phase(a[i], ni, ki))

    plt.figure()
    plt.title(f'Reflection magnitude n={ni}, k={ki}')
    plt.grid()
    plt.plot(a, Rp)
    plt.plot(a, Rs)
    plt.plot(a, 0.5*(Rs+Rp))
    plt.legend(['Rp', 'Rs', 'R'])

    plt.figure()
    plt.title(f'Reflection Phase n={ni}, k={ki}')
    plt.grid()
    plt.plot(a, P)
    plt.legend(['P'])


plot_fresnel(0.15, 3.48)
plt.show()