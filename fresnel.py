import math
import numpy as np
import matplotlib.pyplot as plt


def total_internal_reflection_angle(n1, n2):
    if n1 > n2:
        n1, n2 = n2, n1
    return transmission_angle(n1, n2, 0.5 * math.pi)


def transmission_angle(n1, n2, angle):
    return math.asin(n1 * math.sin(angle) / n2)


def brewster_angle(n1, n2):
    if n1 > n2:
        n1, n2 = n2, n1
    return math.atan(n2 / n1)


def fresnel_rp(n1, n2, angle):
    """
    Fresnel reflection coefficient for parallel E-field.
    """

    if n1 > n2 and angle > total_internal_reflection_angle(n1, n2):
        #todo: implement
        return 1.0

    ci = math.cos(angle)
    ct = math.cos(transmission_angle(n1, n2, angle))
    return (n1*ct - n2*ci) / (n2*ci + n1*ct)


def fresnel_rs(n1, n2, angle):
    """
    Fresnel reflection coefficient for perpendicular E-field.
    """

    if n1 > n2 and angle > total_internal_reflection_angle(n1, n2):
        #todo: implement
        return 1.0

    ci = math.cos(angle)
    ct = math.cos(transmission_angle(n1, n2, angle))
    return (n1*ci - n2*ct) / (n1*ci + n2*ct)


def fresnel_tp(n1, n2, angle):
    """
    Fresnel transmission coefficient for parallel E-field.
    """

    if n1 > n2 and angle > total_internal_reflection_angle(n1, n2):
        return 0.0

    t = transmission_angle(n1, n2, angle)
    ci = math.cos(angle)
    ct = math.cos(t)
    return 2*n1*ci / (n2*ci + n1*ct)


def fresnel_ts(n1, n2, angle):
    """
    Fresnel transmission coefficient for perpendicular E-field.
    """

    if n1 > n2 and angle > total_internal_reflection_angle(n1, n2):
        return 0.0

    t = transmission_angle(n1, n2, angle)
    ci = math.cos(angle)
    ct = math.cos(t)
    return 2*n1*ci / (n1*ci + n2*ct)


def plot_fresnel(n1, n2):

    n = 1000
    a = np.linspace(0, 90, n)
    rp = np.zeros(n)
    rs = np.zeros(n)
    tp = np.zeros(n)
    ts = np.zeros(n)
    Rp = np.zeros(n)
    Rs = np.zeros(n)
    Tp = np.zeros(n)
    Ts = np.zeros(n)
    for i in range(n):

        incident = math.radians(a[i])
        if n1 > n2 and incident > total_internal_reflection_angle(n1, n2):
            transmitted = 0.0
        else:
            transmitted = transmission_angle(n1, n2, incident)

        rp[i] = fresnel_rp(n1, n2, incident)
        rs[i] = fresnel_rs(n1, n2, incident)
        tp[i] = fresnel_tp(n1, n2, incident)
        ts[i] = fresnel_ts(n1, n2, incident)
        Rp[i] = rp[i]**2    # beam area for reflected and incident identical, i.e. area normalisation = 1.0
        Rs[i] = rs[i]**2    # beam area for reflected and incident identical, i.e. area normalisation = 1.0
        Tp[i] = (tp[i]**2) * n2*math.cos(transmitted) / (n1*math.cos(incident))     # normalise with projected beam area to give flux
        Ts[i] = (ts[i]**2) * n2*math.cos(transmitted) / (n1*math.cos(incident))     # normalise with projected beam area to give flux

    plt.figure()
    plt.title(f'Coefficients n1={n1}, n2={n2}')
    plt.grid()
    plt.plot(a, rp)
    plt.plot(a, rs)
    plt.plot(a, tp)
    plt.plot(a, ts)
    plt.legend(['rp', 'rs', 'tp', 'ts'])

    plt.figure()
    plt.title(f'Reflection/transmission n1={n1}, n2={n2}')
    plt.grid()
    plt.plot(a, Rp)
    plt.plot(a, Rs)
    plt.plot(a, 0.5*(Rp + Rs))
    plt.plot(a, Tp)
    plt.plot(a, Ts)
    plt.plot(a, 0.5*(Tp + Ts))
    plt.plot(a, 0.5*(Rp + Rs) + 0.5*(Tp + Ts))
    plt.legend(['Rp', 'Rs', 'R', 'Tp', 'Ts', 'T', 'TOTAL'])


plot_fresnel(1.0, 1.5)
plot_fresnel(1.5, 1.0)
plt.show()
