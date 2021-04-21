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
    if t == 0 or angle == 0:
        return 1.0
    # return 2 * n1 * math.cos(angle) / (n2 * math.cos(angle) + n1 * math.cos(t))
    return math.sqrt(math.sin(2*angle) * math.sin(2*t) / (math.sin(angle+t) * math.cos(angle-t))**2)


def fresnel_ts(n1, n2, angle):
    """
    Fresnel transmission coefficient for perpendicular E-field.
    """
    # if angle == 0:
    #     # todo: take limit of original eqn?
    #     #return 1 + fresnel_rs(n1, n2, angle)
    #     return 1 + (n1 - n2) / (n1 + n2)

    if n1 > n2 and angle > total_internal_reflection_angle(n1, n2):
        return 0.0

    t = transmission_angle(n1, n2, angle)
    if t == 0 or angle == 0:
        return 1.0
    # return 2 * n1 * math.cos(angle) / (n1 * math.cos(angle) + n2 * math.cos(t))
    return math.sqrt(math.cos(angle - t)**2 * math.sin(2*angle) * math.sin(2*t) / (math.sin(angle+t) * math.cos(angle-t))**2)


def plot_fresnel(n1, n2):
    # incident_angle = math.radians(25)
    # transmitted_angle = transmission_angle(n1, n2, incident_angle)
    #
    # print(f'incident angle: {math.degrees(incident_angle)}')
    # print(f'transmission angle: {math.degrees(transmitted_angle)}')
    # print(f'Brewster\'s angle: {math.degrees(brewster_angle(n1, n2))}')
    # print(f'TIR angle: {math.degrees(total_internal_reflection_angle(n1, n2))}')

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
        rp[i] = fresnel_rp(n1, n2, math.radians(a[i]))
        rs[i] = fresnel_rs(n1, n2, math.radians(a[i]))
        tp[i] = fresnel_tp(n1, n2, math.radians(a[i]))
        ts[i] = fresnel_ts(n1, n2, math.radians(a[i]))
        Rp[i] = rp[i]**2
        Rs[i] = rs[i]**2
        Tp[i] = tp[i]**2
        Ts[i] = ts[i]**2

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
    plt.legend(['Rp', 'Rs', 'R', 'Tp', 'Ts', 'T'])


plot_fresnel(1.0, 1.5)
plot_fresnel(1.5, 1.0)
plt.show()
