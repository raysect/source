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
    return -(n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)


def fresnel_rs(n1, n2, angle):
    """
    Fresnel reflection coefficient for perpendicular E-field.
    """

    if n1 > n2 and angle > total_internal_reflection_angle(n1, n2):
        #todo: implement
        return 1.0

    ci = math.cos(angle)
    ct = math.cos(transmission_angle(n1, n2, angle))
    return (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)


def fresnel_tp(n1, n2, angle):
    """
    Fresnel transmission coefficient for parallel E-field.
    """

    if angle == 0:
        # todo: take limit of original eqn?
        return 1.0 + fresnel_rp(n1, n2, angle)

    if n1 > n2 and angle > total_internal_reflection_angle(n1, n2):
        return 0.0

    t = transmission_angle(n1, n2, angle)
    return fresnel_ts(n1, n2, angle) / math.cos(angle - t)


def fresnel_ts(n1, n2, angle):
    """
    Fresnel transmission coefficient for perpendicular E-field.
    """
    if angle == 0:
        # todo: take limit of original eqn?
        return 1 + fresnel_rs(n1, n2, angle)

    if n1 > n2 and angle > total_internal_reflection_angle(n1, n2):
        return 0.0

    t = transmission_angle(n1, n2, angle)
    return 2 * (math.cos(angle) * math.sin(t)) / (math.sin(angle + t))



n1 = 1.0
n2 = 1.5
incident_angle = math.radians(56.31)
# transmitted_angle = transmission_angle(n1, n2, incident_angle)

print(f'incident angle: {math.degrees(incident_angle)}')
# print(f'transmission angle: {math.degrees(transmitted_angle)}')
print(f'Brewster\'s angle: {math.degrees(brewster_angle(n1, n2))}')
print(f'TIR angle: {math.degrees(total_internal_reflection_angle(n1, n2))}')

n = 100
a = np.linspace(0, 90, n)
rp = np.zeros(n)
rs = np.zeros(n)
tp = np.zeros(n)
ts = np.zeros(n)
for i in range(n):
    rp[i] = fresnel_rp(n1, n2, math.radians(a[i]))
    rs[i] = fresnel_rs(n1, n2, math.radians(a[i]))
    tp[i] = fresnel_tp(n1, n2, math.radians(a[i]))
    ts[i] = fresnel_ts(n1, n2, math.radians(a[i]))

plt.figure()
plt.grid()
plt.plot(a, rp)
plt.plot(a, rs)
plt.plot(a, tp)
plt.plot(a, ts)
plt.legend(['rp', 'rs', 'tp', 'ts'])
plt.show()


