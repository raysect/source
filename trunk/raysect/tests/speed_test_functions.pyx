from raysect.core.math.vector cimport Vector, new_vector

# using python like calls

def ctest1(int n):
    
    cdef Vector v1, v2, v3
    
    v1 = Vector([5.0, 6.0, 7.0])
    v2 = Vector([-10.0, 55.0, 23.0])

    for i in range(0, n):

        v3 = v1 + v2

    return v3  


def ctest2(int n):

    cdef Vector v1, v2, v3

    v1 = Vector([5.0, 6.0, 7.0])
    v2 = Vector([-10.0, 55.0, 23.0])

    for i in range(0, n):

        v3 = v1 - v2

    return v3


def ctest3(int n):

    cdef Vector v1, v2
    cdef double r

    v1 = Vector([5.0, 6.0, 7.0])
    v2 = Vector([-10.0, 55.0, 23.0])

    for i in range(0, n):

        r = v1.dot(v2)

    return r


def ctest4(int n):

    cdef Vector v1, v2, v3

    v1 = Vector([5.0, 6.0, 7.0])
    v2 = Vector([-10.0, 55.0, 23.0])

    for i in range(0, n):

        v3 = v1.cross(v2)
        
    return v3


def ctest5(int n):

    cdef Vector incident, normal, reflected

    incident = Vector([1,-1,0]).normalise()
    normal = Vector([0,1,0]).normalise()

    for i in range(0, n):

        reflected = incident - 2.0 * normal * (normal.dot(incident))

    return reflected

# optimised

def cotest1(int n):
    
    cdef Vector v1, v2, v3
    cdef double x,y,z
    
    v1 = new_vector(5.0, 6.0, 7.0)
    v2 = new_vector(-10.0, 55.0, 23.0)

    for i in range(0, n):

        v3 = v1.add(v2)

    return v3  


def cotest2(int n):

    cdef Vector v1, v2, v3

    v1 = new_vector(5.0, 6.0, 7.0)
    v2 = new_vector(-10.0, 55.0, 23.0)

    for i in range(0, n):

        v3 = v1.sub(v2)

    return v3


def cotest3(int n):

    cdef Vector v1, v2
    cdef double r

    v1 = new_vector(5.0, 6.0, 7.0)
    v2 = new_vector(-10.0, 55.0, 23.0)

    for i in range(0, n):

        r = v1.dot(v2)

    return r


def cotest4(int n):

    cdef Vector v1, v2, v3

    v1 = new_vector(5.0, 6.0, 7.0)
    v2 = new_vector(-10.0, 55.0, 23.0)

    for i in range(0, n):

        v3 = v1.cross(v2)
        
    return v3


def cotest5(int n):

    cdef Vector incident, normal, reflected
    cdef double d

    incident = new_vector(1, -1, 0).normalise()
    normal = new_vector(0, 1, 0).normalise()

    for i in range(0, n):

        # r = i - 2*n*(n.i)
        d = normal.dot(incident)
        reflected = new_vector(incident.get_x() - 2.0 * normal.get_x() * d,
                               incident.get_y() - 2.0 * normal.get_y() * d,
                               incident.get_z() - 2.0 * normal.get_z() * d)

    return reflected
