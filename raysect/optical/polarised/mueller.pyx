from raysect.core.math._mat4 cimport _Mat4


# todo: write docstring
cdef class MuellerMatrix(_Mat4):
    """

    """

    def __repr__(self):
        """String representation."""

        cdef int i, j

        s = "MuellerMatrix(["
        for i in range(0, 4):
            s += "["
            for j in range(0, 4):
                s += str(self.m[i][j])
                if j < 3:
                    s += ", "
            s += "]"
            if i < 3:
                s += ", "
        return s + "])"

    def __mul__(object x, object y):
        """Multiplication operator.
        """

        cdef MuellerMatrix mx, my

        if isinstance(x, MuellerMatrix) and isinstance(y, MuellerMatrix):

            mx = <MuellerMatrix>x
            my = <MuellerMatrix>y
            return new_muellermatrix(
                mx.m[0][0] * my.m[0][0] + mx.m[0][1] * my.m[1][0] + mx.m[0][2] * my.m[2][0] + mx.m[0][3] * my.m[3][0],
                mx.m[0][0] * my.m[0][1] + mx.m[0][1] * my.m[1][1] + mx.m[0][2] * my.m[2][1] + mx.m[0][3] * my.m[3][1],
                mx.m[0][0] * my.m[0][2] + mx.m[0][1] * my.m[1][2] + mx.m[0][2] * my.m[2][2] + mx.m[0][3] * my.m[3][2],
                mx.m[0][0] * my.m[0][3] + mx.m[0][1] * my.m[1][3] + mx.m[0][2] * my.m[2][3] + mx.m[0][3] * my.m[3][3],
                mx.m[1][0] * my.m[0][0] + mx.m[1][1] * my.m[1][0] + mx.m[1][2] * my.m[2][0] + mx.m[1][3] * my.m[3][0],
                mx.m[1][0] * my.m[0][1] + mx.m[1][1] * my.m[1][1] + mx.m[1][2] * my.m[2][1] + mx.m[1][3] * my.m[3][1],
                mx.m[1][0] * my.m[0][2] + mx.m[1][1] * my.m[1][2] + mx.m[1][2] * my.m[2][2] + mx.m[1][3] * my.m[3][2],
                mx.m[1][0] * my.m[0][3] + mx.m[1][1] * my.m[1][3] + mx.m[1][2] * my.m[2][3] + mx.m[1][3] * my.m[3][3],
                mx.m[2][0] * my.m[0][0] + mx.m[2][1] * my.m[1][0] + mx.m[2][2] * my.m[2][0] + mx.m[2][3] * my.m[3][0],
                mx.m[2][0] * my.m[0][1] + mx.m[2][1] * my.m[1][1] + mx.m[2][2] * my.m[2][1] + mx.m[2][3] * my.m[3][1],
                mx.m[2][0] * my.m[0][2] + mx.m[2][1] * my.m[1][2] + mx.m[2][2] * my.m[2][2] + mx.m[2][3] * my.m[3][2],
                mx.m[2][0] * my.m[0][3] + mx.m[2][1] * my.m[1][3] + mx.m[2][2] * my.m[2][3] + mx.m[2][3] * my.m[3][3],
                mx.m[3][0] * my.m[0][0] + mx.m[3][1] * my.m[1][0] + mx.m[3][2] * my.m[2][0] + mx.m[3][3] * my.m[3][0],
                mx.m[3][0] * my.m[0][1] + mx.m[3][1] * my.m[1][1] + mx.m[3][2] * my.m[2][1] + mx.m[3][3] * my.m[3][1],
                mx.m[3][0] * my.m[0][2] + mx.m[3][1] * my.m[1][2] + mx.m[3][2] * my.m[2][2] + mx.m[3][3] * my.m[3][2],
                mx.m[3][0] * my.m[0][3] + mx.m[3][1] * my.m[1][3] + mx.m[3][2] * my.m[2][3] + mx.m[3][3] * my.m[3][3]
            )

        return NotImplemented

    cdef MuellerMatrix mul(self, MuellerMatrix m):

        return new_muellermatrix(
            self.m[0][0] * m.m[0][0] + self.m[0][1] * m.m[1][0] + self.m[0][2] * m.m[2][0] + self.m[0][3] * m.m[3][0],
            self.m[0][0] * m.m[0][1] + self.m[0][1] * m.m[1][1] + self.m[0][2] * m.m[2][1] + self.m[0][3] * m.m[3][1],
            self.m[0][0] * m.m[0][2] + self.m[0][1] * m.m[1][2] + self.m[0][2] * m.m[2][2] + self.m[0][3] * m.m[3][2],
            self.m[0][0] * m.m[0][3] + self.m[0][1] * m.m[1][3] + self.m[0][2] * m.m[2][3] + self.m[0][3] * m.m[3][3],
            self.m[1][0] * m.m[0][0] + self.m[1][1] * m.m[1][0] + self.m[1][2] * m.m[2][0] + self.m[1][3] * m.m[3][0],
            self.m[1][0] * m.m[0][1] + self.m[1][1] * m.m[1][1] + self.m[1][2] * m.m[2][1] + self.m[1][3] * m.m[3][1],
            self.m[1][0] * m.m[0][2] + self.m[1][1] * m.m[1][2] + self.m[1][2] * m.m[2][2] + self.m[1][3] * m.m[3][2],
            self.m[1][0] * m.m[0][3] + self.m[1][1] * m.m[1][3] + self.m[1][2] * m.m[2][3] + self.m[1][3] * m.m[3][3],
            self.m[2][0] * m.m[0][0] + self.m[2][1] * m.m[1][0] + self.m[2][2] * m.m[2][0] + self.m[2][3] * m.m[3][0],
            self.m[2][0] * m.m[0][1] + self.m[2][1] * m.m[1][1] + self.m[2][2] * m.m[2][1] + self.m[2][3] * m.m[3][1],
            self.m[2][0] * m.m[0][2] + self.m[2][1] * m.m[1][2] + self.m[2][2] * m.m[2][2] + self.m[2][3] * m.m[3][2],
            self.m[2][0] * m.m[0][3] + self.m[2][1] * m.m[1][3] + self.m[2][2] * m.m[2][3] + self.m[2][3] * m.m[3][3],
            self.m[3][0] * m.m[0][0] + self.m[3][1] * m.m[1][0] + self.m[3][2] * m.m[2][0] + self.m[3][3] * m.m[3][0],
            self.m[3][0] * m.m[0][1] + self.m[3][1] * m.m[1][1] + self.m[3][2] * m.m[2][1] + self.m[3][3] * m.m[3][1],
            self.m[3][0] * m.m[0][2] + self.m[3][1] * m.m[1][2] + self.m[3][2] * m.m[2][2] + self.m[3][3] * m.m[3][2],
            self.m[3][0] * m.m[0][3] + self.m[3][1] * m.m[1][3] + self.m[3][2] * m.m[2][3] + self.m[3][3] * m.m[3][3]
        )
