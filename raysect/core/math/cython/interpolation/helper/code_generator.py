# Copyright (c) 2014-2023, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Generates code for the bicubic and tricubic interpolators.
"""

import numpy as np


try:
    from .equation import Term, Equation
except ImportError:
    from equation import Term, Equation


def polynomial_evaluation_2d():

    s = 'v = '
    for i in range(4):
        s += '    ' if i else ''
        for j in range(4):
            x = '*x{}'.format(i) if i else ''
            y = '*y{}'.format(j) if j else ''
            s += 'a[{}][{}]{}{}'.format(i, j, x, y)
            s += ' + ' if i < 3 or j < 3 else ''
        s += '\\\n' if i < 3 else ''
    return s


def polynomial_evaluation_3d():

    s = 'v = '
    for i in range(4):
        s += '    ' if i else ''
        for j in range(4):
            s += '    ' if j else ''
            for k in range(4):
                x = '*x{}'.format(i) if i else ''
                y = '*y{}'.format(j) if j else ''
                z = '*z{}'.format(k) if k else ''
                s += 'a[{}][{}][{}]{}{}{}'.format(i, j, k, x, y, z)
                s += ' + ' if i < 3 or j < 3 or k < 3 else ''
            s += '\\\n' if i < 3 or j < 3 else ''
    return s


def generate_equation_2d():

    terms = []
    for i in range(4):
        for j in range(4):
            terms.append(Term(1, (i, j)))
    return Equation(*terms)


def generate_equation_3d():

    terms = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                terms.append(Term(1, (i, j, k)))
    return Equation(*terms)


def generate_matrix_2d(debug=False):

    f = generate_equation_2d()

    # generate differentials
    dfdx = f.differentiate(1, 0)
    dfdy = f.differentiate(0, 1)

    d2fdxdy = f.differentiate(1, 1)

    # display
    if debug:
        print('f: ' + str(f))
        print()
        print('df/dx: ' + str(dfdx))
        print()
        print('df/dy: ' + str(dfdy))
        print()
        print('d2f/dxdy: ' + str(d2fdxdy))
        print()

    # define each element of input vector
    input_vector = []
    for eqn in (f, dfdx, dfdy, d2fdxdy):
        for i in range(2):
            for j in range(2):
                input_vector.append((eqn, i, j))

    if debug:
        print('input_vector:')
        print(input_vector)
        print()

    # define each element of the output vector
    output_vector = []
    for i in range(4):
        for j in range(4):
            output_vector.append((i, j))

    if debug:
        print('output_vector:')
        print(output_vector)
        print()

    # assemble matrix
    matrix = np.zeros((16, 16))
    u = 0
    for eqn, xi, yi in input_vector:
        v = 0
        for xe, ye in output_vector:
            term = eqn.find(xe, ye)
            if term:
                matrix[u][v] = term.evaluate(xi, yi)
            else:
                matrix[u][v] = 0
            v += 1
        u += 1

    if debug:
        print('uninverted matrix:')
        print(matrix)
        print()

    # invert to obtain final matrix
    return np.linalg.inv(matrix).astype(np.int32)


def generate_matrix_3d(debug=False):

    f = generate_equation_3d()

    # generate differentials
    dfdx = f.differentiate(1, 0, 0)
    dfdy = f.differentiate(0, 1, 0)
    dfdz = f.differentiate(0, 0, 1)

    d2fdxdy = f.differentiate(1, 1, 0)
    d2fdxdz = f.differentiate(1, 0, 1)
    d2fdydz = f.differentiate(0, 1, 1)

    d3fdxdydz = f.differentiate(1, 1, 1)

    # display
    if debug:
        print('f: ' + str(f))
        print()
        print('df/dx: ' + str(dfdx))
        print()
        print('df/dy: ' + str(dfdy))
        print()
        print('df/dz: ' + str(dfdz))
        print()
        print('d2f/dxdy: ' + str(d2fdxdy))
        print()
        print('d2f/dxdz: ' + str(d2fdxdz))
        print()
        print('d2f/dydz: ' + str(d2fdydz))
        print()
        print('d3f/dxdydz: ' + str(d3fdxdydz))
        print()

    # define each element of input vector
    input_vector = []
    for eqn in (f, dfdx, dfdy, dfdz, d2fdxdy, d2fdxdz, d2fdydz, d3fdxdydz):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    input_vector.append((eqn, i, j, k))

    if debug:
        print('input_vector:')
        print(input_vector)
        print()

    # define each element of the output vector
    output_vector = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                output_vector.append((i, j, k))

    if debug:
        print('output_vector:')
        print(output_vector)
        print()

    # assemble matrix
    matrix = np.zeros((64, 64))
    u = 0
    for eqn, xi, yi, zi in input_vector:
        v = 0
        for xe, ye, ze in output_vector:
            term = eqn.find(xe, ye, ze)
            if term:
                matrix[u][v] = term.evaluate(xi, yi, zi)
            else:
                matrix[u][v] = 0
            v += 1
        u += 1

    if debug:
        print('uninverted matrix:')
        print(matrix)
        print()

    # invert to obtain final matrix
    return np.linalg.inv(matrix).astype(np.int32)


def generate_matmul_code_2d(debug=False):

    max_terms = 4
    term_justify = 0

    # define each element of input vector
    input_vector = []
    for eqn in ('f', 'dfdx', 'dfdy', 'd2fdxdy'):
        for i in range(2):
            for j in range(2):
                input_vector.append((eqn, i, j))

    # define each element of the output vector
    output_vector = []
    for i in range(4):
        for j in range(4):
            output_vector.append((i, j))

    matrix = generate_matrix_2d()

    s = ''
    u = 0
    for xe, ye in output_vector:
        s += 'a[{}][{}] ='.format(xe, ye)
        v = 0
        add = False
        n_terms = 0
        for eqn, xi, yi in input_vector:
            m = matrix[u][v]
            if m != 0:

                # operator
                if add:
                    if m > 0:
                        s += ' + '
                    else:
                        s += ' - '
                        m = -m
                else:
                    if m > 0:
                        s += '   '
                    else:
                        s += ' - '
                        m = -m

                # constant
                t = ''
                if m != 1:
                    t += '{}*'.format(m)

                # array access
                t += '{}[{}][{}]'.format(eqn, xi, yi)
                t = t.rjust(term_justify)

                s += t

                add = True
                n_terms += 1
                if n_terms == max_terms:
                    s += ' \\\n         '
                    n_terms = 0

            v += 1

        s += '\n'
        u += 1

    return s


def generate_matmul_code_3d(debug=False):

    max_terms = 4
    term_justify = 0

    # define each element of input vector
    input_vector = []
    for eqn in ('f', 'dfdx', 'dfdy', 'dfdz', 'd2fdxdy', 'd2fdxdz', 'd2fdydz', 'd3fdxdydz'):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    input_vector.append((eqn, i, j, k))

    # define each element of the output vector
    output_vector = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                output_vector.append((i, j, k))

    matrix = generate_matrix_3d()

    s = ''
    u = 0
    for xe, ye, ze in output_vector:
        s += 'a[{}][{}][{}] ='.format(xe, ye, ze)
        v = 0
        add = False
        n_terms = 0
        for eqn, xi, yi, zi in input_vector:
            m = matrix[u][v]
            if m != 0:

                # operator
                if add:
                    if m > 0:
                        s += ' + '
                    else:
                        s += ' - '
                        m = -m
                else:
                    if m > 0:
                        s += '   '
                    else:
                        s += ' - '
                        m = -m

                # constant
                t = ''
                if m != 1:
                    t += '{}*'.format(m)

                # array access
                t += '{}[{}][{}][{}]'.format(eqn, xi, yi, zi)
                t = t.rjust(term_justify)

                s += t

                add = True
                n_terms += 1
                if n_terms == max_terms:
                    s += ' \\\n            '
                    n_terms = 0

            v += 1

        s += '\n'
        u += 1

    return s




if __name__ == '__main__':

    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    print('2D polynomial equation:\n')
    print(generate_equation_2d())
    print()

    print('2D matrix:\n')
    print(generate_matrix_2d(debug=False))
    print()

    print('2D matrix multiplication code:\n')
    print(generate_matmul_code_2d(True))
    print()

    print('2D polynomial evaluation code:\n')
    print(polynomial_evaluation_2d())
    print()

    print('3D polynomial equation:\n')
    print(generate_equation_3d())
    print()

    print('3D matrix:\n')
    print(generate_matrix_3d(debug=False))
    print()

    print('3D polynomial evaluation code:\n')
    print(polynomial_evaluation_3d())
    print()

    print('3D matrix multiplication code:\n')
    print(generate_matmul_code_3d(True))
    print()
