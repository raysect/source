# Copyright (c) 2014-2018, Dr Alex Meakins, Raysect Project
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


class Term:

    def __init__(self, const, exponents):

        exponents = np.array(exponents, np.uint8)
        if exponents.ndim != 1:
            raise ValueError('Exponents must be a 1D list of integer exponents.')

        self.const = const
        self.exponents = exponents

    def copy(self):
        return Term(self.const, self.exponents)

    def differentiate(self, *orders):

        orders = np.array(orders, dtype=np.uint8)
        if orders.shape != self.exponents.shape:
            raise ValueError('The number of orders must match the number of exponents.')

        term = self.copy()
        for i, order in enumerate(orders):
            self._differentiate(term, i, order)
        return term

    @staticmethod
    def _differentiate(term, i, order):

        if term.const == 0:
            return

        for _ in range(order):

            if term.exponents[i] == 0:
                term.const = 0
                term.exponents[:] = 0
                return

            term.const *= term.exponents[i]
            term.exponents[i] -= 1

    def __repr__(self):
        return '<Term: {}>'.format(self.__str__())

    def __str__(self):

        variables = 'abcdefghijklmnopqrstuvwxyz'

        if self.const == 0:
            return '0'

        s = ''
        for i in range(len(self.exponents)):
            if self.exponents[i] == 1:
                s += '*{}'.format(variables[i], self.exponents[i])
            elif self.exponents[i] > 1:
                s += '*{}^{}'.format(variables[i], self.exponents[i])

        return '{}{}'.format(self.const, s)


class Equation:

    def __init__(self, *terms):
        self.terms = terms

    def differentiate(self, *order):
        terms = []
        for term in self.terms:
            terms.append(term.differentiate(*order))
        return Equation(*terms)

    def __repr__(self):
        return '<Equation: {}>'.format(self.__str__())

    def __str__(self):
        s = 'f = '
        add = False
        all_zero = True
        for term in self.terms:

            if term.const == 0:
                continue

            if add:
                s += ' + '

            s += str(term)
            add = True
            all_zero = False

        if all_zero:
            s += '0'

        return s

    def total_constant(self, *exponents):

        exponents = np.array(exponents, dtype=np.uint8)
        k = 0
        for term in self.terms:
            if (term.exponents == exponents).all():
                k += term.const
        return k




if __name__ == '__main__':

    # print('2D polynomial evaluation:\n')
    # print(polynomial_evaluation_2d())
    # print()
    #
    # print('3D polynomial evaluation:\n')
    # print(polynomial_evaluation_3d())
    # print()
    #

    e = Equation(
        Term(8, (1, 0, 0)),
        Term(20, (0, 1, 0)),
        Term(1, (0, 0, 1)),
        Term(-56, (2, 0, 0)),
        Term(1, (1, 1, 0)),
        Term(1, (1, 0, 1)),
        Term(5, (2, 3, 2)),
        Term(6, (2, 3, 2)),
        Term(7, (2, 3, 2))
    )

    print(e)
    d = e.differentiate(1, 0, 0)
    print(d)

    print(d.total_constant(0,0,0))


