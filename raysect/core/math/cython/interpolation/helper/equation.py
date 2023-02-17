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
A simple equation differential calculator suitable for polynomial calculations.
"""

import numpy as np


class Term:

    def __init__(self, const, exponents, coeff=None):

        exponents = np.array(exponents, np.uint8)
        if exponents.ndim != 1:
            raise ValueError('Exponents must be a 1D list of integer exponents.')

        self.coeff = coeff or tuple(exponents)
        self.const = const
        self.exponents = exponents

    def copy(self):
        return Term(self.const, self.exponents, self.coeff)

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

    def evaluate(self, *values):
        values = np.array(values, dtype=np.double)
        r = 1
        for i in range(len(self.exponents)):
            r *= values[i] ** self.exponents[i]
        return self.const * r

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

        if not terms:
            raise ValueError('At least one term must be supplied.')

        variables = len(terms[0].exponents)
        for term in terms:
            if len(term.exponents) != variables:
                raise ValueError('All terms must have the same number of exponents.')

        self.variables = variables
        self.terms = terms

    def differentiate(self, *order):
        terms = []
        for term in self.terms:
            terms.append(term.differentiate(*order))
        return Equation(*terms)

    def find(self, *coeff):
        coeff = tuple(coeff)
        for term in self.terms:
            if term.coeff == coeff:
                return term
        return None

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
