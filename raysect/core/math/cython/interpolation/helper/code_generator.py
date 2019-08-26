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





if __name__ == '__main__':

    print('2D polynomial evaluation:\n')
    print(polynomial_evaluation_2d())
    print()

    print('3D polynomial evaluation:\n')
    print(polynomial_evaluation_3d())
    print()


