# cython: language_level=3

# Copyright (c) 2015, Dr Alex Meakins, Raysect Project
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

from .mesh import Mesh, Triangle
from raysect.core import Point
import struct
import os

STL_AUTOMATIC = 0
STL_ASCII = 1
STL_BINARY = 2

#: Amount of bytes to read while using buffered reading
BUFFER_SIZE = 4096
#: The amount of bytes in the header field
HEADER_SIZE = 80
#: The amount of bytes in the count field
COUNT_SIZE = 4


class STLHandler:

    @classmethod
    def import_stl(cls, filename, scaling=1.0, mode=STL_AUTOMATIC, **kwargs):

        if mode == STL_AUTOMATIC:
            pass
        elif mode == STL_ASCII:
            triangles = cls._load_ascii(filename, scaling)
        elif mode == STL_BINARY:
            triangles = cls._load_binary(filename, scaling)

        return Mesh(triangles, smoothing=False, **kwargs)

    @classmethod
    def _load_ascii(cls, filename, scaling):

        with open(filename, 'r') as fh:
            line = cls._get_ascii_line(fh)
            if not line.startswith('solid'):
                raise ValueError('ASCII STL files should start with solid <space>. The application that produced this STL '
                                 'file may be faulty, please report this error. The erroneous line: {}'.format(line))
            return list(cls._ascii_read_triangle(fh, scaling))

    @classmethod
    def _ascii_read_triangle(cls, fh, scaling):

        while True:
            try:
                _ = cls._get_ascii_line(fh, 'facet normal')
                assert cls._get_ascii_line(fh) == 'outer loop'
                v0 = cls._get_ascii_line(fh, 'vertex')
                v1 = cls._get_ascii_line(fh, 'vertex')
                v2 = cls._get_ascii_line(fh, 'vertex')
                assert cls._get_ascii_line(fh) == 'endloop'
                assert cls._get_ascii_line(fh) == 'endfacet'
                yield Triangle(
                    Point(scaling * v0[0], scaling * v0[1], scaling * v0[2]),
                    Point(scaling * v1[0], scaling * v1[1], scaling * v1[2]),
                    Point(scaling * v2[0], scaling * v2[1], scaling * v2[2])
                )
            except AssertionError as e:
                raise RuntimeError(e)
            except StopIteration:
                break

    @classmethod
    def _get_ascii_line(cls, fh, prefix=''):

        # get next line, remove whitespace i.e. indentation, newline character, etc
        line = fh.readline().lower().strip()
        if prefix:
            if line.startswith(prefix):
                values = line.replace(prefix, '', 1).strip().split()
            elif line.startswith('endsolid'):
                raise StopIteration()
            else:
                raise RuntimeError('{} should start with {}'.format(line, prefix))

            if len(values) == 3:
                return [float(v) for v in values]
            else:
                raise RuntimeError('Incorrect number of values in line: {}'.format(line))
        else:
            return line

    # @classmethod
    # def _load_binary(cls, filename, check_size=True, scaling):
    #
    #     with open(filename, 'rb') as fh:
    #         header = fh.read(HEADER_SIZE).lower()
    #         solid_name = header[5:].strip()
    #
    #         count, = struct.unpack('@i', fh.read(COUNT_SIZE))
    #         assert count < MAX_COUNT, ('File too large, got {} triangles which exceeds the maximum of {}'
    #                                    ''.format(count, MAX_COUNT))
    #
    #         if check_size:
    #             # Check the size of the file
    #             fh.seek(0, os.SEEK_END)
    #             raw_size = fh.tell() - HEADER_SIZE - COUNT_SIZE
    #             expected_count = raw_size / cls.dtype.itemsize
    #             if expected_count != count:
    #                 raise ValueError('Expected {} vectors but header indicates {}. '
    #                                  'Is file invalid?'.format(expected_count, count))
    #             #
    #             fh.seek(HEADER_SIZE + COUNT_SIZE)
    #
    #         # Read the rest of the binary data
    #         meshdata = numpy.fromfile(fh, dtype=cls.dtype, count=count)
    #
    #         return solid_name, meshdata


import_stl = STLHandler.import_stl




