# Copyright (c) 2014-2026, Dr Alex Meakins, Raysect Project
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

import unittest
from raysect.core.workflow import SerialEngine, MulticoreEngine, MPIEngine, HybridEngine
from raysect.core.workflow.mpi import HAVE_MPI


if HAVE_MPI:
    from mpi4py import MPI
    MPI_WORLD_SIZE = MPI.COMM_WORLD.size
else:
    MPI_WORLD_SIZE = 0


class TestWorkflow(unittest.TestCase):
    """Test the render engines."""

    def setUp(self):
        self.total = 0
        # A list of numbers [1, N] to perform a weighted sum of.
        self.N = 100
        self.numbers = list(range(1, self.N + 1))
        # Each task represents an index into the list of numbers.
        self.tasks = [i for i, _ in enumerate(self.numbers)]

    def render(self, task, weight=1):
        index = task
        number = self.numbers[index]
        scaled = number * weight
        return scaled

    def update(self, result, neg=False):
        if neg:
            result = -result
        self.total += result

    def test_serial(self):
        engine = SerialEngine()
        engine.run(self.tasks, self.render, self.update)
        self.assertEqual(self.total, self.N * (self.N + 1) / 2)

    def test_serial_args(self):
        engine = SerialEngine()
        engine.run(self.tasks, self.render, self.update, render_args=(2,), update_args=(True,))
        self.assertEqual(self.total, -self.N * (self.N + 1))

    def test_serial_kwargs(self):
        engine = SerialEngine()
        engine.run(self.tasks, self.render, self.update, render_kwargs={'weight': 2}, update_args={'neg': True})
        self.assertEqual(self.total, -self.N * (self.N + 1))

    def test_multicore(self):
        engine = MulticoreEngine(2)
        engine.run(self.tasks, self.render, self.update)
        self.assertEqual(self.total, self.N * (self.N + 1) / 2)

    def test_multicore_args(self):
        engine = MulticoreEngine()
        engine.run(self.tasks, self.render, self.update, render_args=(2,), update_args=(True,))
        self.assertEqual(self.total, -self.N * (self.N + 1))

    def test_multicore_kwargs(self):
        engine = MulticoreEngine()
        engine.run(self.tasks, self.render, self.update, render_kwargs={'weight': 2}, update_args={'neg': True})
        self.assertEqual(self.total, -self.N * (self.N + 1))

    @unittest.skipUnless(HAVE_MPI and MPI_WORLD_SIZE > 1, "Requires MPI and running with mpirun -np >=2.")
    def test_mpi(self):
        engine = MPIEngine()
        engine.run(self.tasks, self.render, self.update)
        if engine.rank == 0:
            self.assertEqual(self.total, self.N * (self.N + 1) / 2)

    @unittest.skipUnless(HAVE_MPI and MPI_WORLD_SIZE > 1, "Requires MPI and running with mpirun -np >=2.")
    def test_mpi_args(self):
        engine = MPIEngine()
        engine.run(self.tasks, self.render, self.update, render_args=(2,), update_args=(True,))
        if engine.rank == 0:
            self.assertEqual(self.total, -self.N * (self.N + 1))

    @unittest.skipUnless(HAVE_MPI and MPI_WORLD_SIZE == 1, "Requires MPI and running with mpirun -np 1.")
    def test_mpi_disallow_1proc_world(self):
        with self.assertRaises(RuntimeError):
            MPIEngine()

    @unittest.skipUnless(HAVE_MPI and MPI_WORLD_SIZE > 1, "Requires MPI and running with mpirun -np >=2.")
    def test_hybrid_serial(self):
        engine = HybridEngine(SerialEngine())
        engine.run(self.tasks, self.render, self.update)
        if engine.rank == 0:
            self.assertEqual(self.total, self.N * (self.N + 1) / 2)

    @unittest.skipUnless(HAVE_MPI and MPI_WORLD_SIZE > 1, "Requires MPI and running with mpirun -np >=2.")
    def test_hybrid_serial_args(self):
        engine = HybridEngine(SerialEngine())
        engine.run(self.tasks, self.render, self.update, render_args=(2,), update_args=(True,))
        if engine.rank == 0:
            self.assertEqual(self.total, -self.N * (self.N + 1))

    @unittest.skipUnless(HAVE_MPI and MPI_WORLD_SIZE > 1, "Requires MPI and running with mpirun -np >=2.")
    def test_hybrid_multicore(self):
        engine = HybridEngine(MulticoreEngine(2))
        engine.run(self.tasks, self.render, self.update)
        if engine.rank == 0:
            self.assertEqual(self.total, self.N * (self.N + 1) / 2)

    @unittest.skipUnless(HAVE_MPI and MPI_WORLD_SIZE > 1, "Requires MPI and running with mpirun -np >=2.")
    def test_hybrid_multicore_args(self):
        engine = HybridEngine(MulticoreEngine(2))
        engine.run(self.tasks, self.render, self.update, render_args=(2,), update_args=(True,))
        if engine.rank == 0:
            self.assertEqual(self.total, -self.N * (self.N + 1))

