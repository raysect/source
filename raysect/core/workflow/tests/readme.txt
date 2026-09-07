This directory contains tests for the render engines.
The Serial and MultiCore render engine tests can be run with the standard test command `./dev/test.sh`.
However, the MPI-based render engines (MPIEngine and HybridEngine) require an MPI runtime for proper testing.
They are therefore skipped if this is not available.

To run the MPI render engine tests, it's necessary to run the test suite using MPI. This has 2 prerequisites:

* Install the mpi4py Python package and an MPI runtime (impi-rt is a suitable manylinux runtime for Intel processors).
* Run the test suite using `mpirun -np 2 ./dev/test.sh -k mpi -k hybrid`.

It's recommended to use the `-k mpi -k hybrid` filters to only run the tests requiring MPI: all other tests should be run separately.
Note that when using `mpirun -np 2` there will be 2 outputs for every test.
It's therefore best to limit to 2 processes to minimise the duplicated output: 2 is the minimum number required to test the MPI functionality properly.
By limiting to only the MPI tests the amount of duplicated output will be further reduced to only these tests.
