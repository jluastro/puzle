import os
from puzle.utils import execute
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

print('Hello World %i of %i' % (rank, size))

os.environ['PATH'] += os.pathsep + '/global/common/shared/das/container_proxy'

if rank == 0:
    stdout, _ = execute('squeue --noheader -u mmedford --format="%i')
    print(stdout)