from puzle.utils import execute, get_logger
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

logger = get_logger(__name__)

print('A: Hello World %i of %i' % (rank, size))
logger.info('B: Hello World %i of %i' % (rank, size))

if rank == 0:
    stdout, _ = execute('squeue --noheader -u mmedford --format="%i')
    logger.debug(stdout)
