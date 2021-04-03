import os
from puzle.utils import execute
import logging
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

logger = logging.getLogger(__name__)

logger.info('Hello World %i of %i' % (rank, size))

os.environ['PATH'] += os.pathsep + '/global/common/shared/das/container_proxy'

if rank == 0:
    stdout, _ = execute('squeue --noheader -u mmedford --format="%i')
    logger.debug(stdout)