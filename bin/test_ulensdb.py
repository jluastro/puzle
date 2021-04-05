#! /usr/bin/env python
"""
test_ulensdb.py
"""

import numpy as np
import time
from puzle.ulensdb import insert_db_id, remove_db_id
from puzle.utils import get_logger

logger = get_logger(__name__)

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logger.info(f'{rank}: Before sleep')
np.random.seed(rank)
time.sleep(abs(np.random.normal()))
logger.info(f'{rank}: After sleep')

insert_db_id()
time.sleep(10)
remove_db_id()
