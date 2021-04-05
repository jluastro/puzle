import os
import time
import numpy as np
import logging
import glob
from pathlib import Path

from puzle.utils import return_data_dir, \
    execute, get_logger

logger = get_logger(__name__)
logging.getLogger('filelock').setLevel(logging.WARNING)


def identify_is_nersc():
    for key in os.environ.keys():
        if 'NERSC' in key:
            return True
    return False


def fetch_db_id():
    slurm_job_id = os.getenv('SLURM_JOB_ID', 0)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    db_id = '%s.%s' % (slurm_job_id, rank)
    return db_id


def load_db_ids():
    ulensdb_folder = '%s/ulensdb' % return_data_dir()

    # load db_ids from disk
    files = glob.glob(f'{ulensdb_folder}/*con')
    db_ids = set([os.path.basename(f).replace('.con', '') for f in files])

    if identify_is_nersc():
        # remove rows that are not currently running
        stdout, _ = execute('squeue --noheader -u mmedford --format="%i')
        job_ids = set([s.replace('"', '') for s in stdout.decode().split('\n')])
        db_ids_to_delete = set([d for d in db_ids if d.split('.')[0] not in job_ids])
        for db_id in db_ids_to_delete:
            os.remove(f'{ulensdb_folder}/{db_id}.con')
        db_ids = set([d for d in db_ids if d.split('.')[0] in job_ids])

    return db_ids


def remove_db_id():
    my_db_id = fetch_db_id()
    if my_db_id is None:
        logger.debug(f'{my_db_id}: Skipping remove_db for local process')
        return

    ulensdb_file = '%s/ulensdb/%s.con' % (return_data_dir(), my_db_id)
    os.remove(ulensdb_file)

    logger.debug(f'{my_db_id}: Delete success')


def insert_db_id(num_ids=2, retry_time=10):
    my_db_id = fetch_db_id()
    if my_db_id is None:
        logger.debug(f'{my_db_id}: Skipping insert_db for local process')
        return

    rank = int(my_db_id.split('.')[1])
    pid = os.getpid()
    np.random.seed(rank + pid)

    ulensdb_folder = '%s/ulensdb' % return_data_dir()
    ulensdb_file = f'{ulensdb_folder}/{my_db_id}.con'

    successFlag = False
    while True:
        time.sleep(retry_time + abs(np.random.normal(scale=.5 * retry_time)))
        db_ids = load_db_ids()
        num_db_ids = len(db_ids)
        logger.debug(f'{my_db_id}: Attempting insert to {ulensdb_folder} | {num_db_ids} db_ids | {db_ids}')
        if num_db_ids < num_ids:
            Path(ulensdb_file).touch()
            successFlag = True

        if successFlag:
            logger.debug(f'{my_db_id}: Insert success')
            return
        else:
            logger.debug(f'{my_db_id}: Insert fail, retry in ~{retry_time} seconds')
