import os
import time
import numpy as np
from filelock import FileLock
from pathlib import Path
import logging

from puzle.utils import execute, get_logger

logger = get_logger(__name__)
ulensdb_file_path = os.getenv('ULENS_DB_FILEPATH')
logging.getLogger('filelock').setLevel(logging.WARNING)


def identify_is_nersc():
    for key in os.environ.keys():
        if 'NERSC' in key:
            return True
    return False


def fetch_db_id():
    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
    else:
        rank = 0
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if slurm_job_id is None:
        return None
    db_id = '%s.%s' % (slurm_job_id, rank)
    return db_id


def load_db_ids():
    # create file if does not exist
    if not os.path.exists(ulensdb_file_path):
        Path(ulensdb_file_path).touch()

    # load db_ids from disk
    lines = open(ulensdb_file_path, 'r').readlines()
    db_ids = set([l.replace('\n', '') for l in lines])

    if identify_is_nersc():
        # remove rows that are not currently running
        stdout, _ = execute('squeue --noheader -u mmedford --format="%i')
        job_ids = set([s.replace('"', '') for s in stdout.decode().split('\n')])
        db_ids = set([d for d in db_ids if d.split('.')[0] in job_ids])

    return db_ids


def remove_db_id():
    lock_path = ulensdb_file_path.replace('.txt', '.lock')
    lock = FileLock(lock_path)

    my_db_id = fetch_db_id()
    if my_db_id is None:
        logger.debug(f'{my_db_id}: Skipping remove_db for local process')
        return

    logger.debug(f'{my_db_id}: Attempting delete from {ulensdb_file_path}')
    with lock:
        db_ids = load_db_ids()
        logger.debug(f'{my_db_id}: db_ids loaded {db_ids}')
        db_ids.remove(my_db_id)

        with open(ulensdb_file_path, 'w') as f:
            for db_id in list(db_ids):
                f.write('%s\n' % db_id)

    logger.debug(f'{my_db_id}: Delete success')


def insert_db_id(num_ids=50):
    lock_path = ulensdb_file_path.replace('.txt', '.lock')
    lock = FileLock(lock_path)

    my_db_id = fetch_db_id()
    if my_db_id is None:
        logger.debug(f'{my_db_id}: Skipping insert_db for local process')
        return

    successFlag = False
    retry_time = 1
    retry_multiplier = 2
    retry_time_max = 60
    while True:
        time.sleep(abs(np.random.normal(scale=retry_time)))
        logger.debug(f'{my_db_id}: Attempting insert to {ulensdb_file_path}')
        with lock:
            logger.debug(f'{my_db_id}: Lock acquired')
            db_ids = load_db_ids()
            if len(db_ids) < num_ids:
                db_ids.add(my_db_id)

                with open(ulensdb_file_path, 'w') as f:
                    for db_id in list(db_ids):
                        f.write('%s\n' % db_id)

                successFlag = True

        if successFlag:
            logger.debug(f'{my_db_id}: Insert success')
            return
        else:
            retry_time = min(retry_time_max, retry_time * retry_multiplier)
            logger.debug(f'{my_db_id}: Insert fail, retry in {retry_time} seconds')
            time.sleep(retry_time)
            retry_multiplier *= 2
