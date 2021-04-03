import os
import time
import numpy as np
from pathlib import Path
import logging

from puzle.utils import return_data_dir, \
    execute, identify_is_nersc

logger = logging.getLogger(__name__)


def fetch_db_id():
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if slurm_job_id is None:
        return None
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    db_id = '%s.%s' % (slurm_job_id, rank)
    return db_id


def load_db_ids():
    # create file if does not exist
    ulensdb_file_path = '%s/ulensdb/ulensdb.txt' % return_data_dir()
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


def remove_db_id(retry_time=5):
    ulensdb_file_path = '%s/ulensdb/ulensdb.txt' % return_data_dir()
    lock_path = ulensdb_file_path.replace('.txt', '.lock')

    my_db_id = fetch_db_id()
    if my_db_id is None:
        logger.debug(f'{my_db_id}: Skipping remove_db for local process')
        return

    successFlag = False
    while True:
        time.sleep(abs(np.random.normal(scale=.01 * retry_time)))
        logger.debug(f'{my_db_id}: Attempting delete from {ulensdb_file_path}')
        if not os.path.exists(lock_path):
            Path(lock_path).touch()

            db_ids = load_db_ids()
            logger.debug(f'{my_db_id}: db_ids loaded {db_ids}')
            db_ids.remove(my_db_id)

            with open(ulensdb_file_path, 'w') as f:
                for db_id in list(db_ids):
                    f.write('%s\n' % db_id)

            if os.path.exists(lock_path):
                os.remove(lock_path)

            successFlag = True

        if successFlag:
            logger.debug(f'{my_db_id}: Delete success')
            return
        else:
            logger.debug(f'{my_db_id}: Delete fail, retry in {retry_time} seconds')
            time.sleep(retry_time)


def insert_db_id(num_ids=50, retry_time=5):
    ulensdb_file_path = '%s/ulensdb/ulensdb.txt' % return_data_dir()
    lock_path = ulensdb_file_path.replace('.txt', '.lock')

    my_db_id = fetch_db_id()
    if my_db_id is None:
        logger.debug(f'{my_db_id}: Skipping insert_db for local process')
        return

    successFlag = False
    while True:
        time.sleep(abs(np.random.normal(scale=.01 * retry_time)))
        logger.debug(f'{my_db_id}: Attempting insert to {ulensdb_file_path}')
        if not os.path.exists(lock_path):
            Path(lock_path).touch()
            db_ids = load_db_ids()
            if len(db_ids) < num_ids:
                db_ids.add(my_db_id)

                with open(ulensdb_file_path, 'w') as f:
                    for db_id in list(db_ids):
                        f.write('%s\n' % db_id)

                successFlag = True
            if os.path.exists(lock_path):
                os.remove(lock_path)

        if successFlag:
            logger.debug(f'{my_db_id}: Insert success')
            return
        else:
            logger.debug(f'{my_db_id}: Insert fail, retry in {retry_time} seconds')
            time.sleep(retry_time)

