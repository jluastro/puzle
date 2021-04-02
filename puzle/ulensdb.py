import os
import time
import numpy as np
from filelock import FileLock
from pathlib import Path
import logging

from puzle.utils import return_data_dir

logger = logging.getLogger(__name__)
# logging.getLogger('filelock').setLevel(logging.WARNING)


def identify_is_nersc():
    for key in os.environ.keys():
        if 'NERSC' in key:
            return True
    return False


def fetch_db_id():
    pid = os.getpid()
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if slurm_job_id is None:
        return None
    db_id = '%s.%s' % (slurm_job_id, pid)
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
        lines = open('/global/homes/m/mmedford/puzle/data/ulensdb/job_ids.txt', 'r').readlines()[1:]
        job_ids = set([l.replace('\n', '') for l in lines])
        db_ids = set([d for d in db_ids if d.split('.')[0] in job_ids])

    return db_ids


def remove_db_id():
    ulensdb_file_path = '%s/ulensdb/ulensdb.txt' % return_data_dir()
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


def insert_db_id(num_ids=50, retry_time=5):
    ulensdb_file_path = '%s/ulensdb/ulensdb.txt' % return_data_dir()
    lock_path = ulensdb_file_path.replace('.txt', '.lock')
    lock = FileLock(lock_path)

    my_db_id = fetch_db_id()
    if my_db_id is None:
        logger.debug(f'{my_db_id}: Skipping insert_db for local process')
        return

    successFlag = False
    while True:
        time.sleep(abs(np.random.normal(scale=.01 * retry_time)))
        logger.debug(f'{my_db_id}: Attempting insert to {ulensdb_file_path}')
        with lock:
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
            logger.debug(f'{my_db_id}: Insert fail, retry in {retry_time} seconds')
            time.sleep(retry_time)
