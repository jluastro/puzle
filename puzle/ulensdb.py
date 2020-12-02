import os
import time
from filelock import FileLock
from pathlib import Path


def fetch_db_id():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    db_id = '%s.%s' % (slurm_job_id, rank)
    return db_id


def load_db_ids():
    file_path = '/global/cscratch1/sd/mmedford/puzle/ulensdb.txt'
    if not os.path.exists(file_path):
        Path(file_path).touch()

    lines = open(file_path, 'r').readlines()
    db_ids = set([l.replace('\n', '') for l in lines])
    return db_ids


def remove_db_id():
    file_path = '/global/cscratch1/sd/mmedford/puzle/ulensdb.txt'
    lock_path = file_path.replace('.txt', '.lock')
    lock = FileLock(lock_path)

    with lock:
        db_ids = load_db_ids()
        my_db_id = fetch_db_id()
        db_ids.remove(my_db_id)

        with open(file_path, 'w') as f:
            for db_id in list(db_ids):
                f.write('%s\n' % db_id)


def insert_db_id(num_ids=50, retry_time=30):
    file_path = '/global/cscratch1/sd/mmedford/puzle/ulensdb.txt'
    lock_path = file_path.replace('.txt', '.lock')
    lock = FileLock(lock_path)

    successFlag = False
    while True:
        with lock:
            db_ids = load_db_ids()
            my_db_id = fetch_db_id()

            if len(db_ids) < num_ids:
                db_ids.add(my_db_id)

                with open(file_path, 'w') as f:
                    for db_id in list(db_ids):
                        f.write('%s\n' % db_id)

                successFlag = True

        if successFlag:
            return
        else:
            time.sleep(retry_time)
