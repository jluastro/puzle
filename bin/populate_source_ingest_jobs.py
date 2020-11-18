#! /usr/bin/env python
"""
populate_source_ingest_jobs.py
"""

import glob

from puzle.models import SourceIngestJob
from puzle import db

n_objects_max = 100000


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


if __name__ == '__main__':

    object_files = glob.glob('field*objects')
    object_files.sort()

    for i, object_file in enumerate(object_files):
        print('Processing %s (%i/%i)' % (object_file, i, len(object_files)))
        lightcurve_filename = object_file.replace('.objects', '.txt')
        n_objects = file_len(object_file)
        process_size = (n_objects // n_objects_max) + 1

        for process_rank in range(process_size):
            sourceIngestJob = SourceIngestJob(
                lightcurve_filename=lightcurve_filename,
                process_rank=process_rank,
                process_size=process_size)
            db.session.add(sourceIngestJob)
        db.session.commit()

        print(f'-- {process_size} jobs added to source_ingest_job table')
