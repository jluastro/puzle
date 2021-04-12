#! /usr/bin/env python
"""
export_sources.py
"""

import os
import numpy as np
import glob
from puzle import db
from puzle.models import SourceIngestJob, Source


def source_to_csv_line(source, source_id):
    line = '%s_%s,' % (source.ingest_job_id, source_id)
    line += '%s,' % str(source.object_id_g)
    line += '%s,' % str(source.object_id_r)
    line += '%s,' % str(source.object_id_i)
    line += '%s,' % str(source.lightcurve_position_g)
    line += '%s,' % str(source.lightcurve_position_r)
    line += '%s,' % str(source.lightcurve_position_i)
    line += '%s,' % source.lightcurve_filename
    line += '%s,' % source.ra
    line += '%s,' % source.dec
    line += '%s' % source.ingest_job_id
    return line


def export_sources(job_id, source_list):

    dir = 'sources_%s' % str(job_id)[:3]

    if not os.path.exists(dir):
        os.makedirs(dir)

    source_exported = []
    fname = f'{dir}/sources.{job_id:06}.txt'
    with open(fname, 'w') as f:
        header = 'id_str,'
        header += 'object_id_g,'
        header += 'object_id_r,'
        header += 'object_id_i,'
        header += 'lightcurve_position_g,'
        header += 'lightcurve_position_r,'
        header += 'lightcurve_position_i,'
        header += 'lightcurve_filename,'
        header += 'ra,'
        header += 'dec,'
        header += 'ingest_job_id'
        f.write(f'{header}\n')

        source_keys = set()
        source_id = 0
        for source in source_list:
            key = (source.object_id_g, source.object_id_r, source.object_id_i)
            if key not in source_keys:
                source_keys.add(key)
                source_line = source_to_csv_line(source, source_id)
                source_exported.append(source)
                source_id += 1
                f.write(f'{source_line}\n')


if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    jobs = db.session.query(SourceIngestJob).filter(SourceIngestJob.uploaded == True).all()
    job_ids_db = [job.id for job in jobs]
    job_ids_db.sort()

    folders = glob.glob('sources_*')
    fis = []
    for folder in folders:
        fis += glob.glob(folder+'/sources.*.txt')
    job_ids_disk = [int(f.split('.')[-2]) for f in fis]

    job_ids_remaining = list(set(job_ids_db) - set(job_ids_disk))
    job_ids_remaining.sort()

    my_job_ids = np.array_split(job_ids_remaining, size)[rank]
    for job_id in my_job_ids:
        sources = db.session.query(Source).filter(Source.ingest_job_id == int(job_id)).all()
        export_sources(job_id, sources)
