#! /usr/bin/env python
"""
populate_process_jobs.py
"""

from puzle.models import SourceIngestJob, StarProcessJob
from puzle import db


def populate_process_jobs():
    source_ingest_jobs = db.session.query(SourceIngestJob).all()

    for source_ingest_job in source_ingest_jobs:
        source_ingest_job_id = source_ingest_job.id
        star_process_job = StarProcessJob(source_ingest_job_id=source_ingest_job_id)
        db.session.add(star_process_job)
    db.session.commit()


if __name__ == '__main__':
    populate_process_jobs()
