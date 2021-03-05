#! /usr/bin/env python
"""
calculate_process_priority.py
"""

import numpy as np

from puzle.models import SourceIngestJob, StarProcessJob
from puzle.density import load_density_polygons, return_density
from puzle import db

density_polygons = load_density_polygons()

jobs = db.session.query(SourceIngestJob, StarProcessJob). \
    outerjoin(SourceIngestJob, StarProcessJob.source_ingest_job_id == SourceIngestJob.id). \
    all()

# find the density at the center of the job field
density_arr = []
for source_ingest_job, star_process_job in jobs:
    ra = (source_ingest_job.ra_start + source_ingest_job.ra_end) / 2.
    dec = (source_ingest_job.dec_start + source_ingest_job.dec_end) / 2.
    density = return_density(ra, dec, density_polygons)
    if density is None:
        density = 0
    density_arr.append(density)

# most to least dense
priority_order = np.argsort(density_arr)[::-1]

priority_idx_arr = []
for i, (_, star_process_job) in enumerate(jobs):
    priority = np.where(priority_order == i)[0][0]
    star_process_job.priority = priority
    db.session.add(star_process_job)
db.session.commit()