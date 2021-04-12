#! /usr/bin/env python
"""
populate_ingest_jobs.py

Source ingest jobs are a set of unique ra and dec boundaries within
which a job will look for sources and associations. This script determines
these jobs by estimating the stellar density in each ZTF field and dynamically
adjusting the density of the grid to divide up denser regions into
more jobs.

The algorithm begins with a global min and max for both ra and dec.
The first job begins at the dec global min and ra global min. A scan
is then performed along a line of constant dec and increasing ra. The
scanner is a point at the average dec of the scanner's dec bounds and
increases in small increments of ra. At each increment, the scanner
estimates the stellar density and attempts to make a box
with `n_objects_max` within it.

Star ingest jobs are simply duplicates of source ingest jobs, but for
ingesting stars within the same ra and dec boundaries.
"""

import numpy as np
import os

from puzle.models import SourceIngestJob, StarIngestJob
from puzle.density import load_density_polygons, return_density, \
    return_density_polygons_filename, save_density_polygons
from puzle import db


def populate_ingest_jobs():
    ra_min, ra_max = 0., 360.
    delta_ra_min, delta_ra_max = 0.125, 2.0
    dec_min, dec_max, delta_dec = -30., 90., 1.0
    n_objects_max = 50000

    density_polygons = load_density_polygons()

    for dec in np.arange(dec_min, dec_max, delta_dec):
        dec_start = dec
        dec_end = dec + delta_dec
        dec_avg = (dec_start + dec_end) / 2.

        ra = ra_min
        while ra < ra_max:
            density = return_density(ra, dec_avg, density_polygons)
            if density:
                delta_ra = (n_objects_max / density) / delta_dec
                delta_ra = min(delta_ra, delta_ra_max)
                delta_ra = max(delta_ra, delta_ra_min)
            else:
                delta_ra = delta_ra_max

            ra_start = ra
            ra_end = min(ra + delta_ra, ra_max)

            job = SourceIngestJob(ra_start=ra_start,
                                  ra_end=ra_end,
                                  dec_start=dec_start,
                                  dec_end=dec_end)
            db.session.add(job)

            ra += delta_ra
        db.session.commit()

    source_jobs = db.session.query(SourceIngestJob).all()
    for source_job in source_jobs:
        star_job = StarIngestJob(source_job.id)
        db.session.add(star_job)
    db.session.commit()


if __name__ == '__main__':
    if not os.path.exists(return_density_polygons_filename()):
        save_density_polygons()
    populate_ingest_jobs()
