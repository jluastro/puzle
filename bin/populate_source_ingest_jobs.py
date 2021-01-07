#! /usr/bin/env python
"""
populate_source_ingest_jobs.py

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
"""

import numpy as np
import os
import glob
import pickle
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from puzle.models import SourceIngestJob
from puzle.utils import lightcurve_file_to_ra_dec
from puzle import db


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def return_density_polygons_filename():
    dir_path_puzle = os.path.dirname(os.path.dirname(
        os.path.realpath(__file__)))
    filename = f'{dir_path_puzle}/data/density_polygons.dat'
    return filename


def save_density_polygons():
    density_polygons = []

    objects_files = glob.glob('field*objects')
    for objects_file in objects_files:
        lightcurve_file = objects_file.replace('.objects', '.txt')
        ra0, ra1, dec0, dec1 = lightcurve_file_to_ra_dec(lightcurve_file)
        if ra0 > ra1:
            ra1 += 360
        polygon = Polygon([(ra0, dec0),
                           (ra0, dec1),
                           (ra1, dec1),
                           (ra1, dec0)])
        num_objects = file_len(objects_file)
        density = num_objects / polygon.area  # objects per square degree
        density_polygons.append((polygon, density))

    filename = return_density_polygons_filename()
    with open(filename, 'wb') as f:
        pickle.dump(density_polygons, f)


def load_density_polygons():
    filename = return_density_polygons_filename()
    density_polygons = pickle.load(open(filename, 'rb'))
    return density_polygons


def return_density(ra, dec, density_polygons):
    point = Point(ra, dec)
    for polygon, density in density_polygons:
        if polygon.contains(point):
            return density
    return None


def populate_source_ingest_jobs():
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


if __name__ == '__main__':
    if not os.path.exists(return_density_polygons_filename()):
        save_density_polygons()
    populate_source_ingest_jobs()
