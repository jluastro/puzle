#! /usr/bin/env python
"""
density.py
"""

import glob
import pickle
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from puzle.utils import lightcurve_file_to_ra_dec, return_data_dir, return_DR3_dir


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def return_density_polygons_filename():
    data_dir = return_data_dir()
    filename = f'{data_dir}/density_polygons.dat'
    return filename


def save_density_polygons():
    density_polygons = []

    DR3_dir = return_DR3_dir()
    objects_files = glob.glob(f'{DR3_dir}/field*objects')
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