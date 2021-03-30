#! /usr/bin/env python
"""
catalog.py
"""

import os
import glob
import h5py
import pickle
import psycopg2
from collections import namedtuple
from astropy.coordinates import SkyCoord
from astropy import units as u
import requests
import numpy as np
from scipy.spatial import cKDTree


def ulens_con():
    """Connection to the NERSC ulens database.

    Connection is opened to the database with:
        from puzle import catalog
        ulens_con = catalog.ulens_con()
    Connection is closed to the database with:
        ulens_con.close()
    The connection to the database should always be closed after completing
    the necessary transactions with the database.

    Args:
        None

    Returns:
        psycopg2.extensions.connection: Connection to the NERSC ulens database

    """

    if 'NERSC_HOST' in os.environ:
        return psycopg2.connect(dbname='ulens',
                                host='nerscdb03.nersc.gov',
                                port=5432,
                                user='ulens_admin',
                                password='frer334322222t3453')
    else:
        return psycopg2.connect(dbname='ulens',
                                host='localhost',
                                port=5432,
                                user='ulens_admin')


def query_ulens_db(query, con=None):
    """Sends a query to a PostgreSQL database, maintaining the state of the db.

    query_db should be used when the query DOES NOT alter the state of the
    database. This is a "read only" operation.

    Args:
        query : str
            Query to be sent to the PostgreSQL database. This query must be a
            single string and must therefore be wrapped in double quotes if the
            query requires single quotes, to specify a text value for instance.
        con : psycopg2.extensions.connection
            Connection object to the database using psycopg2. This is expected
            to be created through the ks_con function. For example:
                from puzle import catalog
                desi_con = catalog.desi_con()
                query_db(query, desi_con)
            If con == None, a connection object will be created and destroyed
            for this single execution.

    Returns:
        result : list of tuples
            The results of the query are returned as a list of tuples.
            There are three common use cases that require different wrapping
            around the query_db function to return usable results.
            Example 1: Returning a single parameter
                query = "SELECT count(*) FROM table"
                result = query_db(query, con)
                result = result[0][0] # count is a single value
            Example 2: Returning a list of single parameters
                query = "SELECT col1 FROM table;"
                result = query_db(query, con)
                result_list = [r[0] for in result] # result_list is a list
            Example 3: Returning a list of multiple parameters
                query = "SELECT col1,col2 FROM table;"
                result_list = query_db(query, con) # result_list is a list

    """
    # Obtain a connection if None is present
    close_con = False
    if con is None:
        con = ulens_con()
        close_con = True
    # Cursor allows code to execute commands and will be automatically closed
    # upon the exit from the query_db function.
    cursor = con.cursor()
    # Executes the query using the cursor.
    cursor.execute(query)
    # Fetches all rows of a query result.
    result = cursor.fetchall()
    # Close the connection if none was given to the function
    if close_con:
        con.close()
    return result


def query_ps1_psc(ra, dec, radius=2, con=None):
    radius_deg = radius / 3600.
    query = f'select obj_id, ra_stack, dec_stack, rf_score, quality_flag, ' \
            f'sqrt((ra_stack-({ra}))^2+(dec_stack-({dec}))^2) as dist from ps1_psc ' \
            f'where q3c_radial_query(ra_stack, dec_stack, {ra}, {dec}, {radius_deg}) ' \
            f'order by dist asc limit 1;'
    result = query_ulens_db(query, con=con)

    if len(result) == 1:
        result = result[0]
        rf_tuple = namedtuple('rf',
                              'obj_id ra_stack dec_stack rf_score quality_flag')
        rf = rf_tuple(obj_id=int(result[0]),
                      ra_stack=float(result[1]),
                      dec_stack=float(result[2]),
                      rf_score=float(result[3]),
                      quality_flag=int(result[4]))
    else:
        rf = None
    return rf


def fetch_ogle_targets():
    target_ids = []
    ra_arr = []
    dec_arr = []
    for year in ['2017', '2018', '2019']:
        URL = f'http://ogle.astrouw.edu.pl/ogle4/ews/{year}/ews.html'
        html_lines = requests.get(URL).content.decode().split('\n')

        target_idxs = [i for i, line in enumerate(html_lines) if 'TARGET="event"' in line and 'XX' not in line]
        target_ids.extend([html_lines[i].split('>')[-2].replace('</A', '') for i in target_idxs])
        ra_arr.extend([html_lines[i+2].replace('<TD>', '') for i in target_idxs])
        dec_arr.extend([html_lines[i+3].replace('<TD>', '') for i in target_idxs])

    target_coords = SkyCoord(ra_arr, dec_arr, frame='icrs', unit=u.degree)

    return {'target_ids': target_ids, 'target_coords': target_coords}


OGLE_TARGETS = fetch_ogle_targets()


def fetch_ogle_target(ra_cand, dec_cand, radius=5):
    target_ids = OGLE_TARGETS['target_ids']
    target_coords = OGLE_TARGETS['target_coords']

    candidate_coord = SkyCoord(ra_cand, dec_cand, frame='icrs', unit=u.degree)

    separation = target_coords.separation(candidate_coord).value
    if np.min(separation) < (radius / 3600.):
        target_id = target_ids[np.argmin(separation)]
    else:
        target_id = None

    return target_id


def generate_ps1_psc_maps():
    ps1_psc_dir = '/global/cfs/cdirs/uLens/PS1_PSC'
    ps1_psc_filenames = glob.glob(f'{ps1_psc_dir}/*.h5')
    ps1_psc_filenames.sort()

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 0

    my_filenames = np.array_split(ps1_psc_filenames, size)[rank]

    for i, ps1_psc_filename in enumerate(my_filenames):
        print(rank, ps1_psc_filename, i, len(my_filenames))
        radec = h5py.File(ps1_psc_filename, 'r')['class_table']['block0_values'][:, :2]
        rf_scores = h5py.File(ps1_psc_filename, 'r')['class_table']['block0_values'][:, 2]
        kdtree = cKDTree(radec)

        map_filename = ps1_psc_filename.replace('.h5', '.map')
        with open(map_filename, 'wb') as fileObj:
            pickle.dump((kdtree, rf_scores), fileObj)


def return_ps1_psc(dec):
    dec_sign = np.sign(dec)
    if dec_sign == 1:
        dec_file = dec
        dec_prefix = ''
    elif dec_sign == -1:
        dec_file = np.abs(dec) + 1 / 3.
        dec_prefix = 'neg'
    else:
        dec_file = 0
        dec_prefix = ''

    dec_floor_str = np.floor(dec_file).astype(int).astype(str)
    dec_third = np.mod(dec_file, 1) / (1 / 3.)

    if dec_third < 1:
        dec_ext_str = '0'
    elif dec_third < 2:
        dec_ext_str = '33'
    elif dec_third < 3:
        dec_ext_str = '66'
    else:
        raise Exception

    ps1_psc_fname = f'/global/cfs/cdirs/uLens/PS1_PSC/' \
                    f'dec_{dec_prefix}{dec_floor_str}_{dec_ext_str}_classifications.map'
    ps1_psc_kdtree, rf_scores = pickle.load(open(ps1_psc_fname, 'rb'))
    return ps1_psc_kdtree, rf_scores


def query_ps1_psc_on_disk(ra, dec, radius=2):
    ps1_psc_kdtree, rf_scores = return_ps1_psc(dec)
    radius_deg = radius / 3600.
    idx_arr = ps1_psc_kdtree.query_ball_point((ra, dec), radius_deg)

    if len(idx_arr) == 0:
        rf = None
    else:
        idx = idx_arr[0]
        rf_score = rf_scores[idx]
        ra_ps1_psc, dec_ps1_psc = ps1_psc_kdtree.data[idx]
        rf_tuple = namedtuple('rf',
                              'ra_stack dec_stack rf_score')
        rf = rf_tuple(ra_stack=ra_ps1_psc,
                      dec_stack=dec_ps1_psc,
                      rf_score=rf_score)
    return rf
