#! /usr/bin/env python
"""
utils.py
"""
import os
import glob
import subprocess
import numpy as np
from datetime import datetime
from zort.radec import return_shifted_ra, return_ZTF_RCID_corners
from shapely.geometry.polygon import Polygon


def return_dir(folder):
    curdir = os.path.abspath(__file__)
    dir = '/'.join(curdir.split('/')[:-2]) + folder
    return dir


def return_DR3_dir():
    return return_dir('/data/DR3')


def return_data_dir():
    return return_dir('/data')


def return_figures_dir():
    return return_dir('/figures')


def execute(cmd,
            shell=False):
    """Executes a command line instruction, captures the stdout and stderr

    Args:
        cmd : str
            Command line instruction, including any executables and parameters.
        shell : bool
            Determines if the command is run through the shell.

    Returns:
        stdout : str
            Contains the standard output of the executed process.
        stderr : str
            Contains the standard error of the executed process.

    """
    # Split the argument into a list suitable for Popen
    args = cmd.split()
    # subprocess.PIPE indicates that a pipe
    # to the standard stream should be opened.
    process = subprocess.Popen(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               shell=shell)
    stdout, stderr = process.communicate()

    return stdout, stderr


def fetch_job_enddate():
    job_id = os.getenv('SLURM_JOB_ID')
    if job_id is None:
        return None

    cmd = 'squeue -u mmedford -j %s --format="%%e" --noheader' % job_id
    stdout, _ = execute(cmd)
    enddate_str = stdout.decode().replace('\n', '').replace('"','')
    enddate = datetime.strptime(enddate_str, '%Y-%m-%dT%H:%M:%S')
    return enddate


def lightcurve_file_to_ra_dec(lightcurve_file):
    _, ra_str, dec_str = os.path.basename(lightcurve_file).split('_')
    ra0, ra1 = ra_str.replace('ra', '').split('to')
    ra0, ra1 = float(ra0), float(ra1)
    dec0, dec1 = dec_str.replace('.txt', '').replace('dec', '').split('to')
    dec0, dec1 = float(dec0), float(dec1)
    return ra0, ra1, dec0, dec1


def lightcurve_file_to_field_id(lightcurve_file):
    lightcurve_file = os.path.basename(lightcurve_file)
    field_id = int(lightcurve_file.split('_')[0].replace('field', ''))
    return field_id


def fetch_lightcurve_rcids(ra_start, ra_end, dec_start, dec_end):
    DR3_dir = return_DR3_dir()
    lightcurve_files = glob.glob(f'{DR3_dir}/field*txt')
    lightcurve_files.sort()

    lightcurve_rcids_arr = []
    for i, lightcurve_file in enumerate(lightcurve_files):
        field_id = lightcurve_file_to_field_id(lightcurve_file)

        ra0, ra1, dec0, dec1 = lightcurve_file_to_ra_dec(lightcurve_file)
        if ra1 < ra0:
            ra0_shifted = return_shifted_ra(ra0, field_id)
            ra1_shifted = return_shifted_ra(ra1, field_id)
        else:
            ra0_shifted = ra0
            ra1_shifted = ra1

        file_polygon = Polygon([(ra0_shifted, dec0),
                                (ra0_shifted, dec1),
                                (ra1_shifted, dec1),
                                (ra1_shifted, dec0)])

        if ra0_shifted < ra0 and ra_start > 180:
            ra_start_shifted = ra_start - 360
            ra_end_shifted = ra_end - 360
        elif ra1_shifted > ra1 and ra_end < 180:
            ra_start_shifted = ra_start + 360
            ra_end_shifted = ra_end + 360
        else:
            ra_start_shifted = ra_start
            ra_end_shifted = ra_end

        job_polygon = Polygon([(ra_start_shifted, dec_start),
                               (ra_start_shifted, dec_end),
                               (ra_end_shifted, dec_end),
                               (ra_end_shifted, dec_start)])

        if not file_polygon.intersects(job_polygon):
            continue

        ZTF_RCID_corners = return_ZTF_RCID_corners(field_id)

        rcids_to_read = []
        for rcid, corners in ZTF_RCID_corners.items():
            rcid_polygon = Polygon(corners)
            if rcid_polygon.intersects(job_polygon):
                rcids_to_read.append(rcid)

        if len(rcids_to_read) > 0:
            lightcurve_rcids_arr.append((lightcurve_file, rcids_to_read))

    return lightcurve_rcids_arr


def stack_ragged(array_list):
    lengths = [np.shape(a)[1] for a in array_list]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(array_list, axis=1)
    return stacked, idx


def save_stacked_array(fname, array_list):
    stacked, idx = stack_ragged(array_list)
    np.savez(fname, stacked_array=stacked, stacked_index=idx)


def load_stacked_arrays(fname):
    npzfile = np.load(fname)
    idx = npzfile['stacked_index']
    stacked = npzfile['stacked_array']
    return np.split(stacked.T, idx, axis=0)
