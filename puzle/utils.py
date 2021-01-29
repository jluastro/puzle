#! /usr/bin/env python
"""
utils.py
"""
import os
import subprocess
from datetime import datetime


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
