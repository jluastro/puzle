#! /usr/bin/env python
"""
utils.py
"""
import os
import subprocess
from datetime import datetime


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

