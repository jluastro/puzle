#! /usr/bin/env python
"""
generate_level4_candidates_lightcurve_plots.py
"""

import os
import numpy as np
from microlens.jlu.model import PSPL_Phot_Par_Param1

from puzle.models import Source, CandidateLevel4
from puzle import db


def return_source_folder(source_id):
    job_id = source_id.split('_')[0]
    job_id_prefix = job_id[:3]
    folder = f'/global/cfs/cdirs/uLens/puzleapp/static/source/{job_id_prefix}/{job_id}'
    return folder


def generate_level4_candidates_lightcurve_plots():
    cands = db.session.query(CandidateLevel4).\
        filter(CandidateLevel4.pspl_gp_fit_finished == True,
               CandidateLevel4.fit_type_pspl_gp != None).\
        order_by(CandidateLevel4.id).\
        all()

    cands_data = []
    for cand in cands:
        for source_id in cand.unique_source_id_arr:
            folder = return_source_folder(source_id)
            fname = f'{folder}/{source_id}_lightcurve.png'
            if not os.path.exists(fname):
                cands_data.append((cand.pspl_gp_fit_dct, cand.unique_source_id_arr))

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    my_cands_data = np.array_split(cands_data, size)[rank]

    for pspl_gp_fit_dct, unique_source_id_arr in my_cands_data:
        for source_id in unique_source_id_arr:
            source = Source.query.filter(Source.id == source_id).first_or_404()
            if source_id in pspl_gp_fit_dct:
                model_params = pspl_gp_fit_dct[source_id]
                model = PSPL_Phot_Par_Param1
            else:
                model_params = None
                model = None
            folder = return_source_folder(source_id)
            source.load_lightcurve_plot(folder=folder, model_params=model_params, model=model)


if __name__ == '__main__':
    generate_level4_candidates_lightcurve_plots()
