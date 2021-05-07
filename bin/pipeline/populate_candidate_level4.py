#! /usr/bin/env python
"""
populate_candidate_level4.py
"""
import os
import numpy as np
import pickle

from microlens.jlu.model import PSPL_Phot_Par_Param1

from puzle.models import Source, CandidateLevel2, CandidateLevel3, CandidateLevel4
from puzle.cands import apply_level3_cuts_to_query, fit_data_to_ulens_opt, return_sigma_peaks
from puzle.stats import average_xy_on_round_x, calculate_eta, calculate_chi_squared_inside_outside
from puzle.utils import return_DR5_dir
from puzle import db


def _parse_object_int(attr):
    if attr == 'None':
        return None
    else:
        return int(attr)


def csv_line_to_source(line):
    attrs = line.replace('\n', '').split(',')
    source = Source(id=attrs[0],
                    object_id_g=_parse_object_int(attrs[1]),
                    object_id_r=_parse_object_int(attrs[2]),
                    object_id_i=_parse_object_int(attrs[3]),
                    lightcurve_position_g=_parse_object_int(attrs[4]),
                    lightcurve_position_r=_parse_object_int(attrs[5]),
                    lightcurve_position_i=_parse_object_int(attrs[6]),
                    lightcurve_filename=attrs[7],
                    ra=float(attrs[8]),
                    dec=float(attrs[9]),
                    ingest_job_id=int(attrs[10]))
    return source


def load_source(source_id):
    DR5_dir = return_DR5_dir()
    source_job_id = int(source_id.split('_')[0])
    source_job_prefix = str(source_job_id)[:3]
    sources_fname = f'{DR5_dir}/sources_{source_job_prefix}/sources.{source_job_id:06d}.txt'

    sources_map_fname = sources_fname.replace('.txt', '.sources_map')
    sources_map = pickle.load(open(sources_map_fname, 'rb'))

    f_sources = open(sources_fname, 'r')
    f_sources.seek(sources_map[source_id])
    line_source = f_sources.readline()
    source = csv_line_to_source(line_source)
    f_sources.close()
    return source


def calculate_chi2_model_params(best_params, data):
    params_to_fit = ['t0', 'u0_amp', 'tE', 'mag_src',
                     'b_sff', 'piE_E', 'piE_N']
    params = {}
    for k, v in best_params.items():
        if k in params_to_fit:
            params[k] = v
    model = PSPL_Phot_Par_Param1(**params,
                                 raL=data['raL'], decL=data['decL'])

    mag_model = model.get_photometry(data['hmjd'], print_warning=False)

    chi2 = np.sum(((data['mag'] - mag_model) / data['magerr']) ** 2)
    return chi2


def populate_candidate_level4():
    query = apply_level3_cuts_to_query(CandidateLevel3.query)
    cands_level3 = query.order_by(CandidateLevel3.id).all()
    cands_level4 = CandidateLevel4.query.with_entities(CandidateLevel4.id).all()
    cand_ids_level4 = set([c[0] for c in cands_level4])
    cands_level3 = [c for c in cands_level3 if c.id not in cand_ids_level4]

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    if rank == 0:
        print('rank 0) %i total candidates identified' % len(cands_level3))

    my_cands_level3 = np.array_split(cands_level3, size)[rank]
    num_my_cands = len(my_cands_level3)
    print(f'rank {rank}) {num_my_cands} my candidates identified')

    for i, cand_level3 in enumerate(my_cands_level3):
        if i % 100 == 0:
            print(f'rank {rank}) {i} candidates complete')
        cand_level2 = CandidateLevel2.query.filter(CandidateLevel2.id==cand_level3.id).first()
        t0_level3 = cand_level3.t0_best
        tE_level3 = cand_level3.tE_best
        num_epochs_arr = []
        num_days_arr = []
        eta_arr = []
        eta_residual_arr = []
        t0_arr = []
        u0_amp_arr = []
        tE_arr = []
        mag_src_arr = []
        b_sff_arr = []
        piE_E_arr = []
        piE_N_arr = []
        chi_squared_ulens_arr = []
        chi_squared_flat_arr = []
        num_days_inside_arr = []
        num_days_outside_arr = []
        chi_squared_flat_inside_arr = []
        chi_squared_flat_outside_arr = []
        delta_hmjd_outside_arr = []
        num_3sigma_peaks_inside_arr = []
        num_3sigma_peaks_outside_arr = []
        num_5sigma_peaks_inside_arr = []
        num_5sigma_peaks_outside_arr = []
        source_dct = {}
        for source_id, color in zip(cand_level3.source_id_arr,
                                    cand_level3.color_arr):
            if source_id in source_dct:
                source = source_dct[source_id]
            else:
                source = load_source(source_id)
                source_dct[source_id] = source

            obj = getattr(source.zort_source, f'object_{color}')
            hmjd = obj.lightcurve.hmjd
            mag = obj.lightcurve.mag
            magerr = obj.lightcurve.magerr
            hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
            _, magerr_round = average_xy_on_round_x(hmjd, magerr)
            ra = obj.ra
            dec = obj.dec
            chi_squared_flat = np.sum(((mag_round - np.mean(mag_round)) / magerr_round) ** 2)
            chi_squared_flat_arr.append(chi_squared_flat)

            num_epochs = int(len(hmjd))
            num_days = int(len(set(np.round(hmjd))))
            num_epochs_arr.append(num_epochs)
            num_days_arr.append(num_days)

            if num_days > 1:
                eta = calculate_eta(mag_round)
                best_params = fit_data_to_ulens_opt(hmjd, mag, magerr, ra, dec,
                                                    t0_guess=t0_level3, tE_guess=tE_level3)
                # best_params = fit_data_to_ulens_opt(hmjd, mag, magerr, ra, dec,
                #                                     t0_guess=cand_level2.t_0_best, tE_guess=cand_level2.t_E_best)
            else:
                eta = 0
                param_names_to_fit = ['t0', 'u0_amp', 'tE', 'mag_src',
                                      'b_sff', 'piE_E', 'piE_N']
                best_params = {k: 0 for k in param_names_to_fit}
                best_params['chi_squared_ulens'] = 0
                best_params['eta_residual'] = 0

            eta_arr.append(eta)
            t0_arr.append(best_params['t0'])
            u0_amp_arr.append(best_params['u0_amp'])
            tE_arr.append(best_params['tE'])
            mag_src_arr.append(best_params['mag_src'])
            b_sff_arr.append(best_params['b_sff'])
            piE_E_arr.append(best_params['piE_E'])
            piE_N_arr.append(best_params['piE_N'])
            eta_residual_arr.append(best_params['eta_residual'])
            chi_squared_ulens_arr.append(best_params['chi_squared_ulens'])

            data = calculate_chi_squared_inside_outside(hmjd=hmjd,
                                                        mag=mag,
                                                        magerr=magerr,
                                                        t0=best_params['t0'],
                                                        tE=best_params['tE'],
                                                        tE_factor=2)
            chi_squared_flat_inside, chi_squared_flat_outside, num_days_inside, num_days_outside, delta_hmjd_outside = data

            num_days_inside_arr.append(num_days_inside)
            num_days_outside_arr.append(num_days_outside)
            chi_squared_flat_inside_arr.append(chi_squared_flat_inside)
            chi_squared_flat_outside_arr.append(chi_squared_flat_outside)
            delta_hmjd_outside_arr.append(delta_hmjd_outside)

            num_3sigma_peaks_inside, num_3sigma_peaks_outside = return_sigma_peaks(
                hmjd, mag, best_params['t0'], best_params['tE'], sigma_factor=3, tE_factor=2)
            num_3sigma_peaks_inside_arr.append(num_3sigma_peaks_inside)
            num_3sigma_peaks_outside_arr.append(num_3sigma_peaks_outside)

            num_5sigma_peaks_inside, num_5sigma_peaks_outside = return_sigma_peaks(
                hmjd, mag, best_params['t0'], best_params['tE'], sigma_factor=5, tE_factor=2)
            num_5sigma_peaks_inside_arr.append(num_5sigma_peaks_inside)
            num_5sigma_peaks_outside_arr.append(num_5sigma_peaks_outside)

        cand_level4 = CandidateLevel4(
            id=cand_level3.id,
            ra=cand_level3.ra,
            dec=cand_level3.dec,
            source_id_arr=cand_level3.source_id_arr,
            color_arr=cand_level3.color_arr,
            pass_arr=cand_level3.pass_arr,
            idx_best=cand_level3.idx_best,
            num_objs_pass=cand_level3.num_objs_pass,
            num_objs_tot=cand_level3.num_objs_tot,
            num_epochs_arr=num_epochs_arr,
            num_days_arr=num_days_arr,
            eta_arr=eta_arr,
            eta_residual_arr=eta_residual_arr,
            t0_arr=t0_arr,
            u0_amp_arr=u0_amp_arr,
            tE_arr=tE_arr,
            mag_src_arr=mag_src_arr,
            b_sff_arr=b_sff_arr,
            piE_E_arr=piE_E_arr,
            piE_N_arr=piE_N_arr,
            chi_squared_ulens_arr=chi_squared_ulens_arr,
            chi_squared_flat_arr=chi_squared_flat_arr,
            num_days_inside_arr=num_days_inside_arr,
            num_days_outside_arr=num_days_outside_arr,
            chi_squared_flat_inside_arr=chi_squared_flat_inside_arr,
            chi_squared_flat_outside_arr=chi_squared_flat_outside_arr,
            delta_hmjd_outside_arr=delta_hmjd_outside_arr,
            num_3sigma_peaks_inside_arr=num_3sigma_peaks_inside_arr,
            num_3sigma_peaks_outside_arr=num_3sigma_peaks_outside_arr,
            num_5sigma_peaks_inside_arr=num_5sigma_peaks_inside_arr,
            num_5sigma_peaks_outside_arr=num_5sigma_peaks_outside_arr,
        )
        db.session.add(cand_level4)

    db.session.commit()
    db.session.close()

    if size > 1:
        comm.Barrier()
    if rank == 0:
        print('Upload to candidate_level4 complete')


if __name__ == '__main__':
    populate_candidate_level4()
