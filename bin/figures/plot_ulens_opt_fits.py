#! /usr/bin/env python
"""
plot_ulens_opt_fits.py
"""

import corner
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy.sql.expression import func

from puzle.ulens import return_ulens_data, return_ulens_metadata, return_ulens_stats
from puzle.cands import fetch_cand_best_obj_by_id
from puzle.models import CandidateLevel2, CandidateLevel3
from puzle.utils import return_figures_dir
from puzle import db


def plot_ulens_opt_corner():
    cands3 = CandidateLevel3.query.with_entities(CandidateLevel3.tE_best,
                                                 CandidateLevel3.u0_amp_best,
                                                 CandidateLevel3.mag_src_best,
                                                 CandidateLevel3.chi_squared_delta_best,
                                                 CandidateLevel3.piE_E_best,
                                                 CandidateLevel3.piE_N_best,
                                                 CandidateLevel3.eta_best,
                                                 CandidateLevel3.eta_residual_best).all()
    tE_cands = np.array([c[0] for c in cands3], dtype=np.float32)
    u0_amp_cands = np.array([c[1] for c in cands3], dtype=np.float32)
    mag_src_cands = np.array([c[2] for c in cands3], dtype=np.float32)
    chi_squared_delta_cands = np.array([c[3] for c in cands3], dtype=np.float32)
    piE_E_cands = np.array([c[4] for c in cands3], dtype=np.float32)
    piE_N_cands = np.array([c[5] for c in cands3], dtype=np.float32)
    eta_cands = np.array([c[6] for c in cands3], dtype=np.float32)
    eta_residual_cands = np.array([c[7] for c in cands3], dtype=np.float32)

    cond = tE_cands > 0
    cond *= ~np.isnan(tE_cands)
    tE_cands = tE_cands[cond]
    log_tE_cands = np.log10(tE_cands)
    u0_amp_cands = u0_amp_cands[cond]
    mag_src_cands = mag_src_cands[cond]
    chi_squared_delta_cands = chi_squared_delta_cands[cond]
    log_chi_squared_delta_cands = np.log10(chi_squared_delta_cands)
    piE_E_cands = piE_E_cands[cond]
    piE_N_cands = piE_N_cands[cond]
    piE_cands = np.hypot(piE_E_cands, piE_N_cands)
    log_piE_cands = np.log10(piE_cands)
    eta_cands = eta_cands[cond]
    eta_residual_cands = eta_residual_cands[cond]

    data_cands = np.vstack((log_tE_cands,
                            u0_amp_cands,
                            mag_src_cands,
                            log_chi_squared_delta_cands,
                            log_piE_cands,
                            eta_cands,
                            eta_residual_cands)).T

    stats = return_ulens_stats(observableFlag=True, bhFlag=True)

    tE_ulens = stats['tE_level3']
    log_tE_ulens = np.log10(tE_ulens)
    u0_amp_ulens = stats['u0_amp_level3']
    mag_src_ulens = stats['mag_src_level3']
    chi_squared_delta_ulens = stats['chi_squared_delta_level3']
    log_chi_squared_delta_ulens = np.log10(chi_squared_delta_ulens)
    piE_ulens = stats['piE_level3']
    log_piE_ulens = np.log10(piE_ulens)
    eta_ulens = stats['eta_level3']
    eta_residual_ulens = stats['eta_residual_level3']

    data_ulens = np.vstack((log_tE_ulens,
                            u0_amp_ulens,
                            mag_src_ulens,
                            log_chi_squared_delta_ulens,
                            log_piE_ulens,
                            eta_ulens,
                            eta_residual_ulens)).T

    idx_sample = np.random.choice(np.arange(len(data_cands)), replace=False, size=len(data_ulens))
    data_cands_sample = data_cands[idx_sample]

    # Plot it.
    labels = ['LOG tE', 'u0_amp', 'mag_src', 'LOG chi_squared_delta', 'LOG piE', 'eta', 'eta_residual']
    data_range = [(.5, 4), (-3, 3), (10, 24), (1, 8), (-3, 2), (0, 1), (0, 4)]

    fig = corner.corner(data_ulens, labels=labels, range=data_range,
                        show_titles=True, title_kwargs={"fontsize": 12}, color='r')
    fig = corner.corner(data_cands_sample, labels=labels, range=data_range,
                        show_titles=True, title_kwargs={"fontsize": 12}, color='k',
                        fig=fig)
    fig.suptitle('Level 3 Fits (ulens red | cands black)')

    fname = '%s/ulens_opt_corner.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def calculate_inside_outside(hmjd, mag, magerr, t0, tE):
    cond = hmjd < t0 + 2 * tE
    cond *= hmjd > t0 - 2 * tE
    num_inside = np.sum(cond)
    num_outside = np.sum(~cond)

    if num_inside >= 3:
        std_inside = np.std(mag[cond])
    else:
        std_inside = 0
    if num_outside >= 3:
        std_outside = np.std(mag[~cond])
    else:
        std_outside = 0

    mag_avg_outside = np.median(mag[~cond])
    chi_squared = np.sum((mag[~cond] - mag_avg_outside) ** 2. / magerr[~cond] ** 2.)

    return std_inside, std_outside, chi_squared, num_outside - 1
    
    
def plot_ulens_opt_inside_outside():
    cands23 = db.session.query(CandidateLevel2, CandidateLevel3). \
        filter(CandidateLevel2.id == CandidateLevel3.id). \
        filter(CandidateLevel2.t_E_best <= 400). \
        order_by(func.random()).with_entities(CandidateLevel3.id,
                                              CandidateLevel2.t_0_best,
                                              CandidateLevel2.t_E_best).limit(10000).all()
    cand_id_arr = [c[0] for c in cands23]
    obj_arr = []
    for i, cand_id in enumerate(cand_id_arr):
        if i % 100 == 0:
            print(i, len(cand_id_arr))
        obj = fetch_cand_best_obj_by_id(cand_id)
        _ = obj.lightcurve.hmjd
        obj_arr.append(obj)

    t0_arr = [c[1] for c in cands23]
    tE_arr = [c[2] for c in cands23]
    std_inside_cands = []
    std_outside_cands = []
    chi_squared_cands = []
    dof_cands = []
    for i, (obj, t0, tE) in enumerate(zip(obj_arr, t0_arr, tE_arr)):
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.mag
        magerr = obj.lightcurve.magerr
        std_inside, std_outside, chi_squared, dof = calculate_inside_outside(hmjd, mag, magerr, t0, tE)
        std_inside_cands.append(std_inside)
        std_outside_cands.append(std_outside)
        chi_squared_cands.append(chi_squared)
        dof_cands.append(dof)
    std_inside_cands = np.array(std_inside_cands)
    std_outside_cands = np.array(std_outside_cands)
    std_ratio_cands = std_outside_cands / std_inside_cands
    std_ratio_cands[np.isinf(std_ratio_cands)] = 0
    chi_squared_cands = np.array(chi_squared_cands)
    dof_cands = np.array(dof_cands)

    data = return_ulens_data(observableFlag=True, bhFlag=True)
    metadata = return_ulens_metadata(observableFlag=True, bhFlag=True)

    size = min(len(data), len(obj_arr))
    idx_sample = np.random.choice(np.arange(len(data)), replace=False, size=size)

    std_inside_ulens = []
    std_outside_ulens = []
    chi_squared_ulens = []
    dof_ulens = []
    for idx in idx_sample:
        hmjd = data[idx][:, 0]
        mag = data[idx][:, 1]
        magerr = data[idx][:, 2]
        t0 = metadata['t0'][idx]
        tE = metadata['tE'][idx]
        std_inside, std_outside, chi_squared, dof = calculate_inside_outside(hmjd, mag, magerr, t0, tE)
        std_inside_ulens.append(std_inside)
        std_outside_ulens.append(std_outside)
        chi_squared_ulens.append(chi_squared)
        dof_ulens.append(dof)
    std_inside_ulens = np.array(std_inside_ulens)
    std_outside_ulens = np.array(std_outside_ulens)
    std_ratio_ulens = std_outside_ulens / std_inside_ulens
    std_ratio_ulens[np.isinf(std_ratio_ulens)] = 0
    chi_squared_ulens = np.array(chi_squared_ulens)
    dof_ulens = np.array(dof_ulens)


def plot_ulens_tE_opt_bias():
    stats = return_ulens_stats(observableFlag=True, bhFlag=False)
    tE_measured = stats['tE_level3']
    log_tE_measured = np.log10(tE_measured)

    metadata = return_ulens_metadata(observableFlag=True, bhFlag=False)
    tE_modeled = metadata['tE']
    log_tE_modeled = np.log10(tE_modeled)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax = ax.flatten()
    for a in ax:
        a.clear()

    x0 = np.linspace(1.3, 2.6, 1000)
    m0, b0 = np.polyfit(log_tE_modeled, log_tE_measured, deg=1)
    ax[0].set_title('m,b = %.2f, %.2f' % (m0, b0))
    ax[0].scatter(log_tE_modeled, log_tE_measured, s=1, alpha=.5)
    ax[0].set_xlabel('log tE modeled')
    ax[0].set_ylabel('log tE measured')
    ax[0].plot(x0, x0, color='white')
    ax[0].plot(x0, x0 * m0 + b0, color='r')
    ax[0].set_xlim(1.3, 2.6)
    ax[0].set_ylim(0, 3)

    x1 = np.linspace(0, 3, 1000)
    m1, b1 = np.polyfit(log_tE_measured, log_tE_modeled, deg=1)
    ax[1].set_title('m,b = %.2f, %.2f' % (m1, b1))
    ax[1].hexbin(log_tE_measured, log_tE_modeled, s=1, alpha=.5)
    ax[1].set_xlabel('log tE measured')
    ax[1].set_ylabel('log tE modeled')
    ax[1].plot(x1, x1, color='white')
    ax[1].plot(x1, x1 * m1 + b1, color='r')
    ax[1].set_xlim(0, 3)
    ax[1].set_ylim(1.3, 2.6)

    fig.tight_layout()