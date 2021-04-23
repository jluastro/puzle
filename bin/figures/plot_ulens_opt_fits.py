#! /usr/bin/env python
"""
plot_ulens_opt_fits.py
"""

import corner
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy.sql.expression import func
from microlens.jlu.model import PSPL_Phot_Par_Param1

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
                                                 CandidateLevel3.eta_residual_best,
                                                 CandidateLevel3.num_days_best).\
        filter(CandidateLevel3.tE_best>0).all()
    tE_cands = np.array([c[0] for c in cands3], dtype=np.float32)
    u0_amp_cands = np.array([c[1] for c in cands3], dtype=np.float32)
    mag_src_cands = np.array([c[2] for c in cands3], dtype=np.float32)
    chi_squared_delta_cands = np.array([c[3] for c in cands3], dtype=np.float32)
    piE_E_cands = np.array([c[4] for c in cands3], dtype=np.float32)
    piE_N_cands = np.array([c[5] for c in cands3], dtype=np.float32)
    eta_cands = np.array([c[6] for c in cands3], dtype=np.float32)
    eta_residual_cands = np.array([c[7] for c in cands3], dtype=np.float32)
    num_days_cands = np.array([c[8] for c in cands3], dtype=np.float32)

    log_tE_cands = np.log10(tE_cands)
    log_piE_E_cands = np.log10(piE_E_cands)
    log_piE_N_cands = np.log10(piE_N_cands)
    piE_cands = np.hypot(piE_E_cands, piE_N_cands)
    log_piE_cands = np.log10(piE_cands)
    
    chi_squared_delta_reduced_cands = chi_squared_delta_cands / num_days_cands

    data_cands = np.vstack((log_tE_cands,
                            u0_amp_cands,
                            mag_src_cands,
                            chi_squared_delta_reduced_cands,
                            log_piE_E_cands,
                            log_piE_N_cands,
                            log_piE_cands,
                            eta_cands,
                            eta_residual_cands)).T

    stats = return_ulens_stats(observableFlag=True, bhFlag=True)
    tE_ulens = stats['tE_level3']
    u0_amp_ulens = stats['u0_amp_level3']
    mag_src_ulens = stats['mag_src_level3']
    chi_squared_delta_ulens = stats['chi_squared_delta_level3']
    piE_E_ulens = stats['piE_E_level3']
    piE_N_ulens = stats['piE_N_level3']
    eta_ulens = stats['eta']
    eta_residual_ulens = stats['eta_residual_level3']

    data = return_ulens_data(observableFlag=True, bhFlag=True)
    num_days_ulens = np.array([len(set(np.round(d[:, 0]))) for d in data])
    
    log_tE_ulens = np.log10(tE_ulens)
    log_piE_E_ulens = np.log10(piE_E_ulens)
    log_piE_N_ulens = np.log10(piE_N_ulens)
    piE_ulens = np.hypot(piE_E_ulens, piE_N_ulens)
    log_piE_ulens = np.log10(piE_ulens)
    
    chi_squared_delta_reduced_ulens = chi_squared_delta_ulens / num_days_ulens

    data_ulens = np.vstack((log_tE_ulens,
                            u0_amp_ulens,
                            mag_src_ulens,
                            chi_squared_delta_reduced_ulens,
                            log_piE_E_ulens,
                            log_piE_N_ulens,
                            log_piE_ulens,
                            eta_ulens,
                            eta_residual_ulens)).T

    idx_sample = np.random.choice(np.arange(len(data_cands)), replace=False, size=len(data_ulens))
    data_cands_sample = data_cands[idx_sample]

    # Plot it.
    labels = ['LOG tE', 'u0_amp', 'mag_src', 'chi_squared_delta_reduced', 'LOG piE_E', 'LOG piE_N', 'LOG piE', 'eta', 'eta_residual']
    data_range = [(.5, 4), (-3, 3), (10, 24), (0, 10), (-3, 2), (-3, 2), (-3, 2), (0, 1), (0, 4)]

    fig, ax = plt.subplots(5, 2, figsize=(8, 8))
    ax = ax.flatten()
    for i, label in enumerate(labels):
        ax[i].set_title(label)
        bins = np.linspace(data_range[i][0],
                           data_range[i][1], 25)
        ax[i].hist(data_ulens[:, i], histtype='step',
                   bins=bins, color='r')
        ax[i].hist(data_cands_sample[:, i], histtype='step',
                   bins=bins, color='k')
        ax[i].set_xlim(data_range[i])
    fig.tight_layout()

    fig = corner.corner(data_ulens, labels=labels, range=data_range,
                        show_titles=True, title_kwargs={"fontsize": 6},
                        label_kwargs={'fontsize': 6}, color='r')
    fig = corner.corner(data_cands_sample, labels=labels, range=data_range,
                        show_titles=True, title_kwargs={"fontsize": 6},
                        label_kwargs={'fontsize': 6}, color='k',
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
    ulens23 = db.session.query(CandidateLevel2, CandidateLevel3). \
        filter(CandidateLevel2.id == CandidateLevel3.id). \
        filter(CandidateLevel2.t_E_best <= 400). \
        order_by(func.random()).with_entities(CandidateLevel3.id,
                                              CandidateLevel2.t_0_best,
                                              CandidateLevel2.t_E_best).limit(10000).all()
    cand_id_arr = [c[0] for c in ulens23]
    obj_arr = []
    for i, cand_id in enumerate(cand_id_arr):
        if i % 100 == 0:
            print(i, len(cand_id_arr))
        obj = fetch_cand_best_obj_by_id(cand_id)
        _ = obj.lightcurve.hmjd
        obj_arr.append(obj)

    t0_arr = [c[1] for c in ulens23]
    tE_arr = [c[2] for c in ulens23]
    std_inside_ulens = []
    std_outside_ulens = []
    chi_squared_ulens = []
    dof_ulens = []
    for i, (obj, t0, tE) in enumerate(zip(obj_arr, t0_arr, tE_arr)):
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.mag
        magerr = obj.lightcurve.magerr
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
    chi_squared_reduced_ulens = chi_squared_ulens / dof_ulens

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
    chi_squared_reduced_ulens = chi_squared_ulens / dof_ulens


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


def _plot_ulens_model(ax, stats_obs, metadata_obs, idx, hmjd):
    t0 = stats_obs['t0_level3'][idx]
    u0_amp = stats_obs['u0_amp_level3'][idx]
    tE = stats_obs['tE_level3'][idx]
    piE_E = stats_obs['piE_E_level3'][idx]
    piE_N = stats_obs['piE_N_level3'][idx]
    b_sff = stats_obs['b_sff_level3'][idx]
    mag_src = stats_obs['mag_src_level3'][idx]
    raL = metadata_obs['ra'][idx]
    decL = metadata_obs['dec'][idx]
    model = PSPL_Phot_Par_Param1(
        t0=t0, u0_amp=u0_amp, tE=tE,
        piE_E=piE_E, piE_N=piE_N,
        b_sff=b_sff, mag_src=mag_src,
        raL=raL, decL=decL)
    hmjd_model = np.linspace(np.min(hmjd), np.max(hmjd), 10000)
    mag_model = model.get_photometry(hmjd_model)
    ax.plot(hmjd_model, mag_model, color='g')
    ax.axvline(t0, color='r', alpha=.3)
    ax.axvline(t0 + tE, color='k', alpha=.3)
    ax.axvline(t0 - tE, color='k', alpha=.3)


def plot_ulens_eta_residual_minmax_vs_opt():
    stats_obs = return_ulens_stats(observableFlag=True,
                                   bhFlag=False)
    eta_residual_level2_obs = stats_obs['eta_residual_level2']
    eta_residual_level3_obs = stats_obs['eta_residual_level3']

    stats_obs_BH = return_ulens_stats(observableFlag=True,
                                      bhFlag=True)
    eta_residual_level2_obs_BH = stats_obs_BH['eta_residual_level2']
    eta_residual_level3_obs_BH = stats_obs_BH['eta_residual_level3']

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].set_title('Observable uLens')
    ax[0].scatter(eta_residual_level2_obs, eta_residual_level3_obs, s=1, alpha=.1)
    ax[1].set_title('Observable BH uLens')
    ax[1].scatter(eta_residual_level2_obs_BH, eta_residual_level3_obs_BH, s=1, alpha=1)

    x = np.linspace(0, 2.5)
    for a in ax:
        a.set_xlabel('eta_residual minmax')
        a.set_ylabel('eta_residual opt')
        a.plot(x, x, alpha=.2, color='k')
        a.set_xlim(0, 2.5)
        a.set_ylim(0, 3)
    fig.tight_layout()

    data_obs = return_ulens_data(observableFlag=True,
                                 bhFlag=False)
    metadata_obs = return_ulens_metadata(observableFlag=True,
                                         bhFlag=False)

    # top left
    cond = eta_residual_level2_obs < 1
    cond *= eta_residual_level3_obs > 1.5
    idx_arr = np.random.choice(np.where(cond == True)[0],
                               replace=False,
                               size=9)

    fig, ax = plt.subplots(5, 2, figsize=(8, 8))
    ax = ax.flatten()
    ax[0].scatter(eta_residual_level2_obs, eta_residual_level3_obs, s=1, alpha=.1)
    for i, idx in enumerate(idx_arr, 1):
        d = data_obs[idx]
        hmjd, mag = d[:, :2].T
        ax[i].scatter(hmjd, mag, s=1)
        _plot_ulens_model(ax[i], stats_obs, metadata_obs, idx, hmjd)
        ax[i].invert_yaxis()
        ax[i].set_xlim(np.min(hmjd), np.max(hmjd))

        ax[0].scatter(eta_residual_level2_obs[idx],
                      eta_residual_level3_obs[idx],
                      s=5, color='r', marker='*')
    fig.tight_layout()

    # top right
    cond = eta_residual_level2_obs > 1.5
    cond *= eta_residual_level3_obs > 1.5
    idx_arr = np.random.choice(np.where(cond == True)[0],
                               replace=False,
                               size=9)

    fig, ax = plt.subplots(5, 2, figsize=(8, 8))
    ax = ax.flatten()
    ax[0].scatter(eta_residual_level2_obs, eta_residual_level3_obs, s=1, alpha=.1)
    for i, idx in enumerate(idx_arr, 1):
        d = data_obs[idx]
        hmjd, mag = d[:, :2].T
        ax[i].scatter(hmjd, mag, s=1, color='b')
        _plot_ulens_model(ax[i], stats_obs, metadata_obs, idx, hmjd)
        ax[i].invert_yaxis()
        ax[i].set_xlim(np.min(hmjd), np.max(hmjd))

        ax[0].scatter(eta_residual_level2_obs[idx],
                      eta_residual_level3_obs[idx],
                      s=5, color='r', marker='*')
    fig.tight_layout()

    # bottom right
    cond = eta_residual_level2_obs > 1.5
    cond *= eta_residual_level3_obs < 1
    idx_arr = np.random.choice(np.where(cond == True)[0],
                               replace=False,
                               size=9)

    fig, ax = plt.subplots(5, 2, figsize=(8, 8))
    ax = ax.flatten()
    ax[0].scatter(eta_residual_level2_obs, eta_residual_level3_obs, s=1, alpha=.1)
    for i, idx in enumerate(idx_arr, 1):
        d = data_obs[idx]
        hmjd, mag = d[:, :2].T
        ax[i].scatter(hmjd, mag, s=1)
        _plot_ulens_model(ax[i], stats_obs, metadata_obs, idx, hmjd)
        ax[i].invert_yaxis()
        ax[i].set_xlim(np.min(hmjd), np.max(hmjd))

        ax[0].scatter(eta_residual_level2_obs[idx],
                      eta_residual_level3_obs[idx],
                      s=5, color='r', marker='*')
    fig.tight_layout()

    # bottom left
    cond = eta_residual_level2_obs < 1
    cond *= eta_residual_level3_obs < 1
    idx_arr = np.random.choice(np.where(cond == True)[0],
                               replace=False,
                               size=9)

    fig, ax = plt.subplots(5, 2, figsize=(8, 8))
    ax = ax.flatten()
    ax[0].scatter(eta_residual_level2_obs, eta_residual_level3_obs, s=1, alpha=.1)
    for i, idx in enumerate(idx_arr, 1):
        d = data_obs[idx]
        hmjd, mag = d[:, :2].T
        ax[i].scatter(hmjd, mag, s=1)
        _plot_ulens_model(ax[i], stats_obs, metadata_obs, idx, hmjd)
        ax[i].invert_yaxis()
        ax[i].set_xlim(np.min(hmjd), np.max(hmjd))
        ax[0].scatter(eta_residual_level2_obs[idx],
                      eta_residual_level3_obs[idx],
                      s=5, color='r', marker='*')
    fig.tight_layout()


def generate_all_figures():
    plot_ulens_opt_corner()
    plot_ulens_opt_inside_outside()
    plot_ulens_tE_opt_bias()
    plot_ulens_eta_residual_minmax_vs_opt()


if __name__ == '__main__':
    generate_all_figures()