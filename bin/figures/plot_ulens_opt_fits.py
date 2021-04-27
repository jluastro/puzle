#! /usr/bin/env python
"""
plot_ulens_opt_fits.py
"""

import corner
import matplotlib.pyplot as plt
import numpy as np
from zort.photometry import magnitudes_to_fluxes, fluxes_to_magnitudes
from microlens.jlu.model import PSPL_Phot_Par_Param1

from puzle.ulens import return_ulens_data, return_ulens_metadata, return_ulens_stats
from puzle.cands import return_cands_eta_resdiual_arrs, fetch_cand_best_obj_by_id
from puzle.models import CandidateLevel2, CandidateLevel3
from puzle.utils import return_figures_dir
from puzle.stats import calculate_chi_squared_inside_outside
from puzle.fit import return_flux_model
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

    bhFlag = False
    stats = return_ulens_stats(observableFlag=True, bhFlag=bhFlag)
    tE_ulens = stats['tE_level3']
    u0_amp_ulens = stats['u0_amp_level3']
    mag_src_ulens = stats['mag_src_level3']
    chi_squared_delta_ulens = stats['chi_squared_delta_level3']
    piE_E_ulens = stats['piE_E_level3']
    piE_N_ulens = stats['piE_N_level3']
    eta_ulens = stats['eta']
    eta_residual_ulens = stats['eta_residual_level3']

    data = return_ulens_data(observableFlag=True, bhFlag=bhFlag)
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
    data_range = [(.5, 4), (-3, 3), (10, 24), (0, 5), (-3, 2), (-3, 2), (-3, 2), (0, 1), (0, 4)]

    fig, ax = plt.subplots(5, 2, figsize=(8, 8))
    fig.suptitle(f'ulens red | cands black | bhFlag {bhFlag}')
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
    fig.suptitle(f'Level 3 Fits (ulens red | cands black | bhFlag {bhFlag})')

    fname = '%s/ulens_opt_corner.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def _plot_cands_model(ax, hmjd, mag, magerr, cand2, cand3):
    t0 = cand3.t0_best
    u0_amp = cand3.u0_amp_best
    tE = cand3.tE_best
    piE_E = cand3.piE_E_best
    piE_N = cand3.piE_N_best
    b_sff = cand3.b_sff_best
    mag_src = cand3.mag_src_best
    raL = cand3.ra
    decL = cand3.dec
    model = PSPL_Phot_Par_Param1(
        t0=t0, u0_amp=u0_amp, tE=tE,
        piE_E=piE_E, piE_N=piE_N,
        b_sff=b_sff, mag_src=mag_src,
        raL=raL, decL=decL)

    hmjd_model = np.linspace(np.min(hmjd), np.max(hmjd), 10000)
    mag_model_opt = model.get_photometry(hmjd_model)
    ax.plot(hmjd_model, mag_model_opt, color='g')
    ax.axvline(t0, color='g', alpha=.3)
    ax.axvline(t0 + tE, color='g', alpha=.3, linestyle='--')
    ax.axvline(t0 - tE, color='g', alpha=.3, linestyle='--')

    t0 = cand2.t_0_best
    tE = cand2.t_E_best
    f0 = cand2.f_0_best
    f1 = cand2.f_1_best
    a_type = cand2.a_type_best

    flux_model_minmax = return_flux_model(hmjd_model, t0, tE, a_type, f0, f1)
    _, fluxerr = magnitudes_to_fluxes(mag, magerr)
    fluxerr_model = np.interp(hmjd_model, hmjd, fluxerr)
    mag_model_minmax, _ = fluxes_to_magnitudes(flux_model_minmax, fluxerr_model)

    ax.plot(hmjd_model, mag_model_minmax, color='r')
    ax.axvline(t0, color='r', alpha=.3)
    ax.axvline(t0 + tE, color='r', alpha=.3, linestyle='--')
    ax.axvline(t0 - tE, color='r', alpha=.3, linestyle='--')


def _plot_ulens_model(ax, hmjd, mag, magerr, stats_obs, metadata_obs, idx):
    t0 = metadata_obs['t0'][idx]
    u0_amp = metadata_obs['u0'][idx]
    tE = metadata_obs['tE'][idx]
    piE_E = metadata_obs['piE_E'][idx]
    piE_N = metadata_obs['piE_N'][idx]
    b_sff = metadata_obs['b_sff'][idx]
    mag_src = metadata_obs['mag_src'][idx]
    raL = metadata_obs['ra'][idx]
    decL = metadata_obs['dec'][idx]
    model = PSPL_Phot_Par_Param1(
        t0=t0, u0_amp=u0_amp, tE=tE,
        piE_E=piE_E, piE_N=piE_N,
        b_sff=b_sff, mag_src=mag_src,
        raL=raL, decL=decL)

    hmjd_model = np.linspace(np.min(hmjd), np.max(hmjd), 10000)
    mag_model_opt = model.get_photometry(hmjd_model)
    ax.plot(hmjd_model, mag_model_opt, color='k', alpha=.3)
    ax.axvline(t0, color='k', alpha=.3)
    ax.axvline(t0 + tE, color='k', alpha=.3, linestyle='--')
    ax.axvline(t0 - tE, color='k', alpha=.3, linestyle='--')

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
    mag_model_opt = model.get_photometry(hmjd_model)
    ax.plot(hmjd_model, mag_model_opt, color='g', alpha=.3)
    ax.axvline(t0, color='g', alpha=.3)
    ax.axvline(t0 + tE, color='g', alpha=.3, linestyle='--')
    ax.axvline(t0 - tE, color='g', alpha=.3, linestyle='--')

    t0 = stats_obs['t0_level2'][idx]
    tE = stats_obs['tE_level2'][idx]
    f0 = stats_obs['f0_level2'][idx]
    f1 = stats_obs['f1_level2'][idx]
    a_type = stats_obs['atype_level2'][idx]

    flux_model_minmax = return_flux_model(hmjd_model, t0, tE, a_type, f0, f1)
    _, fluxerr = magnitudes_to_fluxes(mag, magerr)
    fluxerr_model = np.interp(hmjd_model, hmjd, fluxerr)
    mag_model_minmax, _ = fluxes_to_magnitudes(flux_model_minmax, fluxerr_model)

    ax.plot(hmjd_model, mag_model_minmax, color='r', alpha=.3)
    ax.axvline(t0, color='r', alpha=.3)
    ax.axvline(t0 + tE, color='r', alpha=.3, linestyle='--')
    ax.axvline(t0 - tE, color='r', alpha=.3, linestyle='--')


def _plot_chi_samples_cands(log_reduced_chi_squared_outside_cands,
                            log_reduced_chi_squared_inside_cands,
                            id_cands, tE_factor,
                            outside_low, outside_high,
                            inside_low, inside_high):
    # center cands
    cond = log_reduced_chi_squared_outside_cands > outside_low
    cond *= log_reduced_chi_squared_outside_cands < outside_high
    cond *= log_reduced_chi_squared_inside_cands > inside_low
    cond *= log_reduced_chi_squared_inside_cands < inside_high
    idx_sample_arr = np.random.choice(np.where(cond == True)[0],
                                      replace=False,
                                      size=9)

    fig, ax = plt.subplots(5, 2, figsize=(7, 7))
    fig.suptitle('cands')
    ax = ax.flatten()
    ax[0].hexbin(log_reduced_chi_squared_inside_cands,
                 log_reduced_chi_squared_outside_cands,
                 gridsize=50, mincnt=1)
    ax[0].set_xlabel('Inside %stE LOG \n Reduced Chi Squared' % tE_factor, fontsize=6)
    ax[0].set_ylabel('Outside %stE LOG \n Reduced Chi Squared' % tE_factor, fontsize=6)
    for i, idx in enumerate(idx_sample_arr, 1):
        cand_id = id_cands[idx]
        cand2, cand3 = db.session.query(CandidateLevel2, CandidateLevel3).\
            filter(CandidateLevel2.id==CandidateLevel3.id,
                   CandidateLevel3.id==cand_id).first()
        obj = fetch_cand_best_obj_by_id(cand_id)
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.mag
        magerr = obj.lightcurve.magerr

        ax[i].scatter(hmjd, mag, s=1, color='b')
        _plot_cands_model(ax[i], hmjd, mag, magerr, cand2, cand3)
        ax[i].invert_yaxis()
        ax[i].set_xlim(np.min(hmjd), np.max(hmjd))

        ax[0].scatter(log_reduced_chi_squared_inside_cands[idx],
                      log_reduced_chi_squared_outside_cands[idx],
                      s=5, color='r', marker='*')
    ax[0].set_xlim(-.5, 3.5)
    ax[0].set_ylim(-2, 3)
    fig.tight_layout()
    fig.subplots_adjust(top=.95)


def _plot_chi_samples_ulens(log_reduced_chi_squared_outside_ulens,
                            log_reduced_chi_squared_inside_ulens,
                            data_ulens, stats_ulens, metadata_ulens,
                            idx_ulens, tE_factor,
                            outside_low, outside_high,
                            inside_low, inside_high):
    cond1 = log_reduced_chi_squared_outside_ulens > outside_low
    cond2 = log_reduced_chi_squared_outside_ulens < outside_high
    cond3 = log_reduced_chi_squared_inside_ulens > inside_low
    cond4 = log_reduced_chi_squared_inside_ulens < inside_high
    print('outside > outside_low: %i ulens' % np.sum(cond1))
    print('outside < outside_high: %i ulens' % np.sum(cond2))
    print('inside > inside_low: %i ulens' % np.sum(cond3))
    print('inside < inside_high: %i ulens' % np.sum(cond4))
    cond = cond1 * cond2 * cond3 * cond4
    print('total samples: %i ulens' % np.sum(cond))
    idx_sample_arr = np.random.choice(idx_ulens[cond == True],
                                      replace=False,
                                      size=9)

    fig, ax = plt.subplots(5, 2, figsize=(7, 7))
    fig.suptitle('uLens')
    ax = ax.flatten()
    ax[0].hexbin(log_reduced_chi_squared_inside_ulens,
                 log_reduced_chi_squared_outside_ulens,
                 gridsize=50, mincnt=1)
    ax[0].set_xlabel('Inside %stE LOG \n Reduced Chi Squared' % tE_factor, fontsize=6)
    ax[0].set_ylabel('Outside %stE LOG \n Reduced Chi Squared' % tE_factor, fontsize=6)
    for i, idx in enumerate(idx_sample_arr, 1):
        d = data_ulens[idx]
        hmjd, mag, magerr = d[:, :3].T
        ax[i].scatter(hmjd, mag, s=1, color='b')
        _plot_ulens_model(ax[i], hmjd, mag, magerr, stats_ulens, metadata_ulens, idx)
        ax[i].invert_yaxis()
        ax[i].set_xlim(np.min(hmjd), np.max(hmjd))
        ax[i].set_title('inside: %.2f | outside: %.2f' % (log_reduced_chi_squared_inside_ulens[idx_ulens == idx],
                                                          log_reduced_chi_squared_outside_ulens[idx_ulens == idx]))

        ax[0].scatter(log_reduced_chi_squared_inside_ulens[idx_ulens == idx],
                      log_reduced_chi_squared_outside_ulens[idx_ulens == idx],
                      s=5, color='r', marker='*')
    ax[0].set_xlim(-.5, 3.5)
    ax[0].set_ylim(-2, 3)
    fig.tight_layout()
    fig.subplots_adjust(top=.95)


def plot_ulens_opt_inside_outside():
    tE_factor = 3

    cands_outside_column = getattr(CandidateLevel3, f'chi_squared_outside_{tE_factor}tE_best')
    cands_inside_column = getattr(CandidateLevel3, f'chi_squared_inside_{tE_factor}tE_best')
    cands_inside_num = getattr(CandidateLevel3, f'num_epochs_inside_{tE_factor}tE_best')
    cands = CandidateLevel3.query.with_entities(
        cands_inside_column, cands_outside_column,
        cands_inside_num, CandidateLevel3.num_epochs_best,
        CandidateLevel3.id).\
        filter(cands_inside_column!=0, cands_outside_column!=0).\
        order_by(CandidateLevel3.id).all()
    chi_squared_inside_cands = np.array([c[0] for c in cands])
    chi_squared_outside_cands = np.array([c[1] for c in cands])
    num_epochs_inside_cands = np.array([c[2] for c in cands])
    num_epochs_cands = np.array([c[3] for c in cands])
    id_cands = np.array([c[4] for c in cands])

    reduced_chi_squared_inside_cands = chi_squared_inside_cands / (num_epochs_inside_cands - 1)
    reduced_chi_squared_outside_cands = chi_squared_outside_cands / (num_epochs_cands - num_epochs_inside_cands - 1)
    reduced_chi_squared_ratio_cands = reduced_chi_squared_outside_cands / reduced_chi_squared_inside_cands

    log_reduced_chi_squared_inside_cands = np.log10(reduced_chi_squared_inside_cands)
    log_reduced_chi_squared_outside_cands = np.log10(reduced_chi_squared_outside_cands)
    log_reduced_chi_squared_ratio_cands = np.log10(reduced_chi_squared_ratio_cands)
    # log_reduced_chi_squared_ratio_cands = log_reduced_chi_squared_outside_cands / log_reduced_chi_squared_inside_cands

    bhFlag = False
    data_ulens = return_ulens_data(observableFlag=True, bhFlag=bhFlag)
    stats_ulens = return_ulens_stats(observableFlag=True, bhFlag=bhFlag)
    metadata_ulens = return_ulens_metadata(observableFlag=True, bhFlag=bhFlag)
    # size = min(len(data), len(obj_arr))
    # idx_sample = np.random.choice(np.arange(len(data)), replace=False, size=size)
    idx_sample = np.arange(len(data_ulens))

    chi_squared_inside_ulens = []
    chi_squared_outside_ulens = []
    num_epochs_inside_ulens = []
    num_epochs_ulens = []
    idx_ulens = []
    for idx in idx_sample:
        hmjd = data_ulens[idx][:, 0]
        mag = data_ulens[idx][:, 1]
        magerr = data_ulens[idx][:, 2]
        t0 = stats_ulens['t0_level3'][idx]
        tE = stats_ulens['tE_level3'][idx]
        num_epochs = len(hmjd)

        info = calculate_chi_squared_inside_outside(hmjd, mag, magerr,
                                                    t0, tE, tE_factor=tE_factor)
        if np.any(np.array(info) == 0):
            continue
        chi_squared_inside, chi_squared_outside, num_inside = info
        chi_squared_inside_ulens.append(chi_squared_inside)
        chi_squared_outside_ulens.append(chi_squared_outside)
        num_epochs_inside_ulens.append(num_inside)
        num_epochs_ulens.append(num_epochs)
        idx_ulens.append(idx)

    idx_ulens = np.array(idx_ulens)
    chi_squared_inside_ulens = np.array(chi_squared_inside_ulens)
    chi_squared_outside_ulens = np.array(chi_squared_outside_ulens)
    num_epochs_inside_ulens = np.array(num_epochs_inside_ulens)
    num_epochs_ulens = np.array(num_epochs_ulens)
    
    reduced_chi_squared_inside_ulens = chi_squared_inside_ulens / (num_epochs_inside_ulens - 1)
    reduced_chi_squared_outside_ulens = chi_squared_outside_ulens / (num_epochs_ulens - num_epochs_inside_ulens - 1)
    reduced_chi_squared_ratio_ulens = reduced_chi_squared_outside_ulens / reduced_chi_squared_inside_ulens
    
    log_reduced_chi_squared_inside_ulens = np.log10(reduced_chi_squared_inside_ulens)
    log_reduced_chi_squared_outside_ulens = np.log10(reduced_chi_squared_outside_ulens)
    log_reduced_chi_squared_ratio_ulens = np.log10(reduced_chi_squared_ratio_ulens)
    # log_reduced_chi_squared_ratio_ulens = log_reduced_chi_squared_outside_ulens / log_reduced_chi_squared_inside_ulens

    inout_bins = np.linspace(-1, 3, 25)
    ratio_bins = np.linspace(-3, 3, 25)
    
    fig, ax = plt.subplots(2, 1, figsize=(7, 7))
    for a in ax: a.clear()
    ax[0].set_title('ulens')
    ax[0].scatter(num_epochs_inside_ulens, chi_squared_inside_ulens, label='inside',
                  s=.1, alpha=.1)
    ax[0].scatter(num_epochs_ulens - num_epochs_inside_ulens, chi_squared_outside_ulens, label='outside',
                  s=.1, alpha=.1)
    ax[1].set_title('cands')
    ax[1].scatter(num_epochs_inside_cands, chi_squared_inside_cands, label='inside',
                  s=.1, alpha=.1)
    ax[1].scatter(num_epochs_cands - num_epochs_inside_cands, chi_squared_outside_cands, label='outside',
                  s=.1, alpha=.1)
    for a in ax:
        a.set_yscale('log')
        a.set_xscale('log')
        a.set_xlabel('number of epochs')
        a.set_ylabel('chi squared')
        a.legend(markerscale=20)
    fig.tight_layout()

    fig, ax = plt.subplots(3, 1, figsize=(7, 7))
    fig.suptitle(f'tE factor: {tE_factor}')
    for a in ax: a.clear()
    ax[0].set_title('Inside')
    ax[0].hist(log_reduced_chi_squared_inside_ulens, color='r', bins=inout_bins, histtype='step', density=True, label=f'ulens | BH {bhFlag}')
    ax[0].hist(log_reduced_chi_squared_inside_cands, color='k', bins=inout_bins, histtype='step', density=True, label='cands')
    ax[1].set_title('Outside')
    ax[1].hist(log_reduced_chi_squared_outside_ulens, color='r', bins=inout_bins, histtype='step', density=True, label=f'ulens | BH {bhFlag}')
    ax[1].hist(log_reduced_chi_squared_outside_cands, color='k', bins=inout_bins, histtype='step', density=True, label='cands')
    ax[2].set_title('Ratio')
    ax[2].hist(log_reduced_chi_squared_ratio_ulens, color='r', bins=ratio_bins, histtype='step', density=True, label=f'ulens | BH {bhFlag}')
    ax[2].hist(log_reduced_chi_squared_ratio_cands, color='k', bins=ratio_bins, histtype='step', density=True, label='cands')
    for a in ax:
        a.set_xlabel('LOG Reduced Chi Squared')
        a.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=.9)

    fig, ax = plt.subplots(2, 1, figsize=(7, 7))
    fig.suptitle(f'tE factor: {tE_factor}')
    for a in ax: a.clear()
    ax[0].set_title('ulens')
    ax[0].hexbin(log_reduced_chi_squared_inside_ulens,
                 log_reduced_chi_squared_outside_ulens,
                 gridsize=25, mincnt=1)
    ax[1].set_title('cands')
    ax[1].hexbin(log_reduced_chi_squared_inside_cands,
                 log_reduced_chi_squared_outside_cands,
                 gridsize=50, mincnt=1)
    for a in ax:
        a.set_xlim(-.5, 3.5)
        a.set_ylim(-2, 3)
        a.grid(True)
        a.set_xlabel('Inside %stE LOG Reduced Chi Squared' % tE_factor)
        a.set_ylabel('Outside %stE LOG Reduced Chi Squared' % tE_factor)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)

    delta_ulens = reduced_chi_squared_inside_ulens - reduced_chi_squared_outside_ulens
    delta_cands = reduced_chi_squared_inside_cands - reduced_chi_squared_outside_cands

    fig, ax = plt.subplots(2, 1, figsize=(7, 7))
    fig.suptitle(f'tE factor: {tE_factor}')
    for a in ax: a.clear()
    bins = 150
    ax[0].hist(delta_ulens,
               density=True, color='r', label='ulens', histtype='step', bins=bins)
    ax[0].hist(delta_cands,
               density=True, color='k', label='cands', histtype='step', bins=bins)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('chi_squared_inside - chi_squared_outside')
    ax[0].set_title('%i candidates' % len(delta_cands))
    cond_cands = delta_cands >= np.min(delta_ulens)
    ax[1].set_title('%i candidates' % np.sum(cond_cands))
    ax[1].hist(np.log10(delta_ulens),
               density=True, color='r', label='ulens', histtype='step', bins=bins)
    ax[1].hist(np.log10(delta_cands[cond_cands]),
               density=True, color='k', label='cands', histtype='step', bins=bins)
    ax[1].set_xlabel('LOG chi_squared_inside - chi_squared_outside')
    fig.tight_layout()
    fig.subplots_adjust(top=.9)

    ax[1].scatter(log_reduced_chi_squared_outside_cands,
                 reduced_chi_squared_inside_cands - reduced_chi_squared_outside_cands,
                 s=.1, alpha=.1)

    # center uLens
    _plot_chi_samples_ulens(log_reduced_chi_squared_outside_ulens,
                            log_reduced_chi_squared_inside_ulens,
                            data_ulens, stats_ulens, metadata_ulens,
                            idx_ulens, tE_factor,
                            outside_low=0, outside_high=0.5,
                            inside_low=0.5, inside_high=1.5)

    # high outside uLens
    _plot_chi_samples_ulens(log_reduced_chi_squared_outside_ulens,
                            log_reduced_chi_squared_inside_ulens,
                            data_ulens, stats_ulens, metadata_ulens,
                            idx_ulens, tE_factor,
                            outside_low=2, outside_high=3,
                            inside_low=-3, inside_high=3)

    # high mag uLens
    _plot_chi_samples_ulens(log_reduced_chi_squared_outside_ulens,
                            log_reduced_chi_squared_inside_ulens,
                            data_ulens, stats_ulens, metadata_ulens,
                            idx_ulens, tE_factor,
                            outside_low=0, outside_high=0.5,
                            inside_low=2, inside_high=3)

    # low mag uLens
    _plot_chi_samples_ulens(log_reduced_chi_squared_outside_ulens,
                            log_reduced_chi_squared_inside_ulens,
                            data_ulens, stats_ulens, metadata_ulens,
                            idx_ulens, tE_factor,
                            outside_low=0, outside_high=0.5,
                            inside_low=-.5, inside_high=.5)

    # poorly fit uLens
    _plot_chi_samples_ulens(log_reduced_chi_squared_outside_ulens,
                            log_reduced_chi_squared_inside_ulens,
                            data_ulens, stats_ulens, metadata_ulens,
                            idx_ulens, tE_factor,
                            outside_low=1, outside_high=2.5,
                            inside_low=-.5, inside_high=0.5)

    # center cands
    _plot_chi_samples_cands(log_reduced_chi_squared_outside_cands,
                            log_reduced_chi_squared_inside_cands,
                            id_cands, tE_factor,
                            outside_low=-.4, outside_high=1.5,
                            inside_low=-.5, inside_high=1)

    # high mag cands
    _plot_chi_samples_cands(log_reduced_chi_squared_outside_cands,
                            log_reduced_chi_squared_inside_cands,
                            id_cands, tE_factor,
                            outside_low=-1, outside_high=0,
                            inside_low=1, inside_high=2)

    # flat cands
    _plot_chi_samples_cands(log_reduced_chi_squared_outside_cands,
                            log_reduced_chi_squared_inside_cands,
                            id_cands, tE_factor,
                            outside_low=-2, outside_high=-1,
                            inside_low=0, inside_high=1)


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
    ax[0].scatter(log_tE_modeled, log_tE_measured, s=1, alpha=.1)
    ax[0].set_xlabel('log tE modeled')
    ax[0].set_ylabel('log tE measured')
    ax[0].plot(x0, x0, color='r', alpha=.3)
    ax[0].set_xlim(1.3, 2.6)
    ax[0].set_ylim(0, 3)

    x1 = np.linspace(0, 3, 1000)
    ax[1].scatter(log_tE_measured, log_tE_modeled, s=1, alpha=.1)
    ax[1].set_xlabel('log tE measured')
    ax[1].set_ylabel('log tE modeled')
    ax[1].plot(x1, x1, color='r', alpha=.3)
    ax[1].set_xlim(0, 3)
    ax[1].set_ylim(1.3, 2.6)

    fig.tight_layout()


def _plot_eta_minmax_opt_samples_ulens(eta_residual_minmax_ulens,
                                       eta_residual_opt_ulens,
                                       data_ulens, stats_ulens, metadata_ulens,
                                       minmax_low, minmax_high,
                                       opt_low, opt_high):
    # top left
    cond = eta_residual_minmax_ulens > minmax_low
    cond *= eta_residual_minmax_ulens < minmax_high
    cond *= eta_residual_opt_ulens > opt_low
    cond *= eta_residual_opt_ulens < opt_high
    idx_arr = np.random.choice(np.where(cond == True)[0],
                               replace=False,
                               size=9)

    fig, ax = plt.subplots(5, 2, figsize=(7, 7))
    fig.suptitle('uLens')
    ax = ax.flatten()
    ax[0].scatter(eta_residual_minmax_ulens, eta_residual_opt_ulens, s=1, alpha=.1)
    ax[0].set_xlabel('eta_residual minmax', fontsize=6)
    ax[0].set_ylabel('eta_residual opt', fontsize=6)
    for i, idx in enumerate(idx_arr, 1):
        d = data_ulens[idx]
        hmjd, mag, magerr = d[:, :3].T
        ax[i].scatter(hmjd, mag, s=1, color='b')
        _plot_ulens_model(ax[i], hmjd, mag, magerr, stats_ulens, metadata_ulens, idx)
        ax[i].invert_yaxis()
        ax[i].set_xlim(np.min(hmjd), np.max(hmjd))

        ax[0].scatter(eta_residual_minmax_ulens[idx],
                      eta_residual_opt_ulens[idx],
                      s=5, color='r', marker='*')
    fig.tight_layout()
    fig.subplots_adjust(top=.95)


def _plot_eta_minmax_opt_samples_cands(eta_residual_minmax_cands,
                                       eta_residual_opt_cands,
                                       id_cands,
                                       minmax_low, minmax_high,
                                       opt_low, opt_high):
    # top left
    cond = eta_residual_minmax_cands > minmax_low
    cond *= eta_residual_minmax_cands < minmax_high
    cond *= eta_residual_opt_cands > opt_low
    cond *= eta_residual_opt_cands < opt_high
    idx_arr = np.random.choice(np.where(cond == True)[0],
                               replace=False,
                               size=9)

    fig, ax = plt.subplots(5, 2, figsize=(7, 7))
    fig.suptitle('Cands')
    ax = ax.flatten()
    ax[0].scatter(eta_residual_minmax_cands, eta_residual_opt_cands, s=1, alpha=.1)
    ax[0].set_xlabel('eta_residual minmax', fontsize=6)
    ax[0].set_ylabel('eta_residual opt', fontsize=6)
    for i, idx in enumerate(idx_arr, 1):
        cand_id = id_cands[idx]
        cand2, cand3 = db.session.query(CandidateLevel2, CandidateLevel3).\
            filter(CandidateLevel2.id==CandidateLevel3.id,
                   CandidateLevel3.id==cand_id).first()
        obj = fetch_cand_best_obj_by_id(cand_id)
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.mag
        magerr = obj.lightcurve.magerr

        ax[i].scatter(hmjd, mag, s=1, color='b')
        _plot_cands_model(ax[i], hmjd, mag, magerr, cand2, cand3)
        ax[i].invert_yaxis()
        ax[i].set_xlim(np.min(hmjd), np.max(hmjd))

        ax[0].scatter(eta_residual_minmax_cands[idx],
                      eta_residual_opt_cands[idx],
                      s=5, color='r', marker='*')
    fig.tight_layout()
    fig.subplots_adjust(top=.95)


def plot_ulens_eta_residual_minmax_vs_opt():
    stats_ulens = return_ulens_stats(observableFlag=True,
                                   bhFlag=False)
    eta_residual_minmax_ulens = stats_ulens['eta_residual_level2']
    eta_residual_opt_ulens = stats_ulens['eta_residual_level3']

    stats_ulens_BH = return_ulens_stats(observableFlag=True,
                                      bhFlag=True)
    eta_residual_minmax_ulens_BH = stats_ulens_BH['eta_residual_level2']
    eta_residual_opt_ulens_BH = stats_ulens_BH['eta_residual_level3']

    eta_residual_minmax_cands, eta_residual_opt_cands = return_cands_eta_resdiual_arrs()

    # scatter
    fig, ax = plt.subplots(3, 1, figsize=(8, 7))
    for a in ax: a.clear()
    ax[0].set_title('Cands')
    ax[0].scatter(eta_residual_minmax_cands, eta_residual_opt_cands,
                  color='k', s=.1, alpha=.1,)
    ax[1].set_title('Observable uLens')
    ax[1].scatter(eta_residual_minmax_ulens, eta_residual_opt_ulens,
                  color='k', s=1, alpha=.1,)
    ax[2].set_title('Observable BH uLens')
    ax[2].scatter(eta_residual_minmax_ulens_BH, eta_residual_opt_ulens_BH,
                  color='k', s=1, alpha=.5)
    x = np.linspace(0, 2.5)
    for a in ax:
        a.set_xlim(0, 4)
        a.set_ylim(0, 3)
        a.set_xlabel('eta_residual minmax')
        a.set_ylabel('eta_residual opt')
        a.plot(x, x, alpha=.3, color='r')
        a.set_xlim(0, 2.5)
        a.set_ylim(0, 3)
    fig.tight_layout()

    # hexbin
    fig, ax = plt.subplots(3, 1, figsize=(8, 7))
    for a in ax: a.clear()
    ax[0].set_title('Cands')
    ax[0].hexbin(eta_residual_minmax_cands, eta_residual_opt_cands,
                 gridsize=40, mincnt=1)
    ax[1].set_title('Observable uLens')
    ax[1].hexbin(eta_residual_minmax_ulens, eta_residual_opt_ulens,
                 gridsize=40, mincnt=1)
    ax[2].set_title('Observable BH uLens')
    ax[2].hexbin(eta_residual_minmax_ulens, eta_residual_opt_ulens,
                 gridsize=40, mincnt=1)
    x = np.linspace(0, 2.5)
    for a in ax:
        a.set_xlim(0, 4)
        a.set_ylim(0, 3)
        a.set_xlabel('eta_residual minmax')
        a.set_ylabel('eta_residual opt')
        a.plot(x, x, alpha=.3, color='r')
        a.set_xlim(0, 2.5)
        a.set_ylim(0, 3)
    fig.tight_layout()

    data_ulens = return_ulens_data(observableFlag=True,
                                 bhFlag=False)
    metadata_ulens = return_ulens_metadata(observableFlag=True,
                                         bhFlag=False)

    # top left
    _plot_eta_minmax_opt_samples_ulens(eta_residual_minmax_ulens,
                                       eta_residual_opt_ulens,
                                       data_ulens, stats_ulens, metadata_ulens,
                                       minmax_low=0, minmax_high=1,
                                       opt_low=1.5, opt_high=3)

    # top right
    _plot_eta_minmax_opt_samples_ulens(eta_residual_minmax_ulens,
                                       eta_residual_opt_ulens,
                                       data_ulens, stats_ulens, metadata_ulens,
                                       minmax_low=1.5, minmax_high=2.5,
                                       opt_low=1.5, opt_high=3)

    # bottom right
    _plot_eta_minmax_opt_samples_ulens(eta_residual_minmax_ulens,
                                       eta_residual_opt_ulens,
                                       data_ulens, stats_ulens, metadata_ulens,
                                       minmax_low=1.5, minmax_high=2.5,
                                       opt_low=0, opt_high=1)

    # bottom left
    _plot_eta_minmax_opt_samples_ulens(eta_residual_minmax_ulens,
                                       eta_residual_opt_ulens,
                                       data_ulens, stats_ulens, metadata_ulens,
                                       minmax_low=0, minmax_high=1,
                                       opt_low=0, opt_high=1)


def generate_all_figures():
    plot_ulens_opt_corner()
    plot_ulens_opt_inside_outside()
    plot_ulens_tE_opt_bias()
    plot_ulens_eta_residual_minmax_vs_opt()


if __name__ == '__main__':
    generate_all_figures()