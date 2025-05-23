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
from puzle.cands import return_cands_eta_resdiual_arrs, \
    fetch_cand_best_obj_by_id, calculate_chi2_model, apply_level3_cuts_to_query, \
    return_sigma_peaks
from puzle.models import CandidateLevel2, CandidateLevel3
from puzle.utils import return_figures_dir
from puzle.stats import calculate_chi_squared_inside_outside, average_xy_on_round_x, return_CDF
from puzle.fit import return_flux_model
from puzle import db


def plot_ulens_opt_corner():
    query = CandidateLevel3.query.with_entities(CandidateLevel3.tE_best,
                                                 CandidateLevel3.u0_amp_best,
                                                 CandidateLevel3.mag_src_best,
                                                 CandidateLevel3.chi_squared_ulens_best,
                                                 CandidateLevel3.piE_E_best,
                                                 CandidateLevel3.piE_N_best,
                                                 CandidateLevel3.eta_best,
                                                 CandidateLevel3.eta_residual_best,
                                                 CandidateLevel3.num_epochs_best,
                                                 CandidateLevel3.t0_best)
    cands3 = apply_level3_cuts_to_query(query).all()
    tE_cands = np.array([c[0] for c in cands3], dtype=np.float32)
    u0_amp_cands = np.array([c[1] for c in cands3], dtype=np.float32)
    mag_src_cands = np.array([c[2] for c in cands3], dtype=np.float32)
    chi_squared_ulens_cands = np.array([c[3] for c in cands3], dtype=np.float32)
    piE_E_cands = np.array([c[4] for c in cands3], dtype=np.float32)
    piE_N_cands = np.array([c[5] for c in cands3], dtype=np.float32)
    eta_cands = np.array([c[6] for c in cands3], dtype=np.float32)
    eta_residual_cands = np.array([c[7] for c in cands3], dtype=np.float32)
    num_epochs_cands = np.array([c[8] for c in cands3], dtype=np.float32)
    t0_cands = np.array([c[9] for c in cands3], dtype=np.float32)

    log_tE_cands = np.log10(tE_cands)
    log_piE_E_cands = np.log10(piE_E_cands)
    log_piE_N_cands = np.log10(piE_N_cands)
    piE_cands = np.hypot(piE_E_cands, piE_N_cands)
    log_piE_cands = np.log10(piE_cands)
    
    chi_squared_ulens_reduced_cands = chi_squared_ulens_cands / num_epochs_cands

    data_cands = np.vstack((t0_cands,
                            log_tE_cands,
                            u0_amp_cands,
                            mag_src_cands,
                            chi_squared_ulens_reduced_cands,
                            log_piE_E_cands,
                            log_piE_N_cands,
                            log_piE_cands,
                            eta_cands,
                            eta_residual_cands)).T

    bhFlag = True
    level3Flag = True
    stats = return_ulens_stats(observableFlag=True, bhFlag=bhFlag, level3Flag=level3Flag)
    tE_ulens = stats['tE_level3']
    u0_amp_ulens = stats['u0_amp_level3']
    mag_src_ulens = stats['mag_src_level3']
    chi_squared_ulens_ulens = stats['chi_squared_ulens_level3']
    piE_E_ulens = stats['piE_E_level3']
    piE_N_ulens = stats['piE_N_level3']
    eta_ulens = stats['eta']
    eta_residual_ulens = stats['eta_residual_level3']
    t0_ulens = stats['t0_level3']

    data = return_ulens_data(observableFlag=True, bhFlag=bhFlag, level3Flag=level3Flag)
    num_epochs_ulens = np.array([len(d) for d in data])
    
    log_tE_ulens = np.log10(tE_ulens)
    log_piE_E_ulens = np.log10(piE_E_ulens)
    log_piE_N_ulens = np.log10(piE_N_ulens)
    piE_ulens = np.hypot(piE_E_ulens, piE_N_ulens)
    log_piE_ulens = np.log10(piE_ulens)
    
    chi_squared_ulens_reduced_ulens = chi_squared_ulens_ulens / num_epochs_ulens

    data_ulens = np.vstack((t0_ulens,
                            log_tE_ulens,
                            u0_amp_ulens,
                            mag_src_ulens,
                            chi_squared_ulens_reduced_ulens,
                            log_piE_E_ulens,
                            log_piE_N_ulens,
                            log_piE_ulens,
                            eta_ulens,
                            eta_residual_ulens)).T

    # Plot it.
    labels = ['t0', 'LOG tE', 'u0_amp', 'mag_src', 'chi_squared_ulens_reduced', 'LOG piE_E', 'LOG piE_N', 'LOG piE', 'eta', 'eta_residual']
    data_range = [(57000, 60000), (.5, 4), (-3, 3), (10, 24), (0, 5), (-3, 2), (-3, 2), (-3, 2), (0, 1), (0, 4)]

    fig, ax = plt.subplots(5, 2, figsize=(8, 8))
    fig.suptitle(f'ulens red | cands black | bhFlag {bhFlag}')
    ax = ax.flatten()
    for i, label in enumerate(labels):
        ax[i].set_title(label)
        bins = np.linspace(data_range[i][0],
                           data_range[i][1], 25)
        ax[i].hist(data_ulens[:, i], histtype='step',
                   bins=bins, color='r', density=True)
        ax[i].hist(data_cands[:, i], histtype='step',
                   bins=bins, color='k', density=True)
        ax[i].set_xlim(data_range[i])
    fig.tight_layout()

    fig = corner.corner(data_ulens, labels=labels, range=data_range,
                        show_titles=True, title_kwargs={"fontsize": 6},
                        label_kwargs={'fontsize': 6}, color='r')
    fig = corner.corner(data_cands, labels=labels, range=data_range,
                        show_titles=True, title_kwargs={"fontsize": 6},
                        label_kwargs={'fontsize': 6}, color='k',
                        fig=fig)
    fig.suptitle(f'Level 3 Fits (ulens red | cands black | bhFlag {bhFlag})')

    fname = '%s/ulens_opt_corner.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def _plot_cands_model(ax, hmjd, mag, magerr, cand3, cand2=None):
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

    if cand2 is not None:
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


def _plot_ulens_model(ax, hmjd, mag, magerr, stats_obs, metadata_obs, idx,
                      plot_level2=True):
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

    if plot_level2:
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
        _plot_cands_model(ax[i], hmjd, mag, magerr, cand3, cand2)
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


def plot_ulens_opt_chi2_ulens_cut():
    cands = CandidateLevel3.query.with_entities(CandidateLevel3.chi_squared_ulens_best,
                                                CandidateLevel3.num_days_best).all()
    chi2_opt_cands = np.array([c[0] for c in cands])
    num_days_cands = np.array([c[1] for c in cands])
    reduced_chi2_opt_cands = chi2_opt_cands / num_days_cands

    bhFlag = False
    data = return_ulens_data(observableFlag=True, bhFlag=bhFlag)
    metadata = return_ulens_metadata(observableFlag=True, bhFlag=bhFlag)
    stats = return_ulens_stats(observableFlag=True, bhFlag=bhFlag)

    chi2_measured_ulens = stats['chi_squared_ulens_level3']
    num_days_ulens = np.array([len(np.unique(np.floor(d))) for d in data])
    reduced_chi2_measured_ulens = chi2_measured_ulens / num_days_ulens

    plot_ulens_modeled = False
    if plot_ulens_modeled:
        param_names = ['t0', 'u0_amp', 'tE', 'mag_src',
                       'b_sff', 'piE_E', 'piE_N']
        model_class = PSPL_Phot_Par_Param1
        chi2_modeled_ulens = []
        dof_arr = []
        idx_arr = np.arange(len(data))
        for idx in idx_arr:
            if idx % 10000 == 0:
                print('-- ulens sample %i / %i' % (idx, len(idx_arr)))
            hmjd, mag, magerr = data[idx][:, :3].T
            hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
            _, magerr_round = average_xy_on_round_x(hmjd, magerr)

            t0 = metadata['t0'][idx]
            u0 = metadata['u0'][idx]
            tE = metadata['tE'][idx]
            mag_src = metadata['mag_src'][idx]
            piE_E = metadata['piE_E'][idx]
            piE_N = metadata['piE_N'][idx]
            b_sff = metadata['b_sff'][idx]
            ra = metadata['ra'][idx]
            dec = metadata['dec'][idx]

            data_fit = {'hmjd': hmjd_round, 'mag': mag_round, 'magerr': magerr_round, 'raL': ra, 'decL': dec}
            param_values = [t0, u0, tE, mag_src, b_sff, piE_E, piE_N]
            chi2 = calculate_chi2_model(param_values, param_names, model_class, data_fit)
            dof = len(set(np.round(hmjd)))
            chi2_modeled_ulens.append(chi2)
            dof_arr.append(dof)

        chi2_modeled_ulens = np.array(chi2_modeled_ulens)
        dof_arr = np.array(dof_arr)
        reduced_chi2_modeled_ulens = chi2_modeled_ulens / dof_arr

    chi2_thresh = np.percentile(reduced_chi2_measured_ulens, 95)
    cand_frac = 100 * np.sum(reduced_chi2_opt_cands <= chi2_thresh) / len(reduced_chi2_opt_cands)

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle('%.2f%% Percent of Cands below %.3f (95th Percentile)' % (cand_frac, chi2_thresh))
    for a in ax: a.clear()
    bins = np.linspace(0, 5, 50)
    ax[0].hist(reduced_chi2_measured_ulens, label='ulens measured',
               bins=bins, histtype='step', density=True)
    ax[0].hist(reduced_chi2_opt_cands, label='cands measured',
               bins=bins, histtype='step', density=True)
    if plot_ulens_modeled:
        ax[0].hist(reduced_chi2_modeled_ulens, label='ulens modeled',
                   bins=bins, histtype='step', density=True)
    ax[0].axvline(chi2_thresh, color='k', alpha=.3, label='ulens measured 95th percentile')
    ax[1].plot(*return_CDF(reduced_chi2_measured_ulens), label='ulens measured')
    ax[1].plot(*return_CDF(reduced_chi2_opt_cands), label='cands measured')
    if plot_ulens_modeled:
        ax[1].plot(*return_CDF(reduced_chi2_modeled_ulens), label='ulens modeled')
    ax[1].axhline(0.95, color='k', alpha=.3, label='ulens measured 95th percentile')
    ax[1].axvline(chi2_thresh, color='k', alpha=.3)

    for a in ax:
        a.grid(True)
        a.set_xlim(0, 5)
        a.set_xlabel('Reduced Chi Squared')
        a.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=.95)

    fname = '%s/ulens_opt_chi2_ulens_cut.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_ulens_opt_chi2_flat_cut():
    cands = CandidateLevel3.query.with_entities(CandidateLevel3.chi_squared_flat_outside_2tE_best,
                                                CandidateLevel3.num_days_outside_2tE_best).\
        filter(CandidateLevel3.num_days_outside_2tE_best!=0).all()
    chi2_flat_outside_cands = np.array([c[0] for c in cands])
    num_days_outside_cands = np.array([c[1] for c in cands])
    reduced_chi2_flat_outside_cands = chi2_flat_outside_cands / num_days_outside_cands

    bhFlag = False
    stats = return_ulens_stats(observableFlag=True, bhFlag=bhFlag)

    chi2_flat_outside_ulens = stats['chi_squared_outside_level3']
    num_days_outside_ulens = stats['num_days_outside_level3']
    cond_nonzero = num_days_outside_ulens != 0
    reduced_chi2_flat_outside_ulens = chi2_flat_outside_ulens[cond_nonzero] / num_days_outside_ulens[cond_nonzero]

    chi2_thresh = np.percentile(reduced_chi2_flat_outside_ulens, 95)
    cand_frac = 100 * np.sum(reduced_chi2_flat_outside_cands <= chi2_thresh) / len(reduced_chi2_flat_outside_cands)

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle('%.2f%% Percent of Cands below %.3f (95th Percentile)' % (cand_frac, chi2_thresh))
    for a in ax: a.clear()
    bins = np.linspace(0, 5, 50)
    ax[0].hist(reduced_chi2_flat_outside_ulens, label='ulens measured',
               bins=bins, histtype='step', density=True)
    ax[0].hist(reduced_chi2_flat_outside_cands, label='cands measured',
               bins=bins, histtype='step', density=True)
    ax[0].axvline(chi2_thresh, color='k', alpha=.3, label='ulens measured 95th percentile')
    ax[1].plot(*return_CDF(reduced_chi2_flat_outside_ulens), label='ulens measured')
    ax[1].plot(*return_CDF(reduced_chi2_flat_outside_cands), label='cands measured')
    ax[1].axhline(0.95, color='k', alpha=.3, label='ulens measured 95th percentile')
    ax[1].axvline(chi2_thresh, color='k', alpha=.3)

    for a in ax:
        a.grid(True)
        a.set_xlim(0, 5)
        a.set_xlabel('Reduced Chi Squared')
        a.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=.95)

    fname = '%s/ulens_opt_chi2_flat_cut.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_ulens_opt_tE_cut():
    cands = CandidateLevel3.query.with_entities(CandidateLevel3.tE_best).\
        filter(CandidateLevel3.tE_best>0).all()
    # cands = CandidateLevel3.query.with_entities(CandidateLevel3.tE_best).\
    #     filter(CandidateLevel3.tE_best>0,CandidateLevel3.delta_hmjd_outside_2tE_best >= 2 * CandidateLevel3.tE_best).all()
    tE_level3_cands = np.array([c[0] for c in cands])
    log_tE_level3_cands = np.log10(tE_level3_cands)

    bhFlag = False
    metadata = return_ulens_metadata(observableFlag=True, bhFlag=bhFlag)
    stats = return_ulens_stats(observableFlag=True, bhFlag=bhFlag)

    tE_modeled_ulens = metadata['tE']
    log_tE_modeled_ulens = np.log10(tE_modeled_ulens)
    tE_level3_ulens = stats['tE_level3']
    log_tE_level3_ulens = np.log10(tE_level3_ulens)

    tE_thresh = np.percentile(log_tE_level3_ulens, 99)
    cand_frac = 100 * np.sum(log_tE_level3_cands <= tE_thresh) / len(log_tE_level3_cands)

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle('%.2f%% Percent of Cands below %.3f (99th Percentile)' % (cand_frac, tE_thresh))
    for a in ax: a.clear()
    bins = np.linspace(0, 4, 50)
    ax[0].hist(log_tE_level3_ulens, label='ulens measured',
               bins=bins, histtype='step', density=True)
    ax[0].hist(log_tE_modeled_ulens, label='ulens modeled',
               bins=bins, histtype='step', density=True)
    ax[0].hist(log_tE_level3_cands, label='cands measured',
               bins=bins, histtype='step', density=True)
    ax[0].axvline(tE_thresh, color='k', alpha=.3, label='ulens measured 99th percentile')
    ax[1].plot(*return_CDF(log_tE_level3_ulens), label='ulens measured')
    ax[1].plot(*return_CDF(log_tE_modeled_ulens), label='ulens modeled')
    ax[1].plot(*return_CDF(log_tE_level3_cands), label='cands measured')
    ax[1].axhline(0.99, color='k', alpha=.3, label='ulens measured 99th percentile')
    ax[1].axvline(tE_thresh, color='k', alpha=.3)

    for a in ax:
        a.grid(True)
        a.set_xlim(0, 4)
        a.set_xlabel('Einstein Crossing Time')
        a.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=.95)

    fname = '%s/ulens_opt_tE_cut.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_ulens_opt_piE_cut():
    cands = CandidateLevel3.query.with_entities(CandidateLevel3.piE_N_best,
                                                CandidateLevel3.piE_E_best).\
        filter(CandidateLevel3.tE_best>0).all()
    piE_N = np.array([c[0] for c in cands])
    piE_E = np.array([c[1] for c in cands])
    piE_measured_cands = np.hypot(piE_N, piE_E)
    log_piE_measured_cands = np.log10(piE_measured_cands)

    bhFlag = False
    metadata = return_ulens_metadata(observableFlag=True, bhFlag=bhFlag)
    stats = return_ulens_stats(observableFlag=True, bhFlag=bhFlag)

    piE_modeled_ulens = np.hypot(metadata['piE_E'], metadata['piE_N'])
    log_piE_modeled_ulens = np.log10(piE_modeled_ulens)
    piE_measured_ulens = np.hypot(stats['piE_E_level3'], stats['piE_N_level3'])
    log_piE_measured_ulens = np.log10(piE_measured_ulens)

    log_piE_thresh = np.percentile(log_piE_measured_ulens, 95)
    cand_frac = 100 * np.sum(log_piE_measured_cands <= log_piE_thresh) / len(log_piE_measured_cands)

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle('%.2f%% Percent of Cands below %.3f (95th Percentile)' % (cand_frac, log_piE_thresh))
    for a in ax: a.clear()
    bins = np.linspace(-3, 3, 50)
    ax[0].hist(log_piE_measured_ulens, label='ulens measured',
               bins=bins, histtype='step', density=True)
    ax[0].hist(log_piE_modeled_ulens, label='ulens modeled',
               bins=bins, histtype='step', density=True)
    ax[0].hist(log_piE_measured_cands, label='cands measured',
               bins=bins, histtype='step', density=True)
    ax[0].axvline(log_piE_thresh, color='k', alpha=.3, label='ulens measured 99th percentile')
    ax[1].plot(*return_CDF(log_piE_measured_ulens), label='ulens measured')
    ax[1].plot(*return_CDF(log_piE_modeled_ulens), label='ulens modeled')
    ax[1].plot(*return_CDF(log_piE_measured_cands), label='cands measured')
    ax[1].axhline(0.95, color='k', alpha=.3, label='ulens measured 99th percentile')
    ax[1].axvline(log_piE_thresh, color='k', alpha=.3)

    for a in ax:
        a.grid(True)
        a.set_xlim(-3, 3)
        a.set_xlabel('LOG Einstein Parallax')
        a.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=.95)

    fname = '%s/ulens_opt_piE_cut.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_ulens_opt_inside_outside():
    tE_factor = 3

    cands_outside_column = getattr(CandidateLevel3, f'chi_squared_flat_outside_{tE_factor}tE_best')
    cands_inside_column = getattr(CandidateLevel3, f'chi_squared_flat_inside_{tE_factor}tE_best')
    cands_inside_num = getattr(CandidateLevel3, f'num_days_inside_{tE_factor}tE_best')
    cands = CandidateLevel3.query.with_entities(
        cands_inside_column, cands_outside_column,
        cands_inside_num, CandidateLevel3.num_days_best,
        CandidateLevel3.id).\
        filter(cands_inside_column!=0, cands_outside_column!=0).\
        order_by(CandidateLevel3.id).all()
    chi_squared_inside_cands = np.array([c[0] for c in cands])
    chi_squared_outside_cands = np.array([c[1] for c in cands])
    num_days_inside_cands = np.array([c[2] for c in cands])
    num_days_cands = np.array([c[3] for c in cands])
    id_cands = np.array([c[4] for c in cands])

    reduced_chi_squared_inside_cands = chi_squared_inside_cands / (num_days_inside_cands - 1)
    reduced_chi_squared_outside_cands = chi_squared_outside_cands / (num_days_cands - num_days_inside_cands - 1)
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
    num_days_inside_ulens = []
    num_days_ulens = []
    idx_ulens = []
    for idx in idx_sample:
        hmjd = data_ulens[idx][:, 0]
        mag = data_ulens[idx][:, 1]
        magerr = data_ulens[idx][:, 2]
        t0 = stats_ulens['t0_level3'][idx]
        tE = stats_ulens['tE_level3'][idx]
        num_days = len(set(np.round(hmjd)))

        info = calculate_chi_squared_inside_outside(hmjd, mag, magerr,
                                                    t0, tE, tE_factor=tE_factor)
        if np.any(np.array(info) == 0):
            continue
        chi_squared_inside, chi_squared_outside, num_days_inside, num_days_outside, delta_hmjd_outside = info
        chi_squared_inside_ulens.append(chi_squared_inside)
        chi_squared_outside_ulens.append(chi_squared_outside)
        num_days_inside_ulens.append(num_days_inside)
        num_days_ulens.append(num_days)
        idx_ulens.append(idx)

    idx_ulens = np.array(idx_ulens)
    chi_squared_inside_ulens = np.array(chi_squared_inside_ulens)
    chi_squared_outside_ulens = np.array(chi_squared_outside_ulens)
    num_days_inside_ulens = np.array(num_days_inside_ulens)
    num_days_ulens = np.array(num_days_ulens)
    
    reduced_chi_squared_inside_ulens = chi_squared_inside_ulens / (num_days_inside_ulens - 1)
    reduced_chi_squared_outside_ulens = chi_squared_outside_ulens / (num_days_ulens - num_days_inside_ulens - 1)
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
    ax[0].scatter(num_days_inside_ulens, chi_squared_inside_ulens, label='inside',
                  s=.1, alpha=.1)
    ax[0].scatter(num_days_ulens - num_days_inside_ulens, chi_squared_outside_ulens, label='outside',
                  s=.1, alpha=.1)
    ax[1].set_title('cands')
    ax[1].scatter(num_days_inside_cands, chi_squared_inside_cands, label='inside',
                  s=.1, alpha=.1)
    ax[1].scatter(num_days_cands - num_days_inside_cands, chi_squared_outside_cands, label='outside',
                  s=.1, alpha=.1)
    for a in ax:
        a.set_yscale('log')
        a.set_xscale('log')
        a.set_xlabel('number of days')
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

    fig, ax = plt.subplots(figsize=(8, 6))
    x0 = np.linspace(1.3, 2.6, 1000)
    ax.set_title('Microlensing Simulations')
    ax.scatter(log_tE_modeled, log_tE_measured, s=1, alpha=.1)
    ax.set_xlabel('log tE modeled')
    ax.set_ylabel('log tE measured')
    ax.plot(x0, x0, color='r', alpha=.3)
    ax.set_xlim(1.3, 2.6)
    ax.set_ylim(0.5, 3)
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
        _plot_cands_model(ax[i], hmjd, mag, magerr, cand3, cand2)
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


def _plot_sigma_peaks_ulens(data, stats, metadata, cond, sigma_peaks_ulens, sigma_factor, sigma_peaks_thresh):
    idx_arr = np.random.choice(np.where(cond == True)[0], replace=False, size=10)
    fig, ax = plt.subplots(5, 2, figsize=(7, 7))
    fig.suptitle(f'ulens {sigma_peaks_thresh} inside points above {sigma_factor} sigma',
                 fontsize=10)
    ax = ax.flatten()
    for i, idx in enumerate(idx_arr):
        d = data[idx]
        hmjd, mag, magerr = d[:, :3].T
        sigma_peaks = sigma_peaks_ulens[idx]
        ax[i].set_title(f'{sigma_peaks} Peaks | idx {idx}')
        ax[i].scatter(hmjd, mag, s=1, color='b')
        _plot_ulens_model(ax[i], hmjd, mag, magerr, stats, metadata, idx,
                          plot_level2=False)
        ax[i].invert_yaxis()
        ax[i].set_xlim(np.min(hmjd), np.max(hmjd))
    fig.tight_layout()
    fig.subplots_adjust(top=.93)


def _plot_sigma_peaks_cands(cond, cands3, id_cands, sigma_peaks_cands, sigma_factor, sigma_peaks_thresh):
    idx_arr = np.random.choice(np.where(cond == True)[0], replace=False, size=10)
    fig, ax = plt.subplots(5, 2, figsize=(7, 7))
    fig.suptitle(f'cands {sigma_peaks_thresh} inside points above {sigma_factor} sigma',
                 fontsize=10)
    ax = ax.flatten()
    for i, idx in enumerate(idx_arr):
        cand_id = id_cands[idx]
        sigma_peaks = sigma_peaks_cands[idx]
        cand3 = cands3[idx]

        obj = fetch_cand_best_obj_by_id(cand_id)
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.mag
        magerr = obj.lightcurve.magerr

        ax[i].set_title(f'{sigma_peaks} Peaks')
        ax[i].scatter(hmjd, mag, s=1, color='b')
        _plot_cands_model(ax[i], hmjd, mag, magerr, cand3)
        ax[i].invert_yaxis()
        ax[i].set_xlim(np.min(hmjd), np.max(hmjd))
    fig.tight_layout()
    fig.subplots_adjust(top=.93)


def plot_ulens_opt_sigma_peaks():
    # query = apply_level3_cuts_to_query(CandidateLevel3.query)
    query = CandidateLevel3.query
    cands3 = query.\
        filter(CandidateLevel3.tE_best!=0).\
        order_by(CandidateLevel3.id).all()
    id_cands = np.array([c.id for c in cands3])
    num_3sigma_peaks_inside_cands = np.array([c.num_3sigma_peaks_inside_2tE_best for c in cands3], dtype=int)
    num_5sigma_peaks_inside_cands = np.array([c.num_5sigma_peaks_inside_2tE_best for c in cands3], dtype=int)
    num_3sigma_peaks_outside_cands = np.array([c.num_3sigma_peaks_outside_2tE_best for c in cands3], dtype=int)
    num_5sigma_peaks_outside_cands = np.array([c.num_5sigma_peaks_outside_2tE_best for c in cands3], dtype=int)
    t0_cands = np.array([c.t0_best for c in cands3])
    tE_cands = np.array([c.tE_best for c in cands3])
    piE_cands = np.array([np.hypot(c.piE_E_best, c.piE_N_best) for c in cands3])
    num_epochs_cands = np.array([c.num_epochs_best for c in cands3])
    chi_squared_cands = np.array([c.chi_squared_ulens_best for c in cands3])
    reduced_chi_squared_cands = chi_squared_cands / num_epochs_cands

    cond1 = tE_cands <= 595
    cond2 = reduced_chi_squared_cands <= 2.221
    cond3 = piE_cands <= 2.877
    cond4 = num_3sigma_peaks_inside_cands > 5
    cond = cond1 * cond2 * cond3 * cond4
    print('cond1:', np.sum(cond1))
    print('cond2:', np.sum(cond1 * cond2))
    print('cond3:', np.sum(cond1 * cond2 * cond3))
    print('cond4:', np.sum(cond1 * cond2 * cond3 * cond4))

    bhFlag = False
    data = return_ulens_data(observableFlag=True, bhFlag=bhFlag, level3Flag=True)
    stats = return_ulens_stats(observableFlag=True, bhFlag=bhFlag, level3Flag=True)
    metadata = return_ulens_metadata(observableFlag=True, bhFlag=bhFlag, level3Flag=True)
    t0_ulens = stats['t0_level3']
    tE_ulens = stats['tE_level3']

    num_3sigma_peaks_inside_ulens = []
    num_5sigma_peaks_inside_ulens = []
    num_3sigma_peaks_outside_ulens = []
    num_5sigma_peaks_outside_ulens = []
    idx_arr = np.arange(len(t0_ulens))
    for idx in idx_arr:
        if idx % 1000 == 0:
            print('ulens', idx, len(idx_arr))
        hmjd, mag, magerr = data[idx][:, :3].T
        t0 = t0_ulens[idx]
        tE = tE_ulens[idx]

        sigma_peaks_inside, sigma_peaks_outside = return_sigma_peaks(hmjd, mag, t0, tE,
                                                                     tE_factor=2,
                                                                     sigma_factor=3)
        num_3sigma_peaks_inside_ulens.append(sigma_peaks_inside)
        num_3sigma_peaks_outside_ulens.append(sigma_peaks_outside)
        sigma_peaks_inside,sigma_peaks_outside = return_sigma_peaks(hmjd, mag, t0, tE,
                                                                    tE_factor=2,
                                                                    sigma_factor=5)
        num_5sigma_peaks_inside_ulens.append(sigma_peaks_inside)
        num_5sigma_peaks_outside_ulens.append(sigma_peaks_outside)
    num_3sigma_peaks_inside_ulens = np.array(num_3sigma_peaks_inside_ulens)
    num_5sigma_peaks_inside_ulens = np.array(num_5sigma_peaks_inside_ulens)
    num_3sigma_peaks_outside_ulens = np.array(num_3sigma_peaks_outside_ulens)
    num_5sigma_peaks_outside_ulens = np.array(num_5sigma_peaks_outside_ulens)

    for sigma_factor in [3, 5]:
        if sigma_factor == 3:
            sigma_peaks_inside_ulens = num_3sigma_peaks_inside_ulens
            sigma_peaks_inside_cands = num_3sigma_peaks_inside_cands
        elif sigma_factor == 5:
            sigma_peaks_inside_ulens = num_5sigma_peaks_inside_ulens
            sigma_peaks_inside_cands = num_5sigma_peaks_inside_cands

        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        for a in ax: a.clear()
        bins = np.linspace(1, 100, 50)
        ax[0].set_title(f'{sigma_factor} Sigma Inside Peaks')
        ax[0].hist(sigma_peaks_inside_cands, color='k', label='cands', histtype='step', bins=bins, density=True)
        ax[0].hist(sigma_peaks_inside_ulens, color='r', label='ulens', histtype='step', bins=bins, density=True)
        ax[1].plot(*return_CDF(sigma_peaks_inside_cands), color='k', label='cands')
        ax[1].plot(*return_CDF(sigma_peaks_inside_ulens), color='r', label='ulens')
        ax[1].set_ylabel('CDF')
        sigma_peaks_thresh = 1
        for a in ax:
            a.legend()
            a.set_xlabel(f'{sigma_factor} Sigma Peaks Inside 2tE')
            a.set_xlim(0, 100)
            a.axvline(sigma_peaks_thresh, color='c', alpha=.3)
        cands_frac = np.sum(sigma_peaks_inside_cands < sigma_peaks_thresh) / len(sigma_peaks_inside_cands)
        ulens_frac = np.sum(sigma_peaks_inside_ulens < sigma_peaks_thresh) / len(sigma_peaks_inside_ulens)
        ax[1].axhline(cands_frac, color='g', alpha=.3)
        ax[1].axhline(ulens_frac, color='g', alpha=.3)
        fig.tight_layout()

    sigma_factor = 3
    if sigma_factor == 3:
        sigma_peaks_inside_ulens = num_3sigma_peaks_inside_ulens
        sigma_peaks_inside_cands = num_3sigma_peaks_inside_cands
        sigma_peaks_outside_ulens = num_3sigma_peaks_outside_ulens
        sigma_peaks_outside_cands = num_3sigma_peaks_outside_cands
    elif sigma_factor == 5:
        sigma_peaks_inside_ulens = num_5sigma_peaks_inside_ulens
        sigma_peaks_inside_cands = num_5sigma_peaks_inside_cands
        sigma_peaks_outside_ulens = num_5sigma_peaks_outside_ulens
        sigma_peaks_outside_cands = num_5sigma_peaks_outside_cands

    # ulens passing cut
    cond = sigma_peaks_inside_ulens >= sigma_peaks_thresh
    _plot_sigma_peaks_ulens(data, stats, metadata, cond,
                            sigma_peaks_inside_ulens, sigma_factor, sigma_peaks_thresh)

    # ulens failing cut
    cond = sigma_peaks_inside_ulens < sigma_peaks_thresh
    _plot_sigma_peaks_ulens(data, stats, metadata, cond,
                            sigma_peaks_inside_ulens, sigma_factor, sigma_peaks_thresh)

    # cands passing cut
    cond = sigma_peaks_inside_cands >= sigma_peaks_thresh
    _plot_sigma_peaks_cands(cond, cands3, id_cands, sigma_peaks_inside_cands,
                            sigma_factor, sigma_peaks_thresh)

    # cands failing cut
    cond = sigma_peaks_inside_cands < sigma_peaks_thresh
    _plot_sigma_peaks_cands(cond, cands3, id_cands, sigma_peaks_inside_cands,
                            sigma_factor, sigma_peaks_thresh)




def generate_all_figures():
    plot_ulens_opt_corner()
    plot_ulens_opt_inside_outside()
    plot_ulens_tE_opt_bias()
    plot_ulens_eta_residual_minmax_vs_opt()
    plot_ulens_opt_chi2_ulens_cut()
    plot_ulens_opt_chi2_flat_cut()
    plot_ulens_opt_tE_cut()
    plot_ulens_opt_piE_cut()
    plot_ulens_opt_sigma_peaks()


if __name__ == '__main__':
    generate_all_figures()