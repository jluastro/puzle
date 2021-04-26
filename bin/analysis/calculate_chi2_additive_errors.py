#! /usr/bin/env python
"""
calculate_chi2_additive_errors.py
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from astropy.stats import sigma_clip
from zort.photometry import fluxes_to_magnitudes, magnitudes_to_fluxes
from microlens.jlu.model import PSPL_Phot_Par_Param1

from puzle.ulens import return_ulens_data, return_ulens_metadata
from puzle.cands import calculate_chi2
from puzle.utils import return_figures_dir


def plot_photometry(data, model):
    fig = plt.figure(1, figsize=(10, 10))
    plt.clf()
    plt.subplots_adjust(bottom=0.2, left=0.3)

    f1 = plt.gcf().add_axes([0.2, 0.45, 0.7, 0.45])
    f2 = plt.gcf().add_axes([0.2, 0.15, 0.7, 0.25])

    # Get the data out.
    filt_index = 0
    dat_t = data['t_phot' + str(filt_index + 1)]
    dat_m = data['mag' + str(filt_index + 1)]
    dat_me = data['mag_err' + str(filt_index + 1)]

    # Make models.
    # Decide if we sample the models at a denser time, or just the
    # same times as the measurements.
    mod_t = np.arange(dat_t.min(), dat_t.max(), 0.1)
    mod_m_out = model.get_photometry(mod_t, filt_index)
    mod_m_at_dat = model.get_photometry(dat_t, filt_index)

    #####
    # Data
    #####
    f1.errorbar(dat_t, dat_m, yerr=dat_me, fmt='k.', alpha=0.2, label='Data')
    f1.plot(mod_t, mod_m_out, 'r-', label='Model')
    f1.set_ylabel('I (mag)')
    f1.invert_yaxis()
    f1.set_title('Input Data and Output Model')
    f1.get_xaxis().set_visible(False)
    f1.set_xlabel('t - t0 (days)')
    f1.legend()

    #####
    # Residuals
    #####
    f1.get_shared_x_axes().join(f1, f2)
    f2.errorbar(dat_t, dat_m - mod_m_at_dat,
                yerr=dat_me, fmt='k.', alpha=0.2)
    f2.axhline(0, linestyle='--', color='r')
    f2.set_xlabel('Time (HJD)')
    f2.set_ylabel('Obs - Mod')

    return fig


def generate_samples(N_samples=20, brightFlag=False, midFlag=False, faintFlag=False):
    data_ulens = return_ulens_data(observableFlag=True, bhFlag=False)
    metadata_ulens = return_ulens_metadata(observableFlag=True, bhFlag=False)

    if brightFlag:
        label = 'bright'
        cond = metadata_ulens['mag_src'] < 15
        idx_arr = np.random.choice(np.where(cond == True)[0],
                                   replace=False, size=N_samples)
    elif midFlag:
        label = 'mid'
        cond1 = metadata_ulens['mag_src'] > 18
        cond2 = metadata_ulens['mag_src'] < 19
        cond = cond1 * cond2
        idx_arr = np.random.choice(np.where(cond == True)[0],
                                   replace=False, size=N_samples)
    elif faintFlag:
        label = 'faint'
        cond = metadata_ulens['mag_src'] > 20
        idx_arr = np.random.choice(np.where(cond == True)[0],
                                   replace=False, size=N_samples)
    else:
        label = 'all'
        idx_arr = np.random.choice(np.arange(len(data_ulens)),
                                   replace=False, size=N_samples)

    for idx in idx_arr:
        hmjd, mag, magerr = data_ulens[idx][:, :3].T
        t0 = metadata_ulens['t0'][idx]
        u0 = metadata_ulens['u0'][idx]
        tE = metadata_ulens['tE'][idx]
        mag_src = metadata_ulens['mag_src'][idx]
        piE_E = metadata_ulens['piE_E'][idx]
        piE_N = metadata_ulens['piE_N'][idx]
        b_sff = metadata_ulens['b_sff'][idx]
        ra = metadata_ulens['ra'][idx]
        dec = metadata_ulens['dec'][idx]
        model = PSPL_Phot_Par_Param1(t0=t0, u0_amp=u0, tE=tE, mag_src=mag_src,
                                     piE_E=piE_E, piE_N=piE_N, b_sff=b_sff,
                                     raL=ra, decL=dec)
        mag_model = model.get_photometry(hmjd)
        data = {'t_phot1': hmjd,
                'mag1': mag,
                'mag_err1': magerr}
        fig = plot_photometry(data, model)

        chi2 = calculate_chi2(mag, magerr, mag_model, add_err=0)
        reduced_chi2 = chi2 / len(mag)

        amp = model.get_amplification(hmjd)
        flux_total, _ = magnitudes_to_fluxes(mag)
        flux_obj = flux_total / (1 + b_sff * (amp - 1))
        mag_obj, _ = fluxes_to_magnitudes(flux_obj)
        mag_baseline = np.median(mag_obj)
        fig.suptitle('type %s | idx %i | mag_baseline %.2f | chi2 %.3f'
                     % (label, idx, mag_baseline, reduced_chi2))

        fname = '%s/ulens_sample.%s.%i.png' % (return_figures_dir(), label, idx)
        fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
        print('-- %s saved' % fname)
        plt.close(fig)


def return_CDF(arr):
    x = np.sort(arr)
    y = np.arange(len(arr)) / (len(arr) - 1)
    return x, y


def calculate_chi2(mag, magerr, mag_model, add_err=0,
                   hmjd=None, t0=None, tE=None, clip=False):
    if clip:
        # Mask out those points that are within the microlensing event
        ulens_mask = hmjd > t0 - 2 * tE
        ulens_mask *= hmjd < t0 + 2 * tE
        # Use this mask to generated a masked array for sigma clipping
        # By applying this mask, the 3-sigma will not be calculated using these points
        mag_masked = np.ma.array(mag, mask=ulens_mask)
        # Perform the sigma clipping
        mag_masked = sigma_clip(mag_masked, sigma=3, maxiters=5)
        # This masked array is now a mask that includes BOTH the mirolensing event and
        # the 3-sigma outliers that we want removed. We want to remove the mask that
        # is on the micorlensing event. So we only keep the mask for those points
        # outside of +-2 tE.
        chi2_mask = mag_masked.mask * ~ulens_mask
        # This mask is then inverted for the chi2 calculation
        chi2_cond = ~chi2_mask
    else:
        chi2_cond = np.ones(len(mag)).astype(bool)
    chi2 = np.sum(((mag[chi2_cond] - mag_model[chi2_cond]) / np.hypot(magerr[chi2_cond], add_err)) ** 2)
    dof = np.sum(chi2_cond)
    return chi2, dof


def return_ideal_reduced_chi2(size=50):
    data_ulens = return_ulens_data(observableFlag=True, bhFlag=False)
    ideal_chi2_arr = np.array([])
    for d in data_ulens:
        dof = len(d)
        ideal_chi2 = stats.chi2.rvs(dof, size=size) / dof
        ideal_chi2_arr = np.append(ideal_chi2_arr, ideal_chi2)
    return ideal_chi2_arr


def return_reduced_chi2(add_err=0, dof_limit=None,
                        low_mag=False, high_mag=False,
                        calc_mag_baseline=False,
                        chi2_on_clip=False):
    data_ulens = return_ulens_data(observableFlag=True, bhFlag=False)
    metadata_ulens = return_ulens_metadata(observableFlag=True, bhFlag=False)

    reduced_chi2_arr = []
    reduced_chi2_clipped_arr = []
    mag_baseline_arr = []
    idx_arr = np.arange(len(data_ulens))
    for idx in idx_arr:
        if idx % 10000 == 0:
            print('-- ulens sample %i / %i' % (idx, len(idx_arr)))
        hmjd, mag, magerr = data_ulens[idx][:, :3].T
        if low_mag and np.max(mag) > 15:
            continue
        if high_mag and np.max(mag) < 20:
            continue
        if dof_limit and len(mag) != dof_limit:
            continue
        t0 = metadata_ulens['t0'][idx]
        u0 = metadata_ulens['u0'][idx]
        tE = metadata_ulens['tE'][idx]
        mag_src = metadata_ulens['mag_src'][idx]
        piE_E = metadata_ulens['piE_E'][idx]
        piE_N = metadata_ulens['piE_N'][idx]
        b_sff = metadata_ulens['b_sff'][idx]
        ra = metadata_ulens['ra'][idx]
        dec = metadata_ulens['dec'][idx]
        model = PSPL_Phot_Par_Param1(t0=t0, u0_amp=u0, tE=tE, mag_src=mag_src,
                                     piE_E=piE_E, piE_N=piE_N, b_sff=b_sff,
                                     raL=ra, decL=dec)
        mag_model = model.get_photometry(hmjd)
        # if chi2_on_clip:
        #     chi2, dof = calculate_chi2(mag, magerr, mag_model, add_err=add_err,
        #                                hmjd=hmjd, t0=t0, tE=tE, clip=True)
        # else:
        #     chi2, dof = calculate_chi2(mag, magerr, mag_model, add_err=add_err)
        # reduced_chi2 = chi2 / dof
        # reduced_chi2_arr.append(reduced_chi2)

        chi2, dof = calculate_chi2(mag, magerr, mag_model, add_err=add_err,
                                   hmjd=hmjd, t0=t0, tE=tE, clip=True)
        reduced_chi2 = chi2 / dof
        reduced_chi2_clipped_arr.append(reduced_chi2)
        chi2, dof = calculate_chi2(mag, magerr, mag_model, add_err=add_err)
        reduced_chi2 = chi2 / dof
        reduced_chi2_arr.append(reduced_chi2)

        if calc_mag_baseline:
            amp = model.get_amplification(hmjd)
            flux_total, _ = magnitudes_to_fluxes(mag)
            flux_obj = flux_total / (1 + b_sff * (amp - 1))
            mag_obj, _ = fluxes_to_magnitudes(flux_obj)
            mag_baseline = np.median(mag_obj)
            mag_baseline_arr.append(mag_baseline)

    reduced_chi2_arr = np.array(reduced_chi2_arr)
    mag_baseline_arr = np.array(mag_baseline_arr)

    return reduced_chi2_arr


def calculate_chi2_additive_errors():
    chi2_on_clip = True
    dof_limit = None
    low_mag = False
    high_mag = False
    if dof_limit:
        ideal_reduced_chi2_arr = stats.chi2.rvs(dof_limit, size=100000) / dof_limit
    else:
        ideal_reduced_chi2_arr = return_ideal_reduced_chi2()

    # add_err_arr = np.linspace(0.0, 0.025, 5)
    add_err_arr = np.linspace(0.0, 0.05, 5)
    reduced_chi2_sample_arr = []
    for i, add_err in enumerate(add_err_arr):
        print('Evaluating add_err = %.4f (%i / %i)' % (add_err, i+1, len(add_err_arr)))
        reduced_chi2_sample = return_reduced_chi2(add_err,
                                                  dof_limit=dof_limit,
                                                  low_mag=low_mag,
                                                  high_mag=high_mag,
                                                  chi2_on_clip=chi2_on_clip)
        reduced_chi2_sample_arr.append(reduced_chi2_sample)

    ks_arr = []
    for reduced_chi2_sample in reduced_chi2_sample_arr:
        ks = stats.ks_2samp(ideal_reduced_chi2_arr, reduced_chi2_sample).statistic
        ks_arr.append(ks)

    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    fig.suptitle('dof_limit %s | low_mag %s | high_mag %s | %i samples' %
                 (dof_limit, low_mag, high_mag, len(reduced_chi2_sample_arr[0])))
    for a in ax: a.clear()
    bins = np.linspace(0, 3, 20)
    ax[0].hist(ideal_reduced_chi2_arr, bins=bins, density=True,
               histtype='step', label='ideal')
    ax[1].plot(*return_CDF(ideal_reduced_chi2_arr), label='ideal')
    for i, reduced_chi2_sample in enumerate(reduced_chi2_sample_arr):
        ks = ks_arr[i]
        add_err = add_err_arr[i]
        label = 'ulens add_err=%.3f ks=%.2f' % (add_err, ks)
        ax[0].hist(reduced_chi2_sample, bins=bins, density=True,
                   histtype='step', label=label)
        ax[1].plot(*return_CDF(reduced_chi2_sample), label=label)
    # ax[0].set_yscale('log')
    ax[1].legend(fontsize=10)
    ax[0].set_ylabel('N')
    ax[1].set_ylabel('CDF')
    for a in ax[:2]:
        a.set_xlim(0, 3)
        a.set_xlabel('Reduced Chi2')
    ax[2].scatter(add_err_arr, ks_arr)
    ax[2].set_xlabel('Additive Error', fontsize=12)
    ax[2].set_ylabel('KS Test Statistic', fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)


if __name__ == '__main__':
    calculate_chi2_additive_errors()
