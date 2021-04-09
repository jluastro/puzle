#! /usr/bin/env python
"""
plot_cands_eta_eta_residual.py
"""

import numpy as np
from sqlalchemy.sql.expression import func
from puzle.models import Candidate, Source
from puzle.stats import calculate_eta_on_daily_avg, average_xy_on_round_x
from puzle.utils import load_stacked_array, return_data_dir, return_figures_dir
from puzle import db

import matplotlib.pyplot as plt


def return_cands_sample(eta_low, eta_high, eta_residual_low, eta_residual_high, N_cands=5):
    cands_tmp = db.session.query(Candidate).\
        filter(Candidate.eta_best >= eta_low).\
        filter(Candidate.eta_best <= eta_high).\
        filter(Candidate.eta_residual_best >= eta_residual_low).\
        filter(Candidate.eta_residual_best <= eta_residual_high).\
        order_by(func.random()).limit(N_cands).all()
    cands = []
    for cand in cands_tmp:
        source_id = cand.source_id_arr[cand.idx_best]
        color = cand.color_arr[cand.idx_best]
        source = db.session.query(Source).filter(Source.id == source_id).first()
        obj = [o for o in source.zort_source.objects if o.color == color][0]
        cands.append((obj, cand.eta_best, cand.eta_residual_best))
    return cands


def return_eta_arrs():
    cands = Candidate.query.all()
    eta_arr = [c.eta_best for c in cands]
    eta_residual_arr = [c.eta_residual_best for c in cands]
    return eta_arr, eta_residual_arr


def return_eta_ulens_arrs():
    data_dir = return_data_dir()
    fname = f'{data_dir}/ulens_sample_etas.total.npz'
    data = np.load(fname)
    eta_ulens_arr = data['eta']
    eta_residual_ulens_arr = data['eta_residual']
    observable_arr = data['observable']

    fname = f'{data_dir}/ulens_sample_metadata.total.npz'
    metadata = np.load(fname)
    eta_residual_actual_ulens_arr = metadata['eta_residual']

    return eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr


def plot_cands(title, eta_arr, eta_residual_arr, cands):
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax = ax.flatten()
    ax[0].hexbin(eta_arr, eta_residual_arr, mincnt=1)
    ax[0].set_xlabel('eta', fontsize=10)
    ax[0].set_ylabel('eta_residual', fontsize=10)
    for i, (obj, eta, eta_residual) in enumerate(cands, 1):
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.mag
        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        ax[i].scatter(hmjd, mag, color='b', s=5)
        ax[i].scatter(hmjd_round, mag_round, color='g', s=5)
        ax[0].scatter(eta, eta_residual, color='r', s=1)
    for a in ax[1:]:
        a.set_xlabel('hmjd', fontsize=10)
        a.set_ylabel('mag', fontsize=10)
        a.invert_yaxis()
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/cands_region_{title}.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)


def plot_cands_samples(eta_arr, eta_residual_arr):
    regions_of_interest = [(0.95, 1, 2.45, 2.5),
                          (0.75, 0.8, 3, 3.05),
                          (0.95, 1, 1.95, 2),
                          (1.75, 2, 2, 2.5),
                           (1, 1.1, 2.35, 2.55)]
    for i, region in enumerate(regions_of_interest):
        cands = return_cands_sample(*region)
        title = str(i)
        plot_cands(title, eta_arr, eta_residual_arr, cands)


def plot_ulens(title, eta_ulens_arr, eta_residual_ulens_arr, observable_arr,
               eta_sample, eta_residual_sample, data_sample):
    cond = observable_arr == True
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax = ax.flatten()
    for a in ax: a.clear()
    ax[0].hexbin(eta_ulens_arr[cond], eta_residual_ulens_arr[cond],
                 mincnt=1, gridsize=25)
    ax[0].set_xlabel('eta', fontsize=10)
    ax[0].set_ylabel('eta_residual', fontsize=10)
    for i, (d, eta, eta_residual) in enumerate(zip(data_sample, eta_sample, eta_residual_sample), 1):
        hmjd = d[:, 0]
        mag = d[:, 1]
        eta_new = calculate_eta_on_daily_avg(hmjd, mag)
        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        ax[i].set_title('eta = %.3f | %.3f' % (eta_sample[i-1], eta_new))
        ax[i].scatter(hmjd, mag, color='b', s=5)
        ax[i].scatter(hmjd_round, mag_round, color='g', s=5)
        ax[0].scatter(eta, eta_residual, color='r', s=1)
    for a in ax[1:]:
        a.set_xlabel('hmjd', fontsize=10)
        a.set_ylabel('mag', fontsize=10)
        a.invert_yaxis()
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_region_{title}.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)


def return_ulens_sample(eta_ulens_arr, eta_residual_ulens_arr, observable_arr,
                        eta_low, eta_high, eta_residual_low, eta_residual_high,
                        N_sample=5):
    fname = '%s/ulens_sample.total.npz' % return_data_dir()
    data = load_stacked_array(fname)

    cond = observable_arr = True
    cond *= eta_ulens_arr >= eta_low
    cond *= eta_ulens_arr <= eta_high
    cond *= eta_residual_ulens_arr >= eta_residual_low
    cond *= eta_residual_ulens_arr <= eta_residual_high

    if np.sum(cond) == 0:
        return None, None, None

    idx_sample = np.random.choice(np.where(cond == True)[0],
                                  size=min(N_sample, np.sum(cond)), replace=False)

    eta_sample = []
    eta_residual_sample = []
    data_sample = []
    for idx in idx_sample:
        eta_sample.append(eta_ulens_arr[idx])
        eta_residual_sample.append(eta_residual_ulens_arr[idx])
        data_sample.append(data[idx])

    return eta_sample, eta_residual_sample, data_sample


def plot_ulens_samples(eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr):
    regions_of_interst = [(0.1, 0.3, 1.5, 2),
                          (1.5, 1.75, 1.75, 2.25),
                          (0, 0.25, 0, 1),
                          (0.5, 1.0, 0.5, 1.0)]
    for i, region in enumerate(regions_of_interst):
        eta_sample, eta_residual_sample, data_sample = return_ulens_sample(
            eta_ulens_arr, eta_residual_ulens_arr, observable_arr, *region)
        if eta_sample is None:
            print('No samples in region: (%.2f, %.2f, %.2f, %.2f)' % (region[0],
                                                                      region[1],
                                                                      region[2],
                                                                      region[3]))
            continue
        title = str(i)
        plot_ulens(title, eta_ulens_arr, eta_residual_ulens_arr, observable_arr,
                   eta_sample, eta_residual_sample, data_sample)


def plot_eta_eta_residual(eta_arr, eta_residual_arr, eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr):

    cond_obs = observable_arr == True

    # linear-linear
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for a in ax: a.clear()
    ax[0].set_title('cands')
    ax[0].hexbin(eta_arr, eta_residual_arr, mincnt=1, gridsize=25)
    ax[1].set_title('ulens total')
    ax[1].hexbin(eta_ulens_arr, eta_residual_ulens_arr,
                 mincnt=1, gridsize=25)
    ax[2].set_title('ulens observable')
    ax[2].hexbin(eta_ulens_arr[cond_obs], eta_residual_ulens_arr[cond_obs],
                 mincnt=1, gridsize=25)
    xmin = min([a.get_xlim()[0] for a in ax])
    xmax = max([a.get_xlim()[1] for a in ax])
    ymin = min([a.get_ylim()[0] for a in ax])
    ymax = max([a.get_ylim()[1] for a in ax])
    for a in ax:
        a.set_xlim((xmin, xmax))
        a.set_ylim((ymin, ymax))
        a.grid(True)
        a.set_xlabel('log(eta)', fontsize=10)
        a.set_ylabel('log(eta_residual)', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_cands_eta_eta_residual.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)

    # log-linear
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for a in ax: a.clear()
    ax[0].set_title('cands')
    ax[0].hexbin(np.log10(eta_arr), eta_residual_arr, mincnt=1, gridsize=25)
    ax[1].set_title('ulens total')
    ax[1].hexbin(np.log10(eta_ulens_arr), eta_residual_ulens_arr,
                 mincnt=1, gridsize=25)
    ax[2].set_title('ulens observable')
    ax[2].hexbin(np.log10(eta_ulens_arr[cond_obs]), eta_residual_ulens_arr[cond_obs],
                 mincnt=1, gridsize=25)
    xmin = min([a.get_xlim()[0] for a in ax])
    xmax = max([a.get_xlim()[1] for a in ax])
    ymin = min([a.get_ylim()[0] for a in ax])
    ymax = max([a.get_ylim()[1] for a in ax])
    for a in ax:
        a.set_xlim((xmin, xmax))
        a.set_ylim((ymin, ymax))
        a.grid(True)
        a.set_xlabel('log(eta)', fontsize=10)
        a.set_ylabel('eta_residual', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_cands_log-eta_eta_residual.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)

    # log-log
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for a in ax: a.clear()
    ax[0].set_title('cands')
    ax[0].hexbin(np.log10(eta_arr), np.log10(eta_residual_arr), mincnt=1, gridsize=25)
    ax[1].set_title('ulens total')
    ax[1].hexbin(np.log10(eta_ulens_arr), np.log10(eta_residual_ulens_arr),
                 mincnt=1, gridsize=25)
    ax[2].set_title('ulens observable')
    ax[2].hexbin(np.log10(eta_ulens_arr[cond_obs]), np.log10(eta_residual_ulens_arr[cond_obs]),
                 mincnt=1, gridsize=25)
    xmin = min([a.get_xlim()[0] for a in ax])
    xmax = max([a.get_xlim()[1] for a in ax])
    ymin = min([a.get_ylim()[0] for a in ax])
    ymax = max([a.get_ylim()[1] for a in ax])
    for a in ax:
        a.set_xlim((xmin, xmax))
        a.set_ylim((ymin, ymax))
        a.grid(True)
        a.set_xlabel('log(eta)', fontsize=10)
        a.set_ylabel('log(eta_residual)', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_cands_log-eta_log-eta_residual.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)


def plot_eta_residual_ulens_vs_actual(eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr):
    x_min = np.min([np.min(eta_residual_ulens_arr), np.min(eta_residual_actual_ulens_arr)])
    x_max = np.max([np.min(eta_residual_ulens_arr), np.max(eta_residual_actual_ulens_arr)])
    x = np.linspace(x_min, x_max)

    cond_obs = observable_arr == True

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    for a in ax: a.clear()
    ax[0].set_title('ulens total : %i samples' % len(cond_obs))
    ax[0].hexbin(eta_residual_ulens_arr,
                 eta_residual_actual_ulens_arr,
                 gridsize=25, mincnt=1)
    ax[1].set_title('ulens observable : %i samples' % np.sum(cond_obs))
    ax[1].hexbin(eta_residual_ulens_arr[cond_obs],
                 eta_residual_actual_ulens_arr[cond_obs],
                 gridsize=25, mincnt=1)
    for a in ax:
        a.set_xlabel('measured eta_residual ulens', fontsize=10)
        a.set_ylabel('modeled eta_residual ulens', fontsize=10)
        a.plot(x, x, color='r', linewidth=1)
    fig.tight_layout()


    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_eta_residual_vs_actual.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)


def plot_lowest_ulens_eta(eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr):
    cond = observable_arr == True
    cond_idx = np.where(cond == True)[0]
    eta_ulens_obs_arr = eta_ulens_arr[cond]

    idx_arr = np.argsort(eta_ulens_obs_arr)[:8]

    fname = '%s/ulens_sample.total.npz' % return_data_dir()
    data = load_stacked_array(fname)

    fig, ax = plt.subplots(4, 2, figsize=(10, 10))
    ax = ax.flatten()
    for a in ax: a.clear()
    for i, idx in enumerate(idx_arr):
        hmjd = data[cond_idx[idx]][:, 0]
        mag = data[cond_idx[idx]][:, 1]
        eta_new = calculate_eta_on_daily_avg(hmjd, mag)
        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        ax[i].set_title('eta = %.3f | %.3f' % (eta_new, eta_ulens_obs_arr[idx]))
        ax[i].scatter(hmjd, mag, color='b', s=5)
        ax[i].scatter(hmjd_round, mag_round, color='g', s=5)
    for a in ax:
        a.set_xlabel('hmjd', fontsize=10)
        a.set_ylabel('mag', fontsize=10)
        a.invert_yaxis()
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_lowest_eta.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)


def plot_ulens_tE_piE(observable_arr):
    fname = '%s/ulens_sample_metadata.total.npz' % return_data_dir()
    metadata = np.load(fname)

    cond = observable_arr == True

    tE = metadata['tE']
    piE = np.hypot(metadata['piE_E'], metadata['piE_N'])

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()

    tE_bins = np.logspace(np.log10(2e1), 3, 15)
    ax[0].hist(tE, bins=tE_bins, histtype='step', label='ulens total')
    ax[0].hist(tE[cond], bins=tE_bins, histtype='step', label='ulens obs')
    ax[0].set_xlabel('t_E', fontsize=12)
    ax[0].set_yscale('log')
    ax[0].legend()

    piE_bins = np.logspace(-2, np.log10(3), 15)
    ax[1].hist(piE, bins=piE_bins, histtype='step', label='ulens total')
    ax[1].hist(piE[cond], bins=piE_bins, histtype='step', label='ulens obs')
    ax[1].set_xlabel('pi_E', fontsize=12)
    ax[1].set_yscale('log')
    ax[1].legend()

    tE_counts, _ = np.histogram(tE, bins=tE_bins)
    tE_obs_counts, _ = np.histogram(tE[cond], bins=tE_bins)
    tE_obs_frac = tE_obs_counts / tE_counts
    ax[2].plot(tE_bins[:-1], tE_obs_frac, marker='.')
    ax[2].set_xlabel('t_E', fontsize=12)
    ax[2].set_ylabel('N_obs / N_total')

    piE_counts, _ = np.histogram(piE, bins=piE_bins)
    piE_obs_counts, _ = np.histogram(piE[cond], bins=piE_bins)
    piE_obs_frac = piE_obs_counts / piE_counts
    ax[3].plot(piE_bins[:-1], piE_obs_frac, marker='.')
    ax[3].set_xlabel('pi_E', fontsize=12)
    ax[3].set_ylabel('N_obs / N_total')

    for a in ax:
        a.set_xscale('log')
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_tE_piE.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)


def plot_ulens_tE_piE_vs_eta(eta_ulens_arr, eta_residual_ulens_arr, observable_arr):
    fname = '%s/ulens_sample_metadata.total.npz' % return_data_dir()
    metadata = np.load(fname)

    cond = observable_arr == True

    tE = metadata['tE']
    piE = np.hypot(metadata['piE_E'], metadata['piE_N'])

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()

    def return_fit(x, y):
        idx_arr = np.argsort(x)
        x_sort, y_sort = x[idx_arr], y[idx_arr]
        x_log, y_log = np.log10(x_sort), np.log10(y_sort)
        m, b = np.polyfit(x_log, y_log, deg=1)
        return 10 ** (x_log), 10 ** (x_log * m + b)

    ax[0].scatter(tE, eta_ulens_arr,
                  s=5, alpha=.3, label='ulens total')
    ax[0].scatter(tE[cond], eta_ulens_arr[cond],
                  s=5, alpha=.3, label='ulens obs')
    ax[0].plot(*return_fit(tE[cond], eta_ulens_arr[cond]), color='r', alpha=.4)
    ax[0].set_xlabel('tE')
    ax[0].set_ylabel('eta')

    ax[1].scatter(tE, eta_residual_ulens_arr,
                  s=5, alpha=.3, label='ulens total')
    ax[1].scatter(tE[cond], eta_residual_ulens_arr[cond],
                  s=5, alpha=.3, label='ulens obs')
    ax[1].plot(*return_fit(tE[cond], eta_residual_ulens_arr[cond]), color='r', alpha=.4)
    ax[1].set_xlabel('tE')
    ax[1].set_ylabel('eta_residual')

    ax[2].scatter(piE, eta_ulens_arr,
                  s=5, alpha=.3, label='ulens total')
    ax[2].scatter(piE[cond], eta_ulens_arr[cond],
                  s=5, alpha=.3, label='ulens obs')
    ax[2].plot(*return_fit(piE[cond], eta_ulens_arr[cond]), color='r', alpha=.4)
    ax[2].set_xlabel('piE')
    ax[2].set_ylabel('eta')

    ax[3].scatter(piE, eta_residual_ulens_arr,
                  s=5, alpha=.3, label='ulens total')
    ax[3].scatter(piE[cond], eta_residual_ulens_arr[cond],
                  s=5, alpha=.3, label='ulens obs')
    ax[3].plot(*return_fit(piE[cond], eta_residual_ulens_arr[cond]), color='r', alpha=.4)
    ax[3].set_xlabel('piE')
    ax[3].set_ylabel('eta_residual')

    for a in ax:
        a.legend(markerscale=3)
        a.set_xscale('log')
    fig.tight_layout()

    fname = '%s/ulens_tE_piE_vs_eta.png' % return_figures_dir()
    fig.savefig(fname)
    plt.close(fig)


def plot_ulens_eta_by_mag(eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr):
    fname = '%s/ulens_sample_metadata.total.npz' % return_data_dir()
    metadata = np.load(fname)
    mag_src = metadata['mag_src']

    cond_obs = observable_arr == True

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax = ax.flatten()
    for a in ax: a.clear()
    ax[0].set_title('All Observable mag_src')
    ax[0].hexbin(eta_ulens_arr[cond_obs], eta_residual_ulens_arr[cond_obs],
                 mincnt=1, gridsize=25)
    mag_src_bins = [0, 17.5, 18.5, 19.5, 20.5, 21.5]
    for i in range(5):
        mag_src_low = mag_src_bins[i]
        mag_src_high = mag_src_bins[i+1]
        cond_mag_src = (mag_src > mag_src_low) * (mag_src <= mag_src_high)
        ax[i+1].set_title('%.1f < mag_src <= %.1f' % (mag_src_low, mag_src_high))
        ax[i+1].hexbin(eta_ulens_arr[cond_obs*cond_mag_src],
                       eta_residual_ulens_arr[cond_obs*cond_mag_src],
                       mincnt=1, gridsize=25)

        ax[i+1].grid(True)
        ax[i+1].set_xlabel('eta', fontsize=10)
        ax[i+1].set_ylabel('eta_residual', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_eta_by_mag.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)


def plot_ulens_eta_by_tE(eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr):
    fname = '%s/ulens_sample_metadata.total.npz' % return_data_dir()
    metadata = np.load(fname)
    tE = metadata['tE']

    cond_obs = observable_arr == True

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax = ax.flatten()
    for a in ax: a.clear()
    ax[0].set_title('All Observable tE')
    ax[0].hexbin(eta_ulens_arr[cond_obs], eta_residual_ulens_arr[cond_obs],
                 mincnt=1, gridsize=25)
    tE_bins = [20, 50, 80, 120, 150, 1000]
    for i in range(5):
        tE_low = tE_bins[i]
        tE_high = tE_bins[i+1]
        cond_tE = (tE > tE_low) * (tE <= tE_high)
        ax[i+1].set_title('%.1f < tE <= %.1f' % (tE_low, tE_high))
        ax[i+1].hexbin(eta_ulens_arr[cond_obs*cond_tE],
                       eta_residual_ulens_arr[cond_obs*cond_tE],
                       mincnt=1, gridsize=25)

        ax[i+1].grid(True)
        ax[i+1].set_xlabel('eta', fontsize=10)
        ax[i+1].set_ylabel('eta_residual', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_eta_by_tE.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)


def plot_ulens_eta_by_piE(eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr):
    fname = '%s/ulens_sample_metadata.total.npz' % return_data_dir()
    metadata = np.load(fname)
    piE = np.hypot(metadata['piE_E'], metadata['piE_N'])

    cond_obs = observable_arr == True

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax = ax.flatten()
    for a in ax: a.clear()
    ax[0].set_title('All Observable piE')
    ax[0].hexbin(eta_ulens_arr[cond_obs], eta_residual_ulens_arr[cond_obs],
                 mincnt=1, gridsize=25)
    piE_bins = np.logspace(-2, np.log10(3), 6)
    for i in range(5):
        piE_low = piE_bins[i]
        piE_high = piE_bins[i+1]
        cond_piE = (piE > piE_low) * (piE <= piE_high)
        ax[i+1].set_title('%.2f < piE <= %.2f' % (piE_low, piE_high))
        ax[i+1].hexbin(eta_ulens_arr[cond_obs*cond_piE],
                       eta_residual_ulens_arr[cond_obs*cond_piE],
                       mincnt=1, gridsize=25)

        ax[i+1].grid(True)
        ax[i+1].set_xlabel('eta', fontsize=10)
        ax[i+1].set_ylabel('eta_residual', fontsize=10)
    fig.tight_layout()

    figures_dir = return_figures_dir()
    fname = f'{figures_dir}/ulens_eta_by_piE.png'
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    

def generate_all_plots():
    eta_arr, eta_residual_arr = return_eta_arrs()
    eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr = return_eta_ulens_arrs()
    plot_cands_samples(eta_arr, eta_residual_arr)
    plot_ulens_samples(eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr)
    plot_eta_eta_residual(eta_arr, eta_residual_arr, eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr)
    plot_eta_residual_ulens_vs_actual(eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr)
    plot_lowest_ulens_eta(eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr)
    plot_ulens_tE_piE(observable_arr)
    plot_ulens_tE_piE_vs_eta(eta_ulens_arr, eta_residual_ulens_arr, observable_arr)
    plot_ulens_eta_by_mag(eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr)
    plot_ulens_eta_by_tE(eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr)
    plot_ulens_eta_by_piE(eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr)


if __name__ == '__main__':
    generate_all_plots()
