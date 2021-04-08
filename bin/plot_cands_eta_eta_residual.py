#! /usr/bin/env python
"""
plot_cands_eta_eta_residual.py
"""

import numpy as np
from sqlalchemy.sql.expression import func
from popsycle.synthetic import calc_magnification
from puzle.models import Candidate, Source
from puzle.stats import calculate_eta_on_daily_avg_residuals, \
    calculate_eta_on_daily_avg, average_xy_on_round_x
from puzle.utils import load_stacked_array, return_data_dir
from puzle import db

import matplotlib.pyplot as plt


def return_samples(eta_low, eta_high, eta_residual_low, eta_residual_high, N_samples=5):
    cands = db.session.query(Candidate).\
        filter(Candidate.eta_best >= eta_low).\
        filter(Candidate.eta_best <= eta_high).\
        filter(Candidate.eta_residual_best >= eta_residual_low).\
        filter(Candidate.eta_residual_best <= eta_residual_high).\
        order_by(func.random()).limit(N_samples).all()
    samples = []
    for cand in cands:
        source_id = cand.source_id_arr[cand.idx_best]
        color = cand.color_arr[cand.idx_best]
        source = db.session.query(Source).filter(Source.id == source_id).first()
        obj = [o for o in source.zort_source.objects if o.color == color][0]
        samples.append((obj, cand.eta_best, cand.eta_residual_best))
    return samples


def plot_samples(eta_arr, eta_residual_arr, samples):
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax = ax.flatten()
    ax[0].hexbin(eta_arr, eta_residual_arr, mincnt=1)
    ax[0].set_xlabel('eta', fontsize=10)
    ax[0].set_ylabel('eta_residual', fontsize=10)
    for i, (sample, eta, eta_residual) in enumerate(samples, 1):
        hmjd = sample.lightcurve.hmjd
        mag = sample.lightcurve.mag
        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        ax[i].scatter(hmjd, mag, color='b', s=5)
        ax[i].scatter(hmjd_round, mag_round, color='g', s=5)
        ax[0].scatter(eta, eta_residual, color='r', s=1)
    for a in ax[1:]:
        a.set_xlabel('hmjd', fontsize=10)
        a.set_ylabel('mag', fontsize=10)
        a.invert_yaxis()
    fig.tight_layout()


def return_eta_arrs():
    cands = Candidate.query.all()
    eta_arr = [c.eta_best for c in cands]
    eta_residual_arr = [c.eta_residual_best for c in cands]
    return eta_arr, eta_residual_arr


def plot_regions():
    eta_arr, eta_residual_arr = return_eta_arrs()
    regions_of_interst = [(0.95, 1, 2.45, 2.5),
                          (0.75, 0.8, 3, 3.05),
                          (0.95, 1, 1.95, 2),
                          (1.75, 2, 2, 2.5)]
    for region in regions_of_interst:
        samples = return_samples(*region)
        plot_samples(eta_arr, eta_residual_arr, samples)


def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))


def test_for_three_consecutive_decreases(arr):
    return any([strictly_decreasing(arr[i:i+3])
                for i in range(len(arr)-2)])


def return_eta_ulens_arrs():
    fname = '%s/ulens_sample.npz' % return_data_dir()
    data = load_stacked_array(fname)

    fname = '%s/ulens_sample_metadata.npz' % return_data_dir()
    metadata = np.load(fname)

    eta_ulens_arr = []
    eta_residual_ulens_arr = []
    eta_residual_actual_ulens_arr = []
    observable_arr = []
    for i, d in enumerate(data):
        if i % 10 == 0:
            print('Loading candidate %i / %i' % (i, len(data)))
        hmjd = d[:, 0]
        mag = d[:, 1]
        magerr = d[:, 2]

        eta_daily = calculate_eta_on_daily_avg(hmjd, mag)
        eta_residual_daily = calculate_eta_on_daily_avg_residuals(hmjd, mag, magerr)
        if eta_residual_daily is not None:
            eta_ulens_arr.append(eta_daily)
            eta_residual_ulens_arr.append(eta_residual_daily)
            eta_residual_actual_ulens_arr.append(metadata['eta_residual'][i])

            hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
            cond_decreasing = test_for_three_consecutive_decreases(mag_round)

            zp = 21.2477
            A = np.array(calc_magnification(metadata['u0'][i]))
            factor_ZP = 10 ** (zp / 2.5)
            f_S = 10 ** ((metadata['mag_src'][i] - zp) / (-2.5))
            f_tot = f_S / metadata['b_sff'][i]
            lhs = (A - 1) * f_S
            rhs = 3 * np.sqrt(f_tot / factor_ZP)
            cond_bump = lhs > rhs

            if cond_decreasing and cond_bump:
                observable_arr.append(True)
            else:
                observable_arr.append(False)

    eta_ulens_arr = np.array(eta_ulens_arr)
    eta_residual_ulens_arr = np.array(eta_residual_ulens_arr)
    eta_residual_actual_ulens_arr = np.array(eta_residual_actual_ulens_arr)
    observable_arr = np.array(observable_arr)

    return eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr


def plot_eta_eta_residual():
    eta_arr, eta_residual_arr = return_eta_arrs()
    eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr = return_eta_ulens_arrs()

    cond_obs = observable_arr == True

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for a in ax: a.clear()
    ax[0].set_title('cands')
    ax[0].hexbin(eta_arr, eta_residual_arr, mincnt=1)
    ax[0].scatter(eta_ulens_arr, eta_residual_ulens_arr,
                  color='r', s=5, label='ulens')
    ax[0].scatter(eta_ulens_arr, eta_residual_actual_ulens_arr,
                  color='g', s=5, label='ulens actual')
    ax[0].scatter(eta_ulens_arr[cond_obs], eta_residual_actual_ulens_arr[cond_obs],
                  color='k', s=5, label='ulens observable')
    ax[0].legend(markerscale=3)
    ax[1].set_title('ulens total')
    ax[1].hexbin(eta_ulens_arr, eta_residual_ulens_arr,
                 mincnt=1, gridsize=20)
    ax[2].set_title('ulens observable')
    ax[2].hexbin(eta_ulens_arr[cond_obs], eta_residual_ulens_arr[cond_obs],
                 mincnt=1, gridsize=20)
    xmin = min([a.get_xlim()[0] for a in ax])
    xmax = max([a.get_xlim()[1] for a in ax])
    ymin = min([a.get_ylim()[0] for a in ax])
    ymax = max([a.get_ylim()[1] for a in ax])
    for a in ax:
        a.set_xlim((xmin, xmax))
        a.set_ylim((ymin, ymax))
        a.grid(True)
        a.set_xlabel('eta', fontsize=10)
        a.set_ylabel('eta_residual', fontsize=10)
    fig.tight_layout()
