#! /usr/bin/env python
"""
plot_ulens_cands_samples.py
"""

import numpy as np
from sqlalchemy.sql.expression import func
from puzle.models import Candidate, Source
from puzle.stats import calculate_eta_on_daily_avg, average_xy_on_round_x
from puzle.utils import load_stacked_array, return_data_dir, return_figures_dir
from puzle.eta import return_eta_arrs, return_eta_ulens_arrs
from puzle import db

import matplotlib.pyplot as plt


def return_cands_sample(eta_low, eta_high, eta_residual_low, eta_residual_high, N_cands=9):
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


def _plot_cands(title, eta_arr, eta_residual_arr, cands):
    fig, ax = plt.subplots(5, 2, figsize=(10, 10))
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
    plt.close(fig)


def plot_cands_samples(eta_arr=None, eta_residual_arr=None):
    if eta_arr is None:
        eta_arr, eta_residual_arr, _ = return_eta_arrs()
    regions_of_interest = [(1, 1.5, 1, 1.5),
                           (1, 1.5, 1.5, 2),
                           (1, 1.5, 2, 2.5),
                           (0.75, 1, 0.5, 1),
                           (0.25, 0.5, 0.5, 1),
                           (0, 0.25, 0, 0.5),
                           (0, 0.25, 0.5, 1),
                           (0, 0.25, 1, 1.5),
                           (0, 0.25, 1.5, 3),
                           (0.25, 0.5, 1, 1.5),
                           (0.25, 0.5, 1.5, 2),
                           (0.5, 0.75, 1.5, 2),
                           (0.75, 1, 1.5, 2)]
    for i, region in enumerate(regions_of_interest):
        cands = return_cands_sample(*region)
        title = f'{i:02d}'
        _plot_cands(title, eta_arr, eta_residual_arr, cands)


def _plot_ulens(title, eta_ulens_arr, eta_residual_ulens_arr, observable_arr,
               eta_sample, eta_residual_sample, data_sample):
    cond = observable_arr == True
    fig, ax = plt.subplots(5, 2, figsize=(10, 10))
    ax = ax.flatten()
    for a in ax: a.clear()
    ax[0].hexbin(eta_ulens_arr[cond], eta_residual_ulens_arr[cond],
                 mincnt=1, gridsize=25)
    ax[0].plot(np.arange(3), color='c', alpha=.5)
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
    plt.close(fig)


def return_ulens_sample(eta_ulens_arr, eta_residual_ulens_arr, observable_arr,
                        eta_low, eta_high, eta_residual_low, eta_residual_high,
                        N_sample=9):
    fname = '%s/ulens_sample.total.npz' % return_data_dir()
    data = load_stacked_array(fname)

    fname = '%s/ulens_sample_metadata.total.npz' % return_data_dir()
    metadata = np.load(fname)

    cond = observable_arr == True
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

    return eta_sample, eta_residual_sample, data_sample, idx_sample


def plot_ulens_samples(eta_ulens_arr, eta_residual_ulens_arr, observable_arr):
    regions_of_interest = [(0.1, 0.3, 1.5, 2),
                           (0.5, 1.0, 1.75, 2.25),
                           (1.0, 1.5, 1.75, 2.25),
                           (1.5, 1.75, 1.75, 2.25),
                           (0.0, 0.25, 0.5, 1.0),
                           (0.23, 0.5, 0.5, 1.0),
                           (0.5, 1.0, 0.5, 1.0),
                           (0, 0.25, 0, 1)]
    for i, region in enumerate(regions_of_interest):
        eta_sample, eta_residual_sample, data_sample, idx_sample = return_ulens_sample(
            eta_ulens_arr, eta_residual_ulens_arr, observable_arr, *region)
        if eta_sample is None:
            print('No samples in region: (%.2f, %.2f, %.2f, %.2f)' % (region[0],
                                                                      region[1],
                                                                      region[2],
                                                                      region[3]))
            continue
        title = f'{i:02d}'
        _plot_ulens(title, eta_ulens_arr, eta_residual_ulens_arr, observable_arr,
                   eta_sample, eta_residual_sample, data_sample)


def generate_all_figures():
    eta_arr, eta_residual_arr, eta_threshold_low_best = return_eta_arrs()
    eta_ulens_arr, eta_residual_ulens_arr, eta_residual_actual_ulens_arr, observable_arr = return_eta_ulens_arrs()
    plot_cands_samples(eta_arr=eta_arr,
                       eta_residual_arr=eta_residual_arr)
    plot_ulens_samples(eta_ulens_arr=eta_ulens_arr,
                       eta_residual_ulens_arr=eta_residual_ulens_arr,
                       observable_arr=observable_arr)


if __name__ == '__main__':
    generate_all_figures()
