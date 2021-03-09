#! /usr/bin/env python
"""
plot_survival_rates.py
"""

import matplotlib.pyplot as plt
import numpy as np
from puzle import db
from puzle.models import StarProcessJob, SourceIngestJob


def parse_cut_rates(job):
    num_objs = job.num_objs
    num_objs_pass_eta = job.num_objs_pass_eta
    num_objs_pass_eta_residual = job.num_objs_pass_eta_residual
    num_objs_pass_rf = job.num_objs_pass_rf

    eta_cut_rate = np.log10(num_objs_pass_eta / num_objs)
    rf_cut_rate = np.log10(num_objs_pass_rf / num_objs_pass_eta)
    eta_residual_rate = np.log10(num_objs_pass_eta_residual / num_objs_pass_rf)

    return eta_cut_rate, rf_cut_rate, eta_residual_rate


def plot_survival_rates():
    jobs = db.session.query(StarProcessJob).filter(StarProcessJob.finished==True).all()
    eta_cur_rate_arr = []
    rf_cut_rate_arr = []
    eta_residual_rate_arr = []
    for job in jobs:
        eta_cut_rate, rf_cut_rate, eta_residual_rate = parse_cut_rates(job)
        eta_cur_rate_arr.append(eta_cut_rate)
        rf_cut_rate_arr.append(rf_cut_rate)
        eta_residual_rate_arr.append(eta_residual_rate)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    for a in ax:
        a.clear()
    ax[0].set_title('Eta Cut', fontsize=12)
    ax[0].set_xlabel('log(survival rate)', fontsize=12)
    ax[0].hist(eta_cur_rate_arr, bins=6, histtype='step', label='puzle')
    ax[0].axvline(np.log10(0.005), color='r', alpha=.4, label='Price-Whelan')
    ax[0].legend(loc=9)
    ax[1].set_title('Star/Galaxy Cut', fontsize=12)
    ax[1].set_xlabel('log(survival rate)', fontsize=12)
    ax[1].hist(rf_cut_rate_arr, bins=6, histtype='step', label='puzle')
    ax[1].axvline(np.log10(0.10), color='r', alpha=.4, label='Price-Whelan')
    ax[1].legend(loc=9)
    ax[2].set_title('Eta Residual Cut', fontsize=12)
    ax[2].set_xlabel('log(survival rate)', fontsize=12)
    ax[2].hist(eta_residual_rate_arr, bins=6, histtype='step', label='puzle')
    ax[2].axvline(np.log10(0.10), color='r', alpha=.4, label='Price-Whelan')
    ax[2].legend(loc=9)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    fig.suptitle('25 Sample Fields', fontsize=10)


def plot_job_locations():
    # grab ra_arr, dec_arr, N_sources_arr from plot_ingest_progress
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.scatter(ra_arr, dec_arr, c=N_sources_arr, edgecolor='None', s=5)
    cond = N_sources_arr == 0
    ax.scatter(ra_arr[cond], dec_arr[cond], c='r', edgecolor='None', s=5)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('log(num sources)', fontsize=12)
    ax.set_xlabel('ra', fontsize=12)
    ax.set_ylabel('dec', fontsize=12)
    fig.tight_layout()

    jobs = db.session.query(SourceIngestJob, StarProcessJob). \
        outerjoin(SourceIngestJob, StarProcessJob.source_ingest_job_id == SourceIngestJob.id). \
        filter(StarProcessJob.finished==True).all()

    # find the density at the center of the job field
    ra_job_arr = []
    dec_job_arr = []
    for source_ingest_job, _ in jobs:
        ra = (source_ingest_job.ra_start + source_ingest_job.ra_end) / 2.
        dec = (source_ingest_job.dec_start + source_ingest_job.dec_end) / 2.
        ra_job_arr.append(ra)
        dec_job_arr.append(dec)
    ax.scatter(ra_job_arr, dec_job_arr, marker='*', s=5, color='k')



if __name__ == '__main__':
    plot_survival_rates()
