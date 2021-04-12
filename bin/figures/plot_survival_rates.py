#! /usr/bin/env python
"""
plot_survival_rates.py
"""

import matplotlib.pyplot as plt
import numpy as np
from puzle import db
from puzle.models import StarProcessJob, SourceIngestJob


def parse_surv_rates(job):
    num_objs = job.num_objs
    num_objs_pass_eta = job.num_objs_pass_eta
    num_objs_pass_rf = job.num_objs_pass_rf
    num_objs_pass_eta_residual = job.num_objs_pass_eta_residual

    if num_objs_pass_eta == 0:
        eta_surv_rate = 0
        rf_surv_rate = 0
        eta_residual_rate = 0
    else:
        eta_surv_rate = np.log10(num_objs_pass_eta / num_objs)
        if num_objs_pass_rf == 0:
            rf_surv_rate = 0
            eta_residual_rate = 0
        else:
            rf_surv_rate = np.log10(num_objs_pass_rf / num_objs_pass_eta)
            if num_objs_pass_eta_residual == 0:
                eta_residual_rate = 0
            else:
                eta_residual_rate = np.log10(num_objs_pass_eta_residual / num_objs_pass_rf)

    return eta_surv_rate, rf_surv_rate, eta_residual_rate


def plot_survival_rates():
    jobs_low = db.session.query(StarProcessJob).filter(StarProcessJob.finished==True,
                                                       StarProcessJob.num_stars!=0,
                                                       StarProcessJob.priority>10000).all()
    num_jobs_low = len(jobs_low)
    eta_cur_rate_arr_low = []
    rf_surv_rate_arr_low = []
    eta_residual_rate_arr_low = []
    for job in jobs_low:
        eta_surv_rate, rf_surv_rate, eta_residual_rate = parse_surv_rates(job)
        eta_cur_rate_arr_low.append(eta_surv_rate)
        rf_surv_rate_arr_low.append(rf_surv_rate)
        eta_residual_rate_arr_low.append(eta_residual_rate)
        
    jobs_high = db.session.query(StarProcessJob).filter(StarProcessJob.finished==True,
                                                       StarProcessJob.num_stars!=0,
                                                       StarProcessJob.priority<10000).all()
    num_jobs_high = len(jobs_high)
    eta_cur_rate_arr_high = []
    rf_surv_rate_arr_high = []
    eta_residual_rate_arr_high = []
    for job in jobs_high:
        eta_surv_rate, rf_surv_rate, eta_residual_rate = parse_surv_rates(job)
        eta_cur_rate_arr_high.append(eta_surv_rate)
        rf_surv_rate_arr_high.append(rf_surv_rate)
        eta_residual_rate_arr_high.append(eta_residual_rate)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    for a in ax:
        a.clear()
    ax[0].set_title('Eta Cut', fontsize=12)
    ax[0].set_xlabel('log(survival rate)', fontsize=12)
    ax[0].hist(eta_cur_rate_arr_low, bins=6, histtype='step', density=True,
               label='puzle low', color='b')
    ax[0].hist(eta_cur_rate_arr_high, bins=6, histtype='step', density=True,
               label='puzle high', color='g')
    ax[0].axvline(np.log10(0.01), color='r', alpha=.4, label='Price-Whelan')
    ax[1].set_title('Star/Galaxy Cut', fontsize=12)
    ax[1].set_xlabel('log(survival rate)', fontsize=12)
    ax[1].hist(rf_surv_rate_arr_low, bins=6, histtype='step', density=True,
               label='puzle low', color='b')
    ax[1].hist(rf_surv_rate_arr_high, bins=6, histtype='step', density=True,
               label='puzle high', color='g')
    ax[1].axvline(np.log10(0.10), color='r', alpha=.4, label='Price-Whelan')
    ax[2].set_title('Eta Residual Cut', fontsize=12)
    ax[2].set_xlabel('log(survival rate)', fontsize=12)
    ax[2].hist(eta_residual_rate_arr_low, bins=6, histtype='step', density=True,
               label='puzle low', color='b')
    ax[2].hist(eta_residual_rate_arr_high, bins=6, histtype='step', density=True,
               label='puzle high', color='g')
    ax[2].axvline(np.log10(0.10), color='r', alpha=.4, label='Price-Whelan')
    for a in ax:
        a.set_yscale('log')
        a.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=.85)
    fig.suptitle('%i Low Jobs | %i High Jobs' % (num_jobs_low, num_jobs_high))


if __name__ == '__main__':
    plot_survival_rates()
