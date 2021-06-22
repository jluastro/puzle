#! /usr/bin/env python
"""
plot_example_lightcurves.py
"""

import numpy as np
import matplotlib.pyplot as plt

from puzle.stats import average_xy_on_round_x
from puzle.models import CandidateLevel2
from puzle.cands import fetch_cand_best_obj_by_id
from puzle.utils import return_figures_dir, MJD_start, MJD_finish


def plot_example_lightcurves():
    cands = CandidateLevel2.query.with_entities(CandidateLevel2.id,
                                                CandidateLevel2.eta_best).all()
    cand_id_arr = [c[0] for c in cands]
    eta_arr = np.array([c[1] for c in cands])

    eta_sample_arr = np.linspace(0, 0.85, 8)
    eta_sample_arr[0] = 0.01
    obj_arr = []
    for eta in eta_sample_arr:
        idx = np.argmin(np.abs(eta_arr - eta))
        cand_id = cand_id_arr[idx]
        obj = fetch_cand_best_obj_by_id(cand_id)
        obj_arr.append(obj)

    fig, ax = plt.subplots(8, 1, figsize=(10, 8), sharex=True)
    ax = ax.flatten()
    for a in ax: a.clear()
    for i, obj in enumerate(obj_arr):
        # ax[i].set_title(r'$\eta = $ %.2f' % eta_sample_arr[i], fontsize=14)
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.mag
        ax[i].scatter(hmjd, mag, s=100, color='b', marker='.')
        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        ax[i].scatter(hmjd_round, mag_round, s=5, color='r', marker='^')
        ax[i].invert_yaxis()
        if i == len(obj_arr) - 1:
            ax[i].set_xlabel('heliocentric modified julian date (days)')
        if i == len(obj_arr) // 2:
            ax[i].set_ylabel('magnitude', labelpad=25)
        ax[i].set_xlim(MJD_start, MJD_finish)
        if i == 4:
            y_coord = .5
        else:
            y_coord = 0
        x_coord = 0.714
        ax[i].annotate(r'$\eta = $ %.2f' % eta_sample_arr[i], xy=(x_coord, y_coord),
                       xycoords='axes fraction', fontsize=16,
                       horizontalalignment='right', verticalalignment='bottom')
    fig.subplots_adjust(hspace=0.15, top=.98, bottom=.1, right=.98, left=.1)

    fname = '%s/example_lightcurves.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


if __name__ == '__main__':
    plot_example_lightcurves()
