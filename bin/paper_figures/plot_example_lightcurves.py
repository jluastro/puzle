#! /usr/bin/env python
"""
plot_example_lightcurves.py
"""

import numpy as np
import matplotlib.pyplot as plt

from puzle.stats import average_xy_on_round_x
from puzle.models import CandidateLevel2
from puzle.cands import fetch_cand_best_obj_by_id
from puzle.utils import return_figures_dir


def plot_example_lightcurves():
    cands = CandidateLevel2.query.with_entities(CandidateLevel2.id,
                                                CandidateLevel2.eta_best).all()
    cand_id_arr = [c[0] for c in cands]
    eta_arr = np.array([c[1] for c in cands])

    eta_sample_arr = np.arange(0, 1.41, 0.2)
    eta_sample_arr[0] = 0.01
    obj_arr = []
    for eta in eta_sample_arr:
        idx = np.argmin(np.abs(eta_arr - eta))
        cand_id = cand_id_arr[idx]
        obj = fetch_cand_best_obj_by_id(cand_id)
        obj_arr.append(obj)

    fig, ax = plt.subplots(2, 4, figsize=(14, 6))
    ax = ax.flatten()
    for a in ax: a.clear()
    for i, obj in enumerate(obj_arr):
        ax[i].set_title(r'$\eta = $ %.2f' % eta_sample_arr[i])
        hmjd = obj.lightcurve.hmjd
        mag = obj.lightcurve.mag
        ax[i].scatter(hmjd, mag, s=100, color='b', marker='.')
        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        ax[i].scatter(hmjd_round, mag_round, s=5, color='r', marker='^')
        ax[i].invert_yaxis()
        ax[i].set_xlabel('hmjd')
        ax[i].set_ylabel('mag')
    fig.tight_layout()
    fig.suptitle('Example Lightcurves')
    fig.subplots_adjust(top=.9)

    fname = '%s/example_lightcurves.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


if __name__ == '__main__':
    plot_example_lightcurves()
