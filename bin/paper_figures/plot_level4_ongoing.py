#! /usr/bin/env python
"""
plot_level4_ongoing.py
"""

import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt

from puzle.models import CandidateLevel4
from puzle.utils import return_figures_dir, MJD_finish


def plot_ongoing_cands_t0_tE():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.ongoing == True).\
        all()

    t0 = [c.t0_pspl_gp for c in cands]
    t0_err = [c.t0_err_pspl_gp for c in cands]
    tE = [c.tE_pspl_gp for c in cands]
    tE_err = [c.tE_err_pspl_gp for c in cands]

    ztf1_end = Time('2020-09-30').mjd
    ztf2_start = Time('2020-10-01').mjd
    ztf2_end = Time('2023-12-31').mjd

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.clear()
    ax.errorbar(t0, tE, xerr=t0_err, yerr=tE_err, color='b', linestyle='', alpha=.1, ms=.1)
    ax.scatter(t0, tE, color='b', s=2, label='ZTF Candidates Level 4 Ongoing')
    ax.fill_betweenx([0, 1000], ztf2_start, ztf2_end, color='k', alpha=.1, label='ZTF-II Operation')
    ax.axhline(150, color='g', alpha=.6, label='150 days')
    ax.axvline(ztf1_end, color='k', linestyle='-', alpha=.6, label='ZTF-I End')
    ax.axvline(MJD_finish, color='k', linestyle='--', alpha=.6, label='DR5 End')
    ax.set_yscale('log')
    ax.set_xlim(58850, 60000)
    ax.set_ylim(1e1, 7e2)
    ax.set_xlabel(r'$t_0$ (heliocentric modified julian date)')
    ax.set_ylabel(r'$t_E$ (days)')
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[3], handles[0], handles[1], handles[2], handles[4]]
    labels = [labels[3], labels[0], labels[1], labels[2], labels[4]]
    leg = ax.legend(handles, labels,
                    loc=4, markerscale=6, framealpha=1, fontsize=14)
    for i, lh in enumerate(leg.legendHandles):
        if i < 4:
            lh.set_linewidth(3)
            lh.set_alpha(1)
    fig.tight_layout()

    cond = np.array(tE) >= 150
    cond *= np.array(t0) >= ztf2_start
    num_ongoing_cands = np.sum(cond)
    print(f'{num_ongoing_cands} candidates')

    fname = '%s/level4_cands_ongoing_t0_tE.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.02)
    print('-- %s saved' % fname)
    plt.close(fig)


if __name__ == '__main__':
    plot_ongoing_cands_t0_tE()
