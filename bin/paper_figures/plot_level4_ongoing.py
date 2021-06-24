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

    july_2021_mjd = Time('2021-07-15').mjd

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.clear()
    ax.errorbar(t0, tE, xerr=t0_err, yerr=tE_err, color='b', linestyle='', alpha=.1, ms=.1)
    ax.scatter(t0, tE, color='b', s=2)
    ax.axvline(MJD_finish, color='k', alpha=.6, label='DR5 End Date')
    ax.axvline(july_2021_mjd, color='r', alpha=.6, label='2021-07-15')
    ax.axhline(150, color='g', alpha=.6, label='150 days')
    ax.set_yscale('log')
    ax.set_xlim(58850, 59950)
    ax.set_ylim(1e1, 1e3)
    ax.set_xlabel(r'$t_0$ (days)')
    ax.set_ylabel(r'$t_E$ (hmjd)')
    leg = ax.legend(loc=4)
    for lh in leg.legendHandles:
        lh.set_linewidth(3)
        lh.set_alpha(1)
    fig.tight_layout()

    cond = np.array(tE) >= 150
    cond *= np.array(t0) >= july_2021_mjd
    num_ongoing_cands = np.sum(cond)
    print(f'{num_ongoing_cands} candidates')

    fname = '%s/level4_cands_ongoing_t0_tE.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


if __name__ == '__main__':
    plot_ongoing_cands_t0_tE()
