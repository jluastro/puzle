#! /usr/bin/env python
"""
plot_pspl_gp_fit_cuts.py
"""

import matplotlib.pyplot as plt
import numpy as np

from puzle.cands import apply_level4_cuts_to_query
from puzle.models import CandidateLevel4
from puzle.utils import return_figures_dir


def plot_pspl_gp_fit_cuts():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.pspl_gp_fit_finished == True,
               CandidateLevel4.fit_type_pspl_gp != None).\
        with_entities(CandidateLevel4.piE_pspl_gp).\
        all()

    piE_cands = np.array([c[0] for c in cands])

    query = apply_level4_cuts_to_query(CandidateLevel4.query)
    cands_cut = query.\
        filter(CandidateLevel4.pspl_gp_fit_finished == True,
               CandidateLevel4.fit_type_pspl_gp != None).\
        with_entities(CandidateLevel4.piE_pspl_gp).\
        all()

    piE_cands_cut = np.array([c[0] for c in cands_cut])

    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    ax = ax.flatten()
    for a in ax: a.clear()

    fig.tight_layout()

    fname = '%s/plot_pspl_gp_fit_cuts.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


if __name__ == '__main__':
    plot_pspl_gp_fit_cuts()
