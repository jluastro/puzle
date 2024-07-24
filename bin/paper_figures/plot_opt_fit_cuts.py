#! /usr/bin/env python
"""
plot_opt_fit_cuts.py
"""

import matplotlib.pyplot as plt
import numpy as np

from puzle.cands import apply_level3_cuts_to_query
from puzle.ulens import return_ulens_data, return_ulens_stats
from puzle.models import CandidateLevel3
from puzle.utils import return_figures_dir

plt.rc('font', size=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)

def plot_opt_fit_cuts():
    cands = CandidateLevel3.query.with_entities(CandidateLevel3.chi_squared_ulens_best,
                                                CandidateLevel3.num_days_best,
                                                CandidateLevel3.chi_squared_flat_outside_2tE_best,
                                                CandidateLevel3.num_days_outside_2tE_best,
                                                CandidateLevel3.tE_best,
                                                CandidateLevel3.piE_N_best,
                                                CandidateLevel3.piE_E_best).\
        filter(CandidateLevel3.tE_best > 0).all()

    chi2_model_cands = np.array([c[0] for c in cands])
    num_days_cands = np.array([c[1] for c in cands])
    chi2_flat_cands = np.array([c[2] for c in cands])
    num_days_oustside_cands = np.array([c[3] for c in cands])
    tE_cands = np.array([c[4] for c in cands])
    piE_N_cands = np.array([c[5] for c in cands])
    piE_E_cands = np.array([c[6] for c in cands])
    piE_cands = np.hypot(piE_N_cands, piE_E_cands)
    reduced_chi2_model_cands = chi2_model_cands / num_days_cands
    reduced_chi2_flat_cands = chi2_flat_cands / num_days_oustside_cands

    query = apply_level3_cuts_to_query(CandidateLevel3.query)
    cands_cut = query.with_entities(CandidateLevel3.chi_squared_ulens_best,
                                    CandidateLevel3.num_days_best,
                                    CandidateLevel3.chi_squared_flat_outside_2tE_best,
                                    CandidateLevel3.num_days_outside_2tE_best,
                                    CandidateLevel3.tE_best,
                                    CandidateLevel3.piE_N_best,
                                    CandidateLevel3.piE_E_best).\
        filter(CandidateLevel3.tE_best > 0).all()

    chi2_model_cands_cut = np.array([c[0] for c in cands_cut])
    num_days_cands_cut = np.array([c[1] for c in cands_cut])
    chi2_flat_cands_cut = np.array([c[2] for c in cands_cut])
    num_days_oustside_cands_cut = np.array([c[3] for c in cands_cut])
    tE_cands_cut = np.array([c[4] for c in cands_cut])
    piE_N_cands_cut = np.array([c[5] for c in cands_cut])
    piE_E_cands_cut = np.array([c[6] for c in cands_cut])
    piE_cands_cut = np.hypot(piE_N_cands_cut, piE_E_cands_cut)
    reduced_chi2_model_cands_cut = chi2_model_cands_cut / num_days_cands_cut
    reduced_chi2_flat_cands_cut = chi2_flat_cands_cut / num_days_oustside_cands_cut

    bhFlag = False
    data = return_ulens_data(observableFlag=True, bhFlag=bhFlag)
    stats = return_ulens_stats(observableFlag=True, bhFlag=bhFlag)

    chi2_model_ulens = stats['chi_squared_ulens_level3']
    num_days_ulens = np.array([len(np.unique(np.floor(d))) for d in data])
    reduced_chi2_model_ulens = chi2_model_ulens / num_days_ulens

    chi2_flat_outside_ulens = stats['chi_squared_outside_level3']
    num_days_outside_ulens = stats['num_days_outside_level3']
    cond_nonzero = num_days_outside_ulens != 0
    reduced_chi2_flat_ulens = chi2_flat_outside_ulens[cond_nonzero] / num_days_outside_ulens[cond_nonzero]

    tE_ulens = stats['tE_level3']
    piE_ulens = np.hypot(stats['piE_E_level3'], stats['piE_N_level3'])

    chi2_model_thresh = np.percentile(reduced_chi2_model_ulens, 95)
    chi2_flat_thresh = np.percentile(reduced_chi2_flat_ulens, 95)
    piE_thresh = np.percentile(piE_ulens, 95)
    tE_thresh = np.percentile(tE_ulens, 95)

    print('piE threshold', piE_thresh)

    #if you want to check what percentile the old threshold is, uncomment below:
    #from scipy import stats
    #print('piE percentile 1.448', stats.percentileofscore(piE_ulens,1.4482240516735567))

    color_cands = '#861388'
    color_cands_cut = '#177E89'
    color_ulens = '#FF6947'

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    ax = ax.flatten()
    for a in ax: a.clear()
    
    fontsize = 20
    density=True
    bins = np.linspace(0, 10, 50)
    ax[0].hist(reduced_chi2_model_cands, bins=bins, histtype='step', color=color_cands, density=density, linewidth=2)
    ax[0].hist(reduced_chi2_model_cands_cut, bins=bins, histtype='step', color=color_cands_cut, density=density, linewidth=2)
    ax[0].hist(reduced_chi2_model_ulens, bins=bins, histtype='step', color=color_ulens, density=density, linewidth=2)
    ax[0].axvline(chi2_model_thresh, color='k', alpha=.5)
    ax[0].set_xlabel(r'$\chi^2_{\rm reduced, model}$', fontsize=fontsize)
    ax[0].set_ylabel('Frequency', fontsize=fontsize)

    bins = np.linspace(0, 10, 50)
    ax[1].hist(reduced_chi2_flat_cands, bins=bins, histtype='step', color=color_cands, density=density, label='ZTF Candidates Level 3', linewidth=2)
    ax[1].hist(reduced_chi2_flat_cands_cut, bins=bins, histtype='step', color=color_cands_cut, density=density, label='ZTF Candidates Level 4', linewidth=2)
    ax[1].hist(reduced_chi2_flat_ulens, bins=bins, histtype='step', color=color_ulens, density=density, label=r'Simulated $\mu$-lens', linewidth=2)
    ax[1].legend(framealpha=1)
    ax[1].axvline(chi2_flat_thresh, color='k', alpha=.5)
    ax[1].set_xlabel(r'$\chi^2_{\rm reduced, flat}$', fontsize=fontsize)
    ax[1].set_ylabel('Frequency', fontsize=fontsize)

    bins = np.logspace(-3, np.log10(30), 50)
    ax[2].hist(piE_cands, bins=bins, histtype='step', color=color_cands, density=density, linewidth=2)
    ax[2].hist(piE_cands_cut, bins=bins, histtype='step', color=color_cands_cut, density=density, linewidth=2)
    ax[2].hist(piE_ulens, bins=bins, histtype='step', color=color_ulens, density=density, linewidth=2)
    ax[2].axvline(piE_thresh, color='k', alpha=.5)
    ax[2].set_xscale('log')
    ax[2].set_xlabel(r'$\pi_E$', fontsize=fontsize)
    ax[2].set_ylabel('Frequency', fontsize=fontsize)

    bins = np.logspace(1, 3, 50)
    ax[3].hist(tE_cands, bins=bins, histtype='step', color=color_cands, density=density, linewidth=2)
    ax[3].hist(tE_cands_cut, bins=bins, histtype='step', color=color_cands_cut, density=density, linewidth=2)
    ax[3].hist(tE_ulens, bins=bins, histtype='step', color=color_ulens, density=density, linewidth=2)
    # ax[3].axvline(tE_thresh, color='k', alpha=.5)
    ax[3].set_xscale('log')
    ax[3].set_xlabel(r'$t_E$ (days)', fontsize=fontsize)
    ax[3].set_ylabel('Frequency', fontsize=fontsize)

    fig.tight_layout()

    fname = '%s/opt_fit_cuts.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.02)
    print('-- %s saved' % fname)
    plt.close(fig)


if __name__ == '__main__':
    plot_opt_fit_cuts()
