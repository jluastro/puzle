#! /usr/bin/env python
"""
plot_example_ulens_lightcurves.py
"""

import numpy as np
import matplotlib.pyplot as plt
from microlens.jlu.model import PSPL_Phot_Par_Param1

from puzle.stats import average_xy_on_round_x
from puzle.ulens import return_ulens_data, \
    return_ulens_stats, return_ulens_metadata
from puzle.utils import return_figures_dir, MJD_start, MJD_finish


def plot_example_lightcurves():
    data = return_ulens_data(observableFlag=True)
    metadata = return_ulens_metadata(observableFlag=True)
    stats = return_ulens_stats(observableFlag=True)

    np.random.seed(8)
    idx_arr = np.random.choice(np.arange(len(data)), replace=False, size=8)
    eta_arr = np.array([stats['eta'][idx] for idx in idx_arr])
    idx_arr = idx_arr[np.argsort(eta_arr)]

    fig, ax = plt.subplots(8, 1, figsize=(10, 8), sharex=True)
    ax = ax.flatten()
    for a in ax: a.clear()
    for i, idx in enumerate(idx_arr):
        hmjd, mag = data[idx][:, :2].T
        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        ax[i].scatter(hmjd, mag, s=100, color='b', marker='.', label='Epochs')
        ax[i].scatter(hmjd_round, mag_round, s=5, color='r', marker='^', label='Nightly Averages')

        model_params = {'t0': metadata['t0'][idx],
                        'u0_amp': metadata['u0'][idx],
                        'tE': metadata['tE'][idx],
                        'piE_E': metadata['piE_E'][idx],
                        'piE_N': metadata['piE_N'][idx],
                        'b_sff': metadata['b_sff'][idx],
                        'mag_src': metadata['mag_src'][idx],
                        'raL': metadata['ra'][idx],
                        'decL': metadata['dec'][idx]}
        model = PSPL_Phot_Par_Param1(**model_params)
        hmjd_model = np.linspace(hmjd.min(), hmjd.max(), 10000)
        mag_model = model.get_photometry(hmjd_model)
        ax[i].plot(hmjd_model, mag_model, color='k', alpha=.6)

        ax[i].invert_yaxis()
        if i == len(idx_arr) - 1:
            ax[i].set_xlabel('heliocentric modified julian date (days)')
        if i == len(idx_arr) // 2:
            ax[i].set_ylabel('magnitude', labelpad=25)
        ax[i].set_xlim(MJD_start, MJD_finish)
        if i in [1, 2, 4, 5, 6, 7]:
            y_coord = .5
        else:
            y_coord = 0
        x_coord = 0.714
        if i == 0:
            leg = ax[i].legend(loc=2, fontsize=12)
            leg.legendHandles[0]._sizes = [200]
            leg.legendHandles[1]._sizes = [100]
        ax[i].annotate(r'$\eta = $ %.2f' % stats['eta'][idx], xy=(x_coord, y_coord),
                       xycoords='axes fraction', fontsize=16,
                       horizontalalignment='right', verticalalignment='bottom')
    fig.subplots_adjust(hspace=0.15, top=.98, bottom=.1, right=.98, left=.11)

    fname = '%s/example_ulens_lightcurves.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


if __name__ == '__main__':
    plot_example_lightcurves()
