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
from puzle.utils import return_figures_dir


def plot_example_lightcurves():
    data = return_ulens_data(observableFlag=True)
    metadata = return_ulens_metadata(observableFlag=True)
    stats = return_ulens_stats(observableFlag=True)

    np.random.seed(8)
    idx_arr = np.random.choice(np.arange(len(data)), replace=False, size=8)
    fig, ax = plt.subplots(2, 4, figsize=(14, 6))
    ax = ax.flatten()
    for a in ax: a.clear()
    for i, idx in enumerate(idx_arr):
        hmjd, mag = data[idx][:, :2].T
        eta = stats['eta'][idx]
        ax[i].set_title(r'$\eta = $ %.2f' % eta)
        ax[i].scatter(hmjd, mag, s=100, color='b', marker='.')
        hmjd_round, mag_round = average_xy_on_round_x(hmjd, mag)
        ax[i].scatter(hmjd_round, mag_round, s=5, color='r', marker='^')

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
        ax[i].set_xlabel('hmjd')
        ax[i].set_ylabel('mag')
    fig.tight_layout()
    fig.suptitle('Simulated Microlensing Lightcurves')
    fig.subplots_adjust(top=.9)

    fname = '%s/example_ulens_lightcurves.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


if __name__ == '__main__':
    plot_example_lightcurves()
